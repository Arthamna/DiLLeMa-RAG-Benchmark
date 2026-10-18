from __future__ import annotations

from pathlib import Path

from ...metrics import write_json
from ...shared.local_llm import rrag_prompt
from ...shared.records import QARecord
from .r2former import R2AGAdapter


def train_rrag_language(
    pipeline,
    runtime,
    questions: list[QARecord],
    collection: str,
    output: str | Path,
    language: str,
) -> dict:
    import torch

    config = pipeline.rrag_config
    adapter = R2AGAdapter(config, runtime.hidden_size).to(runtime.device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=config.learning_rate)
    stats = {
        "scheduled": len(questions),
        "trained_examples": 0,
        "optimizer_examples": 0,
        "missing_context": 0,
        "no_positive_in_top_k": 0,
        "prompt_truncated": 0,
        "epochs": config.epochs,
    }
    optimizer.zero_grad(set_to_none=True)
    accumulated = 0
    trained_ids: set[str] = set()
    missing_ids: set[str] = set()
    no_positive_ids: set[str] = set()
    truncated_ids: set[str] = set()
    adapter.train()
    for _epoch in range(config.epochs):
        for question in questions:
            if not question.available:
                missing_ids.add(question.id)
                continue
            context = pipeline.retrieve(collection, question.question)
            labels = pipeline.labels_for(context, question.source_ids)
            if not labels or not any(labels):
                no_positive_ids.add(question.id)
                continue
            prompt = rrag_prompt(question.question, context.evidence)
            tokens, lm_labels = runtime.training_tensors(prompt, question.references[0])
            features = torch.tensor(context.features, device=runtime.device).unsqueeze(0)
            relevance_labels = torch.tensor(labels, device=runtime.device).unsqueeze(0)
            injected, _logits, relevance_loss = adapter(features, relevance_labels)
            try:
                inputs_embeds = runtime.inject(tokens.input_ids, injected)
            except ValueError:
                truncated_ids.add(question.id)
                continue
            outputs = runtime.model(
                inputs_embeds=inputs_embeds,
                attention_mask=tokens.attention_mask,
                labels=lm_labels,
            )
            loss = (outputs.loss + relevance_loss) / config.gradient_accumulation_steps
            loss.backward()
            accumulated += 1
            trained_ids.add(question.id)
            stats["optimizer_examples"] += 1
            if accumulated % config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
    if accumulated % config.gradient_accumulation_steps:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    stats["trained_examples"] = len(trained_ids)
    stats["missing_context"] = len(missing_ids)
    stats["no_positive_in_top_k"] = len(no_positive_ids)
    stats["prompt_truncated"] = len(truncated_ids)
    if not trained_ids:
        raise RuntimeError(f"No trainable {language} examples had a positive source in RRAG top-k")

    target = Path(output) / language
    checkpoint = adapter.save_checkpoint(target, runtime.fingerprint, language)
    manifest = {
        **stats,
        "language": language,
        "collection": collection,
        "checkpoint": str(checkpoint.resolve()),
        "model_fingerprint": runtime.fingerprint,
        "base_llm_frozen": True,
        "retriever_frozen": True,
    }
    write_json(target / "training-manifest.json", manifest)
    return manifest

