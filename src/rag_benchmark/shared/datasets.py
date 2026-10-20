from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .records import DocumentRecord, QARecord

ITS_TOP_LEVEL_QA = {
    "Evaluation - Akademik S1 QA.csv": "akademik s1",
    "Evaluation - Akademik S2 QA.csv": "akademik s2",
    "Question Evaluation - File KP QA.csv": "kerja praktik",
}
ITS_EXCLUDED_DIRS = {"per files dataset", "txt files"}


@dataclass(slots=True)
class DatasetBundle:
    documents: list[DocumentRecord]
    questions: list[QARecord]

    @property
    def missing_questions(self) -> list[QARecord]:
        return [question for question in self.questions if not question.available]

    def questions_for_split(self, split: str = "test") -> list[QARecord]:
        if split == "all":
            return list(self.questions)
        return [question for question in self.questions if question.split == split]


def _clean_text(value: str | None) -> str:
    text = (value or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def _identity_text(value: str) -> str:
    return " ".join(value.split()).casefold()


def _digest(value: str, length: int = 24) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _language(value: str | None, default: str) -> str:
    normalized = (value or "").strip().casefold()
    if normalized.startswith(("eng", "en")):
        return "en"
    if normalized.startswith(("ita", "it")):
        return "it"
    if normalized.startswith(("ind", "id", "bahasa")):
        return "id"
    return default


def _read_csv(path: Path, required: set[str] | None = None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = (required or set()) - fields
        if missing:
            raise ValueError(f"Missing CSV columns in {path}: {sorted(missing)}")
        return list(reader)


def _assign_its_splits(questions: list[QARecord]) -> None:
    strata: dict[str, list[QARecord]] = defaultdict(list)
    for question in questions:
        categories = list(question.metadata.get("categories") or ["unknown"])
        strata[sorted(categories)[0]].append(question)
    for category, items in strata.items():
        ordered = sorted(items, key=lambda item: _digest(f"its-split-v1:{category}:{item.id}", 64))
        test_count = 0 if len(ordered) == 1 else max(1, round(len(ordered) * 0.2))
        test_ids = {item.id for item in ordered[:test_count]}
        for item in items:
            item.split = "test" if item.id in test_ids else "train"


def load_its(source: str | Path) -> DatasetBundle:
    root = Path(source).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"ITS dataset directory not found: {root}")
    corpus_files: list[Path] = []
    qa_files: list[Path] = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        if directory.name.casefold() in ITS_EXCLUDED_DIRS:
            continue
        corpus_files.extend(sorted(directory.glob("Dataset IF-Tegration - *.csv")))
        qa_files.extend(sorted(directory.glob("QA Generator*.csv")))
    qa_files.extend(root / name for name in sorted(ITS_TOP_LEVEL_QA) if (root / name).is_file())

    documents: dict[str, DocumentRecord] = {}
    for path in corpus_files:
        category = path.parent.name.casefold()
        for row in _read_csv(path, {"Judul", "Jenis Dokumen", "Konten 1", "Konten 2", "Topik", "language"}):
            parts = [_clean_text(row.get("Konten 1")), _clean_text(row.get("Konten 2"))]
            text = "\n\n".join(part for part in parts if part)
            if not text:
                continue
            raw_parts = [(row.get("Konten 1") or "").strip(), (row.get("Konten 2") or "").strip()]
            content_hash = hashlib.sha256("\n\n".join(part for part in raw_parts if part).encode("utf-8")).hexdigest()
            source_path = path.relative_to(root).as_posix()
            title = _clean_text(row.get("Judul")) or path.stem
            if content_hash in documents:
                current = documents[content_hash]
                for alias in (title, source_path):
                    if alias and alias not in current.aliases:
                        current.aliases.append(alias)
                categories = current.metadata.setdefault("categories", [])
                if category not in categories:
                    categories.append(category)
                continue
            documents[content_hash] = DocumentRecord(
                id=f"its-{content_hash[:24]}", dataset="its",
                language=_language(row.get("language"), "id"), title=title, text=text,
                source_path=source_path,
                metadata={
                    "document_type": _clean_text(row.get("Jenis Dokumen")),
                    "topic": _clean_text(row.get("Topik")),
                    "declared_language": _clean_text(row.get("language")),
                    "categories": [category],
                }, aliases=[],
            )

    category_sources: dict[str, set[str]] = defaultdict(set)
    for document in documents.values():
        for category in document.metadata.get("categories", []):
            category_sources[str(category)].add(document.id)

    merged: dict[str, QARecord] = {}
    for path in qa_files:
        category = ITS_TOP_LEVEL_QA.get(path.name, path.parent.name.casefold())
        for row in _read_csv(path, {"Question", "Expected Answer"}):
            question = _clean_text(row.get("Question") or row.get("question"))
            answer = (row.get("Expected Answer") or "").strip()
            if not question or not answer:
                continue
            key = _identity_text(question)
            if key not in merged:
                merged[key] = QARecord(
                    id=f"its-qa-{_digest(key)}", dataset="its", language="id",
                    question=question, references=[answer], split="train", available=True,
                    metadata={"categories": [category], "relevance_label": "weak_category"},
                )
            else:
                record = merged[key]
                if answer not in record.references:
                    record.references.append(answer)
                if category not in record.metadata["categories"]:
                    record.metadata["categories"].append(category)

    questions = list(merged.values())
    for question in questions:
        question.source_ids = sorted({
            source_id for category in question.metadata["categories"]
            for source_id in category_sources.get(category, set())
        })
    _assign_its_splits(questions)
    return DatasetBundle(list(documents.values()), questions)


def _uniqa_corpus_dir(root: Path, language: str) -> Path:
    for candidate in (root / f"unipa-corpus-24-{language}", root / f"unipa-corpus-{language}"):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"UniQA {language} corpus directory not found under {root}")


def load_uniqa(source: str | Path) -> DatasetBundle:
    root = Path(source).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"UniQA directory not found: {root}")
    documents: list[DocumentRecord] = []
    available_names: dict[str, set[str]] = {"en": set(), "it": set()}
    for language in ("en", "it"):
        for path in sorted(_uniqa_corpus_dir(root, language).rglob("*.txt")):
            text = _clean_text(path.read_text(encoding="utf-8-sig", errors="replace"))
            if not text:
                continue
            relative = path.relative_to(root).as_posix()
            available_names[language].add(path.name)
            documents.append(DocumentRecord(
                id=f"uniqa-{language}-{_digest(relative)}", dataset="uniqa", language=language,
                title=path.stem, text=text, source_path=relative,
                metadata={"source_file": path.name}, aliases=[path.name],
            ))

    questions: list[QARecord] = []
    qa_dir = root / "QA-dataset"
    for language in ("en", "it"):
        split_path = qa_dir / f"qa_splits_{language}.json"
        with split_path.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
        for split in ("train", "test"):
            rows = payload.get(split)
            if not isinstance(rows, list):
                raise ValueError(f"Expected a {split} list in {split_path}")
            for index, row in enumerate(rows):
                missing_fields = {"question", "answer", "context"} - set(row)
                if missing_fields:
                    raise ValueError(f"Missing UniQA fields in {split_path} {split}[{index}]: {sorted(missing_fields)}")
                sources = [Path(str(name)).name for name in row.get("context", [])]
                available = bool(sources) and any(name in available_names[language] for name in sources)
                answer = str(row.get("answer") or "").strip()
                questions.append(QARecord(
                    id=f"uniqa-{language}-{split}-{index:04d}", dataset="uniqa", language=language,
                    question=_clean_text(row.get("question")), references=[answer] if answer else [""],
                    split=split, available=available, source_ids=sources,
                    metadata={"relevance_label": "native_source"},
                ))
    return DatasetBundle(documents, questions)


def load_dataset(dataset: str, source: str | Path) -> DatasetBundle:
    if dataset == "its":
        return load_its(source)
    if dataset == "uniqa":
        return load_uniqa(source)
    raise ValueError(f"Unsupported dataset: {dataset}")

