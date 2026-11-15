from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .grpc_service import serve_json
from .indexing import choose_nlist, clamp_nprobe


@dataclass
class LoadedIndex:
    index: Any
    chunks: Any
    records: list[dict[str, Any]]
    tokenizer: Any
    embedder: Any
    nlist: int
    profile: dict[str, Any]
    metadata: dict[str, Any]


class RetrievalService:
    def __init__(self, protocol_version: int = 1):
        self.protocol_version = protocol_version
        self.indexes: dict[str, LoadedIndex] = {}
        self.lock = threading.RLock()

    def _response(self, **value):
        return {"protocol_version": self.protocol_version, "status": "ok", **value}

    def _check(self, request: dict[str, Any]) -> None:
        if int(request.get("protocol_version", -1)) != self.protocol_version:
            raise ValueError("protocol version mismatch")

    def health(self, request: dict[str, Any]) -> dict[str, Any]:
        self._check(request)
        try:
            import faiss
            import sentence_transformers
            versions = {
                "faiss": getattr(faiss, "__version__", "unknown"),
                "sentence_transformers": sentence_transformers.__version__,
                "python": platform.python_version(),
                "image": os.getenv("BENCHMARK_IMAGE_VERSION", "development"),
            }
        except ImportError as exc:
            return {"protocol_version": self.protocol_version, "status": "error", "error": str(exc)}
        return self._response(role="retrieval", device="cpu", indexes={
            index_id: {
                "index_fingerprint": loaded.metadata.get("index_fingerprint"),
                "nlist": loaded.nlist,
                "nprobe": int(loaded.index.nprobe),
                "profile_hardware_fingerprint": loaded.profile.get("hardware_fingerprint"),
                "profile_model_fingerprint": loaded.profile.get("model_fingerprint"),
            }
            for index_id, loaded in sorted(self.indexes.items())
        }, versions=versions)

    @staticmethod
    def _chunk_documents(documents: list[dict], tokenizer, chunk_size: int):
        import numpy as np

        chunks: list[list[int]] = []
        records: list[dict[str, Any]] = []
        pad_id = int(tokenizer.pad_token_id or 0)
        for document in documents:
            token_ids = tokenizer.encode(str(document.get("text") or ""), add_special_tokens=True)
            if not token_ids:
                continue
            for offset in range(0, len(token_ids), chunk_size):
                value = token_ids[offset: offset + chunk_size]
                value += [pad_id] * (chunk_size - len(value))
                ordinal = offset // chunk_size
                chunks.append(value)
                records.append({
                    "source_id": document["id"],
                    "chunk_id": f"{document['id']}:retro:{ordinal}",
                    "ordinal": ordinal,
                    "text": tokenizer.decode(value, skip_special_tokens=True),
                    "source_path": document.get("source_path", ""),
                    "title": document.get("title", ""),
                    "aliases": document.get("aliases", []),
                    "source_file": (document.get("metadata") or {}).get("source_file", ""),
                })
        if not chunks:
            raise ValueError("Cannot build a RETRO index from an empty corpus")
        return np.asarray(chunks, dtype=np.int64), records

    @staticmethod
    def _target(storage_root: str, index_id: str) -> Path:
        digest = hashlib.sha256(index_id.encode()).hexdigest()[:24]
        return Path(storage_root or "/data/indexes") / digest

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _load(self, index_id: str, target: Path, tokenizer_name: str, retriever_model: str) -> dict[str, Any]:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer
        from transformers import AutoTokenizer

        manifest_path = target / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Index manifest not found: {manifest_path}")
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        if metadata.get("index_id") != index_id:
            raise ValueError("index fingerprint mismatch")
        for filename, expected in (metadata.get("file_sha256") or {}).items():
            if self._sha256(target / filename) != expected:
                raise ValueError(f"index fingerprint mismatch for {filename}")
        tokenizer_name = str(metadata.get("tokenizer") or tokenizer_name)
        retriever_model = str(metadata.get("retriever_model") or retriever_model)
        index = faiss.read_index(str(target / "index.faiss"))
        chunks = np.load(target / "chunks.npy", mmap_mode="r")
        records = json.loads((target / "records.json").read_text(encoding="utf-8"))
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        embedder = SentenceTransformer(retriever_model, device="cpu")
        profile = {}
        profile_path = target / "profile.json"
        if profile_path.is_file():
            cached = json.loads(profile_path.read_text(encoding="utf-8"))
            if cached.get("index_fingerprint") == metadata.get("index_fingerprint"):
                profile = dict(cached.get("profile") or {})
        loaded = LoadedIndex(index, chunks, records, tokenizer, embedder, int(metadata["nlist"]), profile, metadata)
        with self.lock:
            self.indexes[index_id] = loaded
        return metadata

    def load_index(self, request: dict[str, Any]) -> dict[str, Any]:
        self._check(request)
        index_id = str(request.get("index_id") or "")
        if not index_id:
            raise ValueError("index_id is required")
        with self.lock:
            current = self.indexes.get(index_id)
        if current is not None:
            return self._response(index={
                "index_id": index_id, "chunks": len(current.records), "nlist": current.nlist, "cache": "memory",
            })
        target = self._target(str(request.get("storage_root") or "/data/indexes"), index_id)
        metadata = self._load(
            index_id, target, str(request.get("tokenizer") or "t5-base"),
            str(request.get("retriever_model") or "sentence-transformers/all-MiniLM-L6-v2"),
        )
        return self._response(index={**metadata, "cache": "disk"})

    def build_index(self, request: dict[str, Any]) -> dict[str, Any]:
        self._check(request)
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer
        from transformers import AutoTokenizer

        index_id = str(request.get("index_id") or "")
        if not index_id:
            raise ValueError("index_id is required")
        chunk_size = int(request.get("chunk_size", 64))
        tokenizer_name = str(request.get("tokenizer") or "t5-base")
        retriever_model = str(request.get("retriever_model") or "sentence-transformers/all-MiniLM-L6-v2")
        target = self._target(str(request.get("storage_root") or "/data/indexes"), index_id)
        if (target / "manifest.json").is_file():
            metadata = self._load(index_id, target, tokenizer_name, retriever_model)
            return self._response(index={**metadata, "cache": "disk"})
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        chunks, records = self._chunk_documents(list(request.get("documents") or []), tokenizer, chunk_size)
        texts = [item["text"] for item in records]
        embedder = SentenceTransformer(retriever_model, device="cpu")
        embeddings = embedder.encode(
            texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False,
        ).astype("float32")
        nlist = choose_nlist(len(records))
        quantizer = faiss.IndexFlatL2(embeddings.shape[1])
        index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1], nlist, faiss.METRIC_L2)
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = clamp_nprobe(16, nlist)
        target.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(target / "index.faiss"))
        np.save(target / "chunks.npy", chunks)
        (target / "records.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
        metadata = {
            "index_id": index_id, "chunks": len(records), "documents": len(request.get("documents") or []),
            "nlist": nlist, "dimension": int(embeddings.shape[1]), "path": str(target),
            "track": request.get("track"), "tokenizer": tokenizer_name,
            "retriever_model": retriever_model,
        }
        metadata["file_sha256"] = {
            name: self._sha256(target / name) for name in ("index.faiss", "chunks.npy", "records.json")
        }
        metadata["index_fingerprint"] = hashlib.sha256(
            json.dumps({key: metadata[key] for key in ("index_id", "chunks", "nlist", "dimension", "file_sha256")},
                       sort_keys=True).encode()
        ).hexdigest()
        (target / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        with self.lock:
            self.indexes[index_id] = LoadedIndex(index, chunks, records, tokenizer, embedder, nlist, {}, metadata)
        return self._response(index=metadata)

    def build_index_stream(self, requests) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        documents = []
        for request in requests:
            self._check(request)
            if not merged:
                merged = {key: value for key, value in request.items() if key != "documents"}
            elif request.get("index_id") != merged.get("index_id"):
                raise ValueError("streamed index_id changed during BuildIndex")
            documents.extend(request.get("documents") or [])
        if not merged:
            raise ValueError("BuildIndex stream is empty")
        merged["documents"] = documents
        return self.build_index(merged)

    def set_profile(self, request: dict[str, Any]) -> dict[str, Any]:
        self._check(request)
        index = self.indexes[str(request["index_id"])]
        index.profile = dict(request.get("profile") or {})
        target = Path(str(index.metadata["path"]))
        (target / "profile.json").write_text(json.dumps({
            "index_fingerprint": index.metadata.get("index_fingerprint"),
            "profile": index.profile,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return self._response(index_id=request["index_id"], profile=index.profile)

    def set_nprobe(self, request: dict[str, Any]) -> dict[str, Any]:
        self._check(request)
        loaded = self.indexes[str(request["index_id"])]
        value = clamp_nprobe(int(request.get("nprobe", 16)), loaded.nlist)
        with self.lock:
            loaded.index.nprobe = value
        return self._response(index_id=request["index_id"], nprobe=value, nlist=loaded.nlist)

    def profile_nprobe(self, request: dict[str, Any]) -> dict[str, Any]:
        self._check(request)
        loaded = self.indexes[str(request["index_id"])]
        queries = list(request.get("queries") or [item["text"] for item in loaded.records[:8]])
        if not queries:
            raise ValueError("Cannot profile an empty index")
        embedded = loaded.embedder.encode(queries, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        repetitions = max(1, int(request.get("repetitions", 5)))
        values = sorted({clamp_nprobe(int(value), loaded.nlist) for value in request.get("nprobes", (1, 2, 4, 8, 16, 32, 64))})
        timings = {}
        for nprobe in values:
            samples = []
            for _ in range(repetitions):
                started = time.perf_counter()
                with self.lock:
                    loaded.index.nprobe = nprobe
                    loaded.index.search(embedded, 2)
                samples.append((time.perf_counter() - started) * 1000)
            timings[str(nprobe)] = sum(samples) / len(samples)
        return self._response(index_id=request["index_id"], nlist=loaded.nlist, search_latency_ms=timings)

    @staticmethod
    def _dynamic_nprobe(loaded: LoadedIndex, request: dict[str, Any]) -> int:
        profile = loaded.profile
        decoder = list(profile.get("decoder_latency_ms") or [])
        interval = int(request.get("interval", 64))
        seq_len = int(request.get("seq_len", 64))
        if not decoder:
            return clamp_nprobe(int(request.get("nprobe", 16)), loaded.nlist)
        budget = sum(float(decoder[min(i, len(decoder) - 1)]) for i in range(seq_len, seq_len + interval))
        budget -= float(profile.get("embedding_latency_ms", 0.0)) + float(profile.get("extra_overhead_ms", 10.0))
        budget *= float(profile.get("search_latency_budget_discount", 1.0))
        per_probe = max(float(profile.get("search_latency_ms_per_nprobe", 0.01)), 1e-9)
        value = int(max(1.0, budget) / per_probe)
        value = max(int(profile.get("min_nprobe", 1)), min(int(profile.get("max_nprobe", 64)), value))
        return clamp_nprobe(value, loaded.nlist)

    def retrieve(self, request: dict[str, Any]) -> dict[str, Any]:
        self._check(request)
        import numpy as np

        loaded = self.indexes.get(str(request.get("index_id") or ""))
        if loaded is None:
            raise KeyError(f"Index is not loaded: {request.get('index_id')}")
        queries = [str(item) for item in request.get("queries") or []]
        if not queries:
            raise ValueError("queries cannot be empty")
        started = time.perf_counter()
        embedded = loaded.embedder.encode(queries, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        embedding_ms = (time.perf_counter() - started) * 1000
        nprobe = self._dynamic_nprobe(loaded, request) if request.get("dynamic_nprobe") else clamp_nprobe(int(request.get("nprobe", 16)), loaded.nlist)
        with self.lock:
            loaded.index.nprobe = nprobe
        neighbours = int(request.get("num_neighbours", 2))
        searched = time.perf_counter()
        with self.lock:
            distances, indices = loaded.index.search(embedded, neighbours)
        search_ms = (time.perf_counter() - searched) * 1000
        continuation = int(request.get("continuation_chunks", 1))
        stale_offset = int(request.get("stale_steps", 0))
        chunk_size = int(request.get("chunk_size", 64))
        neighbour_len = chunk_size * (1 + continuation)
        results = []
        pad_id = int(loaded.tokenizer.pad_token_id or 0)
        for row_distances, row_indices in zip(distances, indices):
            hits = []
            for distance, position in zip(row_distances, row_indices):
                if position < 0:
                    continue
                record = loaded.records[int(position)]
                token_values: list[int] = []
                needed = 1 + continuation + (1 if stale_offset else 0)
                for step in range(needed):
                    candidate = int(position) + step
                    if candidate < len(loaded.records) and loaded.records[candidate]["source_id"] == record["source_id"]:
                        token_values.extend(int(value) for value in loaded.chunks[candidate])
                    else:
                        token_values.extend([pad_id] * chunk_size)
                token_values = token_values[stale_offset: stale_offset + neighbour_len]
                token_values += [pad_id] * (neighbour_len - len(token_values))
                hits.append({
                    "token_ids": token_values,
                    "source_id": record["source_id"],
                    "chunk_id": record["chunk_id"],
                    "text": record["text"],
                    "score": float(-distance),
                    "metadata": {key: record.get(key) for key in ("source_path", "title", "aliases", "source_file")},
                })
            results.append({"hits": hits})
        return self._response(results=results, diagnostics={
            "embedding_latency_ms": round(embedding_ms, 3),
            "search_latency_ms": round(search_ms, 3),
            "nprobe": nprobe,
            "nlist": loaded.nlist,
        })


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--protocol-version", type=int, default=1)
    args = parser.parse_args(argv)
    service = RetrievalService(args.protocol_version)
    serve_json("rag.RetrievalService", {
        "Health": service.health,
        "LoadIndex": service.load_index,
        "Retrieve": service.retrieve,
        "SetProfile": service.set_profile,
        "SetNprobe": service.set_nprobe,
        "ProfileNprobe": service.profile_nprobe,
    }, args.host, args.port, stream_methods={"BuildIndex": service.build_index_stream})


if __name__ == "__main__":
    main()
