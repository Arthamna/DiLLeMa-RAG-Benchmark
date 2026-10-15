from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .models import DocumentRecord, QARecord

ITS_TOP_LEVEL_QA = {
    "Evaluation - Akademik S1 QA.csv",
    "Evaluation - Akademik S2 QA.csv",
    "Question Evaluation - File KP QA.csv",
}
ITS_EXCLUDED_DIRS = {"per files dataset", "txt files"}


@dataclass(slots=True)
class DatasetBundle:
    documents: list[DocumentRecord]
    questions: list[QARecord]

    @property
    def missing_questions(self) -> list[QARecord]:
        return [question for question in self.questions if not question.available]


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
        for row in _read_csv(
            path,
            {"Judul", "Jenis Dokumen", "Konten 1", "Konten 2", "Topik", "language"},
        ):
            parts = [_clean_text(row.get("Konten 1")), _clean_text(row.get("Konten 2"))]
            text = "\n\n".join(part for part in parts if part)
            if not text:
                continue
            # The source contains meaningful fixed-layout whitespace. Normalize
            # only field boundaries so records that differ in source text are
            # not accidentally collapsed (the canonical fixture yields 53).
            raw_parts = [(row.get("Konten 1") or "").strip(), (row.get("Konten 2") or "").strip()]
            dedupe_text = "\n\n".join(part for part in raw_parts if part)
            content_hash = hashlib.sha256(dedupe_text.encode("utf-8")).hexdigest()
            source_path = path.relative_to(root).as_posix()
            title = _clean_text(row.get("Judul")) or path.stem
            if content_hash in documents:
                current = documents[content_hash]
                for alias in (title, source_path):
                    if alias and alias not in current.aliases:
                        current.aliases.append(alias)
                continue
            documents[content_hash] = DocumentRecord(
                id=f"its-{content_hash[:24]}",
                dataset="its",
                language=_language(row.get("language"), "id"),
                title=title,
                text=text,
                source_path=source_path,
                metadata={
                    "document_type": _clean_text(row.get("Jenis Dokumen")),
                    "topic": _clean_text(row.get("Topik")),
                    "declared_language": _clean_text(row.get("language")),
                },
                aliases=[],
            )

    merged: dict[str, QARecord] = {}
    for path in qa_files:
        for row in _read_csv(path, {"Question", "Expected Answer"}):
            question = _clean_text(row.get("Question") or row.get("question"))
            answer = (row.get("Expected Answer") or "").strip()
            if not question or not answer:
                continue
            key = _identity_text(question)
            if key not in merged:
                merged[key] = QARecord(
                    id=f"its-qa-{_digest(key)}",
                    dataset="its",
                    language="id",
                    question=question,
                    references=[answer],
                    split="all",
                    available=True,
                )
            elif answer not in merged[key].references:
                merged[key].references.append(answer)

    return DatasetBundle(list(documents.values()), list(merged.values()))


def _uniqa_corpus_dir(root: Path, language: str) -> Path:
    candidates = [
        root / f"unipa-corpus-24-{language}",
        root / f"unipa-corpus-{language}",
    ]
    for candidate in candidates:
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
        corpus = _uniqa_corpus_dir(root, language)
        for path in sorted(corpus.rglob("*.txt")):
            text = _clean_text(path.read_text(encoding="utf-8-sig", errors="replace"))
            if not text:
                continue
            relative = path.relative_to(root).as_posix()
            available_names[language].add(path.name)
            documents.append(
                DocumentRecord(
                    id=f"uniqa-{language}-{_digest(relative)}",
                    dataset="uniqa",
                    language=language,
                    title=path.stem,
                    text=text,
                    source_path=relative,
                    metadata={"source_file": path.name},
                    aliases=[path.name],
                )
            )

    questions: list[QARecord] = []
    qa_dir = root / "QA-dataset"
    for language in ("en", "it"):
        split_path = qa_dir / f"qa_splits_{language}.json"
        with split_path.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
        test_rows = payload.get("test")
        if not isinstance(test_rows, list):
            raise ValueError(f"Expected a test list in {split_path}")
        for index, row in enumerate(test_rows):
            missing_fields = {"question", "answer", "context"} - set(row)
            if missing_fields:
                raise ValueError(f"Missing UniQA fields in {split_path} test[{index}]: {sorted(missing_fields)}")
            question = _clean_text(row.get("question"))
            answer = str(row.get("answer") or "").strip()
            sources = [Path(str(name)).name for name in row.get("context", [])]
            available = bool(sources) and any(name in available_names[language] for name in sources)
            questions.append(
                QARecord(
                    id=f"uniqa-{language}-test-{index:04d}",
                    dataset="uniqa",
                    language=language,
                    question=question,
                    references=[answer] if answer else [""],
                    split="test",
                    available=available,
                    source_ids=sources,
                )
            )

    return DatasetBundle(documents, questions)


def load_dataset(dataset: str, source: str | Path) -> DatasetBundle:
    if dataset == "its":
        return load_its(source)
    if dataset == "uniqa":
        return load_uniqa(source)
    raise ValueError(f"Unsupported dataset: {dataset}")

