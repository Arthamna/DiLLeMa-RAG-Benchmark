import csv
import json
from collections import Counter
from pathlib import Path

import pytest

from rag_benchmark.shared.datasets import load_its, load_uniqa


def _csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_its_schema_deduplication_and_multireference(tmp_path):
    category = tmp_path / "akademik"
    _csv(
        category / "Dataset IF-Tegration - akademik.csv",
        ["Judul", "Jenis Dokumen", "Konten 1", "Konten 2", "Topik", "language"],
        [
            {"Judul": "Panduan α", "Jenis Dokumen": "Panduan", "Konten 1": "Isi Unicode café", "Konten 2": "Bagian dua", "Topik": "Uji", "language": "id"},
            {"Judul": "Alias", "Jenis Dokumen": "Panduan", "Konten 1": "Isi Unicode café", "Konten 2": "Bagian dua", "Topik": "Uji", "language": "id"},
        ],
    )
    _csv(
        category / "QA Generator - akademik.csv",
        ["Question", "Expected Answer"],
        [
            {"Question": "Apa isi panduan?", "Expected Answer": "Jawaban pertama yang cukup panjang."},
            {"Question": "Apa isi panduan?", "Expected Answer": "Jawaban alternatif yang juga benar."},
        ],
    )
    bundle = load_its(tmp_path)
    assert len(bundle.documents) == 1
    assert "café" in bundle.documents[0].text
    assert "Alias" in bundle.documents[0].aliases
    assert len(bundle.questions) == 1
    assert bundle.questions_for_split("all") == bundle.questions
    assert len(bundle.questions[0].references) == 2
    assert bundle.questions[0].split == "train"
    assert bundle.questions[0].source_ids == [bundle.documents[0].id]
    assert bundle.questions[0].metadata["relevance_label"] == "weak_category"


def test_its_rejects_missing_schema_columns(tmp_path):
    _csv(
        tmp_path / "broken" / "Dataset IF-Tegration - broken.csv",
        ["Judul", "Konten 1"],
        [{"Judul": "Incomplete", "Konten 1": "text"}],
    )
    with pytest.raises(ValueError, match="Missing CSV columns"):
        load_its(tmp_path)


def test_uniqa_native_test_and_missing_sources(tmp_path):
    for language in ("en", "it"):
        corpus = tmp_path / f"unipa-corpus-24-{language}" / "course-info"
        corpus.mkdir(parents=True)
        (corpus / f"present_{language}.txt").write_text("Café naïve — contenuto.", encoding="utf-8")
        payload = {
            "train": [],
            "test": [
                {"question": f"question {language}", "context": [f"present_{language}.txt"], "answer": "four word reference answer"},
                {"question": f"missing {language}", "context": [f"absent_{language}.txt"], "answer": "another four word answer"},
            ],
        }
        qa = tmp_path / "QA-dataset"
        qa.mkdir(exist_ok=True)
        (qa / f"qa_splits_{language}.json").write_text(json.dumps(payload), encoding="utf-8")
    bundle = load_uniqa(tmp_path)
    assert Counter(document.language for document in bundle.documents) == {"en": 1, "it": 1}
    assert Counter(question.language for question in bundle.missing_questions) == {"en": 1, "it": 1}
    assert all(question.split == "test" for question in bundle.questions)


def test_canonical_dataset_counts():
    family = Path(__file__).resolve().parents[2]
    its_root = family / "langchain-chatbot-main" / "dataset"
    uniqa_root = family / "UniQA-main"
    if not its_root.is_dir() or not uniqa_root.is_dir():
        pytest.skip("Canonical sibling datasets are not present")
    its = load_its(its_root)
    assert (len(its.documents), len(its.questions)) == (53, 340)
    assert sum(len(question.references) for question in its.questions) == 345
    assert Counter(question.split for question in its.questions) == {"train": 271, "test": 69}
    assert len(its.questions_for_split("all")) == 340
    assert not ({q.id for q in its.questions_for_split("train")} & {q.id for q in its.questions_for_split("test")})
    uniqa = load_uniqa(uniqa_root)
    assert Counter(document.language for document in uniqa.documents) == {"en": 522, "it": 521}
    assert Counter((question.language, question.split) for question in uniqa.questions) == {
        ("en", "train"): 5298,
        ("it", "train"): 5298,
        ("en", "test"): 1573,
        ("it", "test"): 1573,
    }
    assert Counter(question.language for question in uniqa.questions_for_split("test") if not question.available) == {"en": 42, "it": 14}

