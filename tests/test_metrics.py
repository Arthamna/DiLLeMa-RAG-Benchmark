from rag_benchmark.metrics import bleu_variant, normalize_content, summarize


def _row(prediction, references, dataset="its", language="id", available=True):
    return {
        "dataset": dataset,
        "language": language,
        "prediction": prediction,
        "references": references,
        "status": "ok" if prediction else "error",
        "available": available,
    }


def test_bleu_identical_empty_and_multireference():
    identical = [_row("this is an exact reference answer", ["this is an exact reference answer"])]
    assert bleu_variant(identical)["score"] == 100.0
    assert bleu_variant([_row("", ["this is an exact reference answer"])])["score"] == 0.0
    multi = [_row("this is the accepted alternate answer", ["wrong primary answer has words", "this is the accepted alternate answer"])]
    assert bleu_variant(multi)["score"] == 100.0


def test_content_normalization_removes_urls_and_fixed_boilerplate():
    value = "The course is open. For more information visit https://example.test/course."
    assert normalize_content(value) == "The course is open."
    raw = bleu_variant([_row(value, ["The course is open."])])["score"]
    content = bleu_variant([_row(value, ["The course is open."])], content_only=True)["score"]
    assert content > raw


def test_uniqa_strict_and_valid_denominators():
    rows = [
        _row("one two three four", ["one two three four"], "uniqa", "en", True),
        _row("", ["missing context answer here"], "uniqa", "en", False),
        _row("uno due tre quattro", ["uno due tre quattro"], "uniqa", "it", True),
    ]
    report = summarize(rows)
    assert report["strict"]["en"]["count"] == 2
    assert report["corpus_valid"]["en"]["count"] == 1
    assert "macro_average" in report["strict"]

