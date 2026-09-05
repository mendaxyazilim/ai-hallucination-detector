"""
Tests for the self-consistency scoring engine. All test fixtures below are
constructed by hand (not run through any model) so the expected agreement/
disagreement pattern is known in advance -- this is standard unit-testing
practice for a deterministic scoring function, separate from the project's
real demo results (see tests/test_end_to_end.py and results/*.json, which
ARE produced by actually running the local-reference adapter).
"""

from consistency_checker import (
    score_consistency, extract_entities, risk_level_for, aggregate,
    _extract_numbers, _extract_proper_nouns, _tokenize,
)


# ---- entity extraction -----------------------------------------------------

def test_extract_numbers_handles_comma_decimal():
    nums = _extract_numbers("Vücut sıcaklığı 36,6 derecedir, yılda 365 gün vardır.")
    assert "36.6" in nums
    assert "365" in nums


def test_extract_proper_nouns_drops_clause_initial_word():
    # "Görelilik" opens its clause (right after the colon) and should be
    # dropped; "Albert" and "Einstein" are genuinely mid-clause and should
    # survive.
    nouns = _extract_proper_nouns("Bildiğim kadarıyla: Görelilik teorisini Albert Einstein geliştirdi.")
    assert "Albert" in nouns
    assert "Einstein" in nouns
    assert "Görelilik" not in nouns


def test_extract_proper_nouns_survives_comma_clause():
    nouns = _extract_proper_nouns("Elimdeki bilgiye göre, Türkiye'nin başkenti Ankara'dır.")
    assert any("Ankara" in n for n in nouns)
    assert "Elimdeki" not in nouns


def test_tokenize_drops_short_words_and_stopwords():
    toks = _tokenize("Bu bir örnek cümledir ve ile de kısa bir test.")
    assert "bu" not in toks
    assert "ile" not in toks
    assert "örnek" in toks


def test_extract_entities_returns_all_three_kinds():
    e = extract_entities("5 Eylül 2026 tarihinde İstanbul'da 1923 sayısı geçti.")
    assert "5 Eylül 2026" in e.dates
    assert "1923" in e.numbers
    assert any("İstanbul" in n for n in e.proper_nouns)


# ---- scoring: agreement cases ----------------------------------------------

def test_perfect_agreement_scores_very_high():
    samples = [
        "Türkiye Cumhuriyeti 1923 yılında ilan edildi.",
        "Bildiğim kadarıyla, Türkiye Cumhuriyeti 1923 yılında ilan edildi.",
        "Kısaca yanıtlamak gerekirse, Türkiye Cumhuriyeti 1923 yılında ilan edildi.",
        "Türkiye Cumhuriyeti 1923 yılında ilan edildi.",
        "Elimdeki bilgiye göre, Türkiye Cumhuriyeti 1923 yılında ilan edildi.",
    ]
    r = score_consistency(samples)
    assert r.score >= 85
    assert r.number_agreement.applicable
    assert r.number_agreement.distinct_values == 1


def test_multi_number_answer_agreeing_is_not_penalized():
    # A correct answer that legitimately restates several numbers (an
    # arithmetic operation) should score as full agreement when every
    # sample cites the exact same combination -- this guards against the
    # majority-vote-per-value bug where "17, 6, 102" would look like three
    # disagreeing numbers instead of one agreeing answer.
    samples = ["17 çarpı 6, 102 eder."] * 5
    r = score_consistency(samples)
    assert r.number_agreement.distinct_values == 1
    assert r.number_agreement.agreement_pct == 100.0


# ---- scoring: disagreement cases -------------------------------------------

def test_diverging_numbers_scores_low():
    samples = [
        "Kesin olarak söyleyebilirim ki cevap 412.",
        "Bu tamamen net: 1958 yılında.",
        "Hiç kuşkusuz, 89.",
        "Elimdeki bilgiye göre kesinlikle Dr. Elena Voskresenskaya.",
        "Kesin olarak söyleyebilirim ki cevap 3,2 milyar.",
    ]
    r = score_consistency(samples)
    assert r.score <= 45
    assert r.number_agreement.applicable
    assert r.number_agreement.distinct_values > 1


def test_rationale_mentions_disagreement_count_when_numbers_differ():
    samples = ["Cevap 100.", "Cevap 100.", "Cevap 250.", "Cevap 100.", "Cevap 100."]
    r = score_consistency(samples)
    assert "100" in r.rationale
    assert "farklı" in r.rationale


def test_rationale_varies_between_agreeing_and_disagreeing_cases():
    agree = score_consistency(["Cevap 100."] * 5)
    disagree = score_consistency(["Cevap 100.", "Cevap 250.", "Cevap 90.", "Cevap 12.", "Cevap 5."])
    assert agree.rationale != disagree.rationale


# ---- the documented "hedging" blind spot -----------------------------------

def test_consistent_hedging_scores_high_despite_no_information():
    # No sample contains a number, date, or name -- only the "known
    # limitation" branch of the rationale should fire, and because the
    # wording is near-identical across samples, lexical agreement (and
    # therefore the overall score) should still come out high.
    samples = [
        "Bu konuda tam olarak emin değilim, ama genel hatlarıyla bir şey söyleyebilirim.",
        "Bu konuda tam olarak emin değilim, ama genel hatlarıyla bir şey diyebilirim.",
        "Bu konuda tam olarak emin değilim, ama genel hatlarıyla bir şey belirtebilirim.",
        "Bu konuda tam olarak emin değilim, ama genel hatlarıyla bir şey aktarabilirim.",
        "Bu konuda tam olarak emin değilim, ama genel hatlarıyla bir şey ifade edebilirim.",
    ]
    r = score_consistency(samples)
    assert not r.number_agreement.applicable
    assert not r.proper_noun_agreement.applicable
    assert r.score >= 70
    assert "sınırlılık" in r.rationale or "garanti etmediğinin" in r.rationale


# ---- edge cases -------------------------------------------------------------

def test_empty_sample_list_returns_zero():
    r = score_consistency([])
    assert r.score == 0.0
    assert r.n_samples == 0


def test_single_sample_returns_neutral_score():
    r = score_consistency(["Tek bir örnek bu."])
    assert r.score == 50.0
    assert r.n_samples == 1


def test_one_empty_one_nonempty_sample_scores_as_fully_divergent_pair():
    r = score_consistency(["Cevap 100.", ""])
    assert r.lexical_agreement == 0.0


# ---- risk levels & aggregate -------------------------------------------------

def test_risk_level_thresholds():
    assert risk_level_for(90) == "DÜŞÜK HALÜSİNASYON RİSKİ"
    assert risk_level_for(60) == "ORTA HALÜSİNASYON RİSKİ"
    assert risk_level_for(20) == "YÜKSEK HALÜSİNASYON RİSKİ"


def test_aggregate_computes_per_category_and_overall():
    per_prompt = [
        {"category": "aritmetik", "result": {"score": 80.0}},
        {"category": "aritmetik", "result": {"score": 60.0}},
        {"category": "cografya", "result": {"score": 40.0}},
    ]
    agg = aggregate(per_prompt)
    assert agg["n_prompts"] == 3
    assert agg["category_summary"]["aritmetik"]["avg_score"] == 70.0
    assert agg["category_summary"]["cografya"]["avg_score"] == 40.0
    assert agg["overall_reliability_score"] == round((80 + 60 + 40) / 3, 1)
