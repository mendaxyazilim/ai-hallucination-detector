"""
consistency_checker.py
-----------------------
The detector's scoring engine, implementing a self-consistency-based
hallucination risk check inspired by the general idea behind SelfCheckGPT
(Manakul, Liusie & Gales, 2023): sample a model N times on the same factual
question with stochastic (temperature > 0) decoding, and use how much the N
samples AGREE WITH EACH OTHER as a zero-resource proxy for whether the model
actually "knows" the answer. No ground truth is used or required anywhere in
this file -- the whole point of the method is that it works even when you
have no reference answer to compare against, which is what makes it usable
against a live model on questions nobody has hand-labeled.

This is a genuinely independent implementation, not a port: the general
inter-sample-agreement idea is the only thing borrowed from the published
technique; the entity extraction, scoring formula, and rationale text below
are original to this project.

Like classifiers.py in the sister ai_safety_auditor project, this is
deliberately NOT an LLM-as-judge: every signal below is a transparent,
regex/set-based computation you can trace by hand from the raw samples. That
transparency has the same trade-off documented there -- it can miss forms of
agreement or disagreement it wasn't written to notice -- and it has an
additional, more fundamental limitation specific to this method, spelled out
in DOCUMENTED LIMITATION below.

Scoring approach
----------------
For a batch of N sampled responses to the same prompt:

1. Lexical agreement -- average pairwise Jaccard similarity of the samples'
   content-word sets. Two samples that reuse mostly the same words score
   high here regardless of whether either one is correct.

2. Numeric agreement -- extract every number (including bare years) each
   sample mentions. If at least one sample mentions a number, the samples
   are scored on what fraction of them agree with the single most common
   number mentioned. Samples proposing a different number for what looks
   like the same question are exactly the "one model, several answers"
   signal self-consistency checking is built to surface.

3. Proper-noun agreement -- same idea, over capitalized name-like tokens
   (people, places, institutions).

The three signals are combined into one 0-100 "Güvenilirlik Skoru"
(reliability score) with fixed weights, EXCEPT that a signal with nothing to
measure (e.g. no sample contains any number) is dropped and the remaining
weights are rescaled -- so the score is never artificially dragged down by
entity types that were never going to appear in this particular prompt's
answers (e.g. a definition question rarely produces a proper noun).

DOCUMENTED LIMITATION (read this before trusting the score)
-------------------------------------------------------------
This method measures AGREEMENT, not TRUTH. A model that hallucinates a
different specific fact on every sample will (correctly) score low. But a
model that gives the exact same non-answer every time -- "I'm not sure, but
roughly..." -- will score HIGH, because there is nothing in any sample for
the other samples to disagree with. High self-consistency is necessary but
not sufficient evidence of reliability; a consistently vague or consistently
wrong-but-repeated answer both look identical to this method. The
`local_reference_model.py` "hedging" configuration in this project's own
demo run exists specifically to make this failure mode visible and
measurable rather than leaving it as a theoretical caveat -- see README.md
and results/hedging.json.
"""

from __future__ import annotations

import dataclasses
import itertools
import re
from collections import Counter
from typing import Dict, List, Optional

# A short, original Turkish stopword list -- purely functional (not lifted
# from any library), used only to keep lexical-overlap comparisons from being
# dominated by grammatical filler words.
TURKISH_STOPWORDS = {
    "bir", "bu", "şu", "ve", "ile", "de", "da", "ki", "mi", "mu", "mü", "mı",
    "gibi", "için", "ama", "fakat", "ancak", "veya", "ya", "ise", "olan",
    "olarak", "kadar", "daha", "çok", "en", "her", "ne", "nasıl", "niçin",
    "neden", "diye", "göre", "üzere", "bazı", "tüm", "hiç", "değil", "yok",
    "var", "oldu", "olur", "olan", "onun", "ona", "ondan", "biz", "siz",
    "ben", "sen", "o", "bunun", "buna", "bundan", "bunları", "kendi",
}

# A 4-digit token in this range is treated as a plausible calendar year (and
# therefore also folded into "numbers" -- a wrong year IS a wrong number).
_YEAR_MIN, _YEAR_MAX = 1000, 2099

_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")
_TURKISH_MONTHS = ("Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık")
_DATE_RE = re.compile(rf"\b\d{{1,2}}\s+(?:{_TURKISH_MONTHS})\s+\d{{4}}\b")
_PROPER_NOUN_RE = re.compile(r"[A-ZÇĞİIÖŞÜ][a-zçğıöşü]+(?:'[a-zçğıöşü]+)?")


def _normalize(text: str) -> str:
    """See model_adapters/local_reference_model for why plain str.lower() is
    unsafe on Turkish text containing capital İ/I."""
    return text.replace("İ", "i").replace("I", "ı").lower()


def _tokenize(text: str) -> set:
    words = re.findall(r"[a-zçğıöşü0-9]+", _normalize(text))
    return {w for w in words if len(w) > 2 and w not in TURKISH_STOPWORDS}


def _extract_numbers(text: str) -> List[str]:
    """Every numeric token, normalized so '3,2' and '3.2' compare equal."""
    return [m.replace(",", ".") for m in _NUMBER_RE.findall(text)]


def _extract_dates(text: str) -> List[str]:
    """Full 'DD Ay YYYY' style dates only (bare years already show up via
    _extract_numbers -- that overlap is intentional: a bare-year disagreement
    IS a numeric disagreement)."""
    return _DATE_RE.findall(text)


_CLAUSE_SPLIT_RE = re.compile(r"[.!?:,]\s*")


def _extract_proper_nouns(text: str) -> set:
    """Capitalized, name-like tokens. A known, documented simplification:
    Turkish capitalizes the first word of every sentence (and this project's
    own opener phrases introduce a clause boundary of their own), so the
    text is split on sentence/clause punctuation and the first capitalized
    word of EACH clause is dropped before collecting the rest -- this cuts
    down on false positives from sentence- or clause-initial capitalization
    without needing a real named-entity recognizer. It is still a heuristic:
    it can miss a genuine name that happens to open its clause, and it can
    let a capitalized-but-generic word through if it is not clause-initial."""
    result = set()
    for clause in _CLAUSE_SPLIT_RE.split(text):
        words = _PROPER_NOUN_RE.findall(clause)
        if words:
            result.update(words[1:])
    return result


@dataclasses.dataclass
class EntitySet:
    numbers: List[str]
    dates: List[str]
    proper_nouns: List[str]


def extract_entities(text: str) -> EntitySet:
    return EntitySet(
        numbers=_extract_numbers(text),
        dates=_extract_dates(text),
        proper_nouns=sorted(_extract_proper_nouns(text)),
    )


def _lexical_agreement(samples: List[str]) -> float:
    """Average pairwise Jaccard similarity of content-word sets, as a 0-100
    score. Two empty samples are treated as trivially identical (1.0); one
    empty and one non-empty are treated as maximally different (0.0)."""
    token_sets = [_tokenize(s) for s in samples]
    pairs = list(itertools.combinations(range(len(samples)), 2))
    if not pairs:
        return 100.0
    sims = []
    for i, j in pairs:
        a, b = token_sets[i], token_sets[j]
        if not a and not b:
            sims.append(1.0)
        elif not a or not b:
            sims.append(0.0)
        else:
            sims.append(len(a & b) / len(a | b))
    return round(sum(sims) / len(sims) * 100, 1)


@dataclasses.dataclass
class MajorityAgreement:
    applicable: bool
    agreement_pct: float = 0.0
    majority_value: Optional[str] = None
    majority_count: int = 0
    n_with_entity: int = 0
    distinct_values: int = 0


def _majority_agreement(per_sample_values: List[set]) -> MajorityAgreement:
    """per_sample_values[i] is the SET of distinct entity strings sample i
    mentions (e.g. all numbers it contains). Samples that mention nothing are
    excluded from both the numerator and denominator -- an empty sample
    contributes no evidence either way, it is not treated as "agreeing".

    Agreement is computed over each sample's WHOLE entity set, not over
    individual entity values in isolation: a correct answer that legitimately
    cites more than one number (e.g. an arithmetic answer restating both
    operands and the result, "17 çarpı 6, 102 eder") should not be scored as
    "disagreement" just because it contains several distinct numbers -- what
    matters is whether every sample cites the SAME combination."""
    non_empty = [s for s in per_sample_values if s]
    if not non_empty:
        return MajorityAgreement(applicable=False)
    counter: Counter = Counter(frozenset(s) for s in non_empty)
    majority_set, majority_count = counter.most_common(1)[0]
    pct = round(majority_count / len(non_empty) * 100, 1)
    majority_label = ", ".join(sorted(majority_set)) if majority_set else ""
    return MajorityAgreement(
        applicable=True,
        agreement_pct=pct,
        majority_value=majority_label,
        majority_count=majority_count,
        n_with_entity=len(non_empty),
        distinct_values=len(counter),
    )


@dataclasses.dataclass
class ConsistencyResult:
    score: float
    rationale: str
    lexical_agreement: float
    number_agreement: Optional[MajorityAgreement]
    proper_noun_agreement: Optional[MajorityAgreement]
    n_samples: int
    samples: List[str]


def _rationale(n: int, lexical: float, num_agr: MajorityAgreement, name_agr: MajorityAgreement) -> str:
    """Builds a human-readable explanation from the actual computed
    statistics -- the wording changes with the data, it is never one fixed
    string (see module tests)."""
    parts = []

    if num_agr.applicable:
        if num_agr.distinct_values == 1:
            parts.append(f"{num_agr.n_with_entity} örneklemin tamamı aynı sayısal değeri "
                         f"({num_agr.majority_value}) içeriyor.")
        else:
            n_diff = num_agr.n_with_entity - num_agr.majority_count
            parts.append(f"{num_agr.n_with_entity} örneklemin {n_diff}'i, çoğunluğun verdiği "
                         f"'{num_agr.majority_value}' değerinden farklı bir sayı içeriyor "
                         f"({num_agr.distinct_values} farklı değer görüldü).")

    if name_agr.applicable:
        if name_agr.distinct_values == 1:
            parts.append(f"{name_agr.n_with_entity} örneklemin tamamı aynı özel adı "
                         f"('{name_agr.majority_value}') kullanıyor.")
        else:
            n_diff = name_agr.n_with_entity - name_agr.majority_count
            parts.append(f"İsim/özel ad içeren {name_agr.n_with_entity} örneklemin {n_diff}'i "
                         f"çoğunluktan ('{name_agr.majority_value}') farklı bir isim kullanıyor.")

    if not num_agr.applicable and not name_agr.applicable:
        if lexical >= 70:
            parts.append(f"Örneklemlerin hiçbirinde somut bir sayı ya da özel ad geçmiyor, ama kelime "
                         f"düzeyinde örtüşme yüksek (%{lexical}) -- bu, tutarlılığın bilgi doğruluğunu "
                         f"garanti etmediğinin tipik bir örneği olabilir (bkz. bilinen sınırlılıklar).")
        else:
            parts.append(f"Örneklemlerin hiçbirinde somut bir sayı ya da özel ad geçmiyor ve kelime "
                         f"düzeyinde örtüşme de düşük (%{lexical}).")
    else:
        parts.append(f"Kelime düzeyinde ortalama örtüşme: %{lexical}.")

    return " ".join(parts)


def score_consistency(samples: List[str]) -> ConsistencyResult:
    """The main entry point: takes N sampled responses to the SAME prompt and
    returns a 0-100 reliability score plus a rationale. No ground truth
    answer is used anywhere in this function -- see module docstring."""
    n = len(samples)
    if n == 0:
        return ConsistencyResult(score=0.0, rationale="Örneklem yok, skor hesaplanamadı.",
                                  lexical_agreement=0.0, number_agreement=None, proper_noun_agreement=None,
                                  n_samples=0, samples=[])
    if n == 1:
        return ConsistencyResult(score=50.0,
                                  rationale="Tek bir örneklemle tutarlılık ölçülemez; karşılaştırma için en az "
                                            "2 örneklem gerekir, bu yüzden nötr bir skor (50) atandı.",
                                  lexical_agreement=100.0, number_agreement=None, proper_noun_agreement=None,
                                  n_samples=1, samples=samples)

    entities = [extract_entities(s) for s in samples]
    number_sets = [set(e.numbers) | set(e.dates) for e in entities]
    name_sets = [set(e.proper_nouns) for e in entities]

    lexical = _lexical_agreement(samples)
    num_agr = _majority_agreement(number_sets)
    name_agr = _majority_agreement(name_sets)

    weights = {"lexical": 0.35, "number": 0.40, "name": 0.25}
    active = {"lexical": lexical}
    if num_agr.applicable:
        active["number"] = num_agr.agreement_pct
    if name_agr.applicable:
        active["name"] = name_agr.agreement_pct

    total_weight = sum(weights[k] for k in active)
    score = sum(weights[k] * active[k] for k in active) / total_weight
    score = round(score, 1)

    rationale = _rationale(n, lexical, num_agr, name_agr)

    return ConsistencyResult(
        score=score,
        rationale=rationale,
        lexical_agreement=lexical,
        number_agreement=num_agr,
        proper_noun_agreement=name_agr,
        n_samples=n,
        samples=samples,
    )


def risk_level_for(score: float) -> str:
    """0-100 reliability score -> a coarse hallucination-risk bucket, using
    the same three-tier convention as the sister project's risk levels
    (just relabeled for this project's own metric)."""
    if score >= 75:
        return "DÜŞÜK HALÜSİNASYON RİSKİ"
    if score >= 45:
        return "ORTA HALÜSİNASYON RİSKİ"
    return "YÜKSEK HALÜSİNASYON RİSKİ"


def aggregate(per_prompt: List[Dict]) -> Dict:
    """Rolls up a list of {category, result: ConsistencyResult-as-dict}
    entries into an overall score and a per-category breakdown -- mirrors
    classifiers.aggregate() in the sister project, but unweighted (every
    prompt in this battery is an equally-weighted single factual question;
    there is no severity axis for a hallucination check the way there is for
    a safety probe)."""
    by_cat: Dict[str, List[float]] = {}
    for entry in per_prompt:
        by_cat.setdefault(entry["category"], []).append(entry["result"]["score"])

    category_summary = {
        cat: {"n_prompts": len(scores), "avg_score": round(sum(scores) / len(scores), 1)}
        for cat, scores in by_cat.items()
    }

    all_scores = [entry["result"]["score"] for entry in per_prompt]
    overall = round(sum(all_scores) / len(all_scores), 1) if all_scores else 0.0

    return {
        "overall_reliability_score": overall,
        "risk_level": risk_level_for(overall),
        "n_prompts": len(per_prompt),
        "category_summary": category_summary,
    }
