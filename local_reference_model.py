"""
local_reference_model.py
-------------------------
A small, fully transparent, LOCALLY-RUNNING text generation system used as
the worked example in this project's demo run -- see README.md ("Neden
yerel bir referans model?") for why a hosted model API could not be used.

WHAT THIS IS: a Markov-chain flavor-text generator (via the `markovify`
package, trained on a short original corpus written for this project -- see
CORPUS_TEXT below, which is NOT copied from the sister ai_safety_auditor
project or anywhere else) wrapped in THREE different, real, rule-based
response strategies. Every response used in the demo results is produced by
actually executing this code, nothing is pre-written or faked.

WHAT THIS IS NOT: a production large language model. It has none of the
world knowledge, fluency, or genuine reasoning of a system like GPT-4,
Claude, or Gemini. It is a stand-in target that lets the detector's full
pipeline (sampling, entity extraction, consistency scoring, reporting,
dashboard) be demonstrated end-to-end against a system whose hallucination
behavior we can engineer and therefore verify.

WHY A LOCAL STAND-IN INSTEAD OF A REAL HOSTED MODEL: model_adapters.py ships
fully working adapters for OpenAI, Anthropic, and Google Gemini (and any
OpenAI-compatible endpoint) -- point the CLI at any of them and the exact
same pipeline runs against a real production model. This project's sandbox,
however, was tested against every major hosted model API before writing any
of this and every one of them was rejected at the network layer (see
README.md for the exact test and result); no API key was available either.
Rather than fabricate example transcripts by hand -- which would misrepresent
what a real model actually said and violate this project's core "never
fabricate demo results" rule -- this local system produces genuine,
inspectable, reproducible output that the detector genuinely has to analyze.

The three response configurations (all built on the SAME base generator, so
that only the response *strategy* differs -- exactly what the demo needs to
show real, engineered differences in self-consistency):

  * "confident-fabricator" -- on every single sample, invents a fresh,
    specific-sounding but essentially random fact (a number, a date, or a
    name) to answer the question, phrased with total confidence and no
    hedging. Sampled 5 times, the 5 answers will typically each cite a
    DIFFERENT number/name -- this is precisely the failure mode
    SelfCheckGPT-style self-consistency checking is designed to catch, and
    it should score LOW on the reliability scale.

  * "hedging" -- never commits to a specific fact. Every sample is some
    variation of "I'm not fully certain, but roughly speaking...", with no
    number, name, or date in it at all. Because the samples never disagree
    (there is nothing concrete in them TO disagree about), naive
    lexical/entity consistency checking will score this config HIGH --
    which is exactly the known blind spot of self-consistency methods this
    project documents in the README rather than hides: consistent hedging
    is indistinguishable, to a no-ground-truth consistency check, from
    consistent correctness.

  * "grounded" -- looks up the question in a small, fixed, hand-written
    local knowledge base (KNOWLEDGE_BASE below) and returns the same
    correct-per-KB fact every time, with only light, cosmetic phrasing
    variation (a different opening clause picked at random) -- never a
    different underlying fact. This should score HIGH on the reliability
    scale, and for the right reason: real agreement on real content, not
    agreement-by-vagueness.
"""

from __future__ import annotations

import random
import re
from typing import List, Optional

import markovify

# Original short corpus written for this project (not copied from the
# sister ai_safety_auditor project's corpus, or anywhere else). Deliberately
# free of numbers, dates, and proper names -- it is only ever used as
# "flavor text" wrapped around a fact (real or fabricated), never as the
# source of the fact itself.
CORPUS_TEXT = """
The fishing boats leave the harbor before the sky has fully lightened.
A narrow footbridge connects the two halves of the old quarter across the canal.
Every stallholder at the covered market knows which regulars prefer their coffee unsweetened.
The lighthouse at the point has been repainted so many times that the layers show through where the paint has chipped.
Wind off the strait rattles the shutters of the houses along the waterfront each evening.
A cooperative of local weavers sells patterned scarves from a single stand near the ferry landing.
The old customs house now serves as a reading room, its high windows still fitted with the original iron latches.
Cats sleep in the sun on the warm stone steps leading down to the water.
A retired captain gives walking tours of the harbor for anyone curious about the shipwrecks charted offshore.
The bakery near the clock tower sells out of simit within an hour of opening.
Fishermen mend their nets on the quay while gulls circle overhead waiting for scraps.
The ferry schedule changes with the season, and regulars know to check the posted board rather than trust memory.
A small workshop behind the tea house repairs clocks, radios, and the occasional accordion.
Students from the coastal school sketch the boats in the harbor for their art class every spring.
The tea house terrace fills up just after the evening call to prayer, when the heat of the day has broken.
An elderly shopkeeper keeps a logbook of the birds that pass through the harbor each migration season.
The old shipyard, no longer in use, is now a favorite spot for evening walks along the water.
Vendors along the promenade sell roasted chestnuts once the weather turns cool.
A mural on the harbor wall depicts the town before the breakwater was built.
The public garden behind the mosque is quiet in the early afternoon, save for the sound of the fountain.
"""

# --- Knowledge base for the "grounded" configuration -----------------------
# A small, fixed, hand-written set of facts (original text, not copied from
# any dataset). Each entry lists the KEYWORDS (already lowercase, Turkish
# dotless/dotted-i issues handled by _normalize()) that must ALL appear in
# the incoming prompt for this fact to be considered "the" answer -- a
# transparent, rule-based lookup, not a semantic/embedding search. This is a
# deliberately small (20-entry) knowledge base: it does NOT cover every
# prompt in the battery on purpose, so the demo also shows what "grounded"
# does when nothing matches (see _FALLBACK_NO_MATCH below).
KNOWLEDGE_BASE = [
    {"keywords": ["cumhuriyet", "ilan"], "fact": "Türkiye Cumhuriyeti 1923 yılında ilan edildi."},
    {"keywords": ["ikinci dünya savaşı", "sona"], "fact": "İkinci Dünya Savaşı 1945 yılında sona erdi."},
    {"keywords": ["ay inişi"], "fact": "İlk insanlı Ay inişi 1969 yılında gerçekleşti."},
    {"keywords": ["berlin duvarı"], "fact": "Berlin Duvarı 1989 yılında yıkıldı."},

    {"keywords": ["artık yıl olmayan", "kaç gün"], "fact": "Artık yıl olmayan bir yılda 365 gün vardır."},
    {"keywords": ["metrekare", "santimetrekare"], "fact": "Bir metrekare 10000 santimetrekaredir."},
    {"keywords": ["kaynama sıcaklığı"], "fact": "Suyun deniz seviyesinde kaynama sıcaklığı 100 santigrat derecedir."},
    {"keywords": ["vücut sıcaklığı"], "fact": "Sağlıklı bir yetişkinin ortalama vücut sıcaklığı 36,6 santigrat derecedir."},

    {"keywords": ["görelilik teorisi"], "fact": "Görelilik teorisini Albert Einstein geliştirdi."},
    {"keywords": ["yurtta barış"], "fact": "'Yurtta barış, dünyada barış' sözü Mustafa Kemal Atatürk'e aittir."},
    {"keywords": ["ilk cumhurbaşkanı"], "fact": "Türkiye Cumhuriyeti'nin ilk cumhurbaşkanı Mustafa Kemal Atatürk'tür."},

    {"keywords": ["fotosentez"], "fact": "Fotosentez, bitkilerin güneş ışığını kullanarak karbondioksit ve sudan glikoz ve oksijen ürettiği süreçtir."},
    {"keywords": ["enflasyon"], "fact": "Enflasyon, genel fiyat düzeyinin sürekli ve belirgin biçimde artmasıdır."},
    {"keywords": ["yarımada"], "fact": "Yarımada, üç tarafı suyla çevrili olan kara parçasıdır."},

    {"keywords": ["17 çarpı 6"], "fact": "17 çarpı 6, 102 eder."},
    {"keywords": ["144", "karekök"], "fact": "144 sayısının karekökü 12'dir."},
    {"keywords": ["250", "yüzde 20"], "fact": "250 sayısının yüzde 20'si 50'dir."},

    {"keywords": ["türkiye'nin başkenti"], "fact": "Türkiye'nin başkenti Ankara'dır."},
    {"keywords": ["en uzun nehri"], "fact": "Dünyanın en uzun nehri olarak yaygın biçimde Nil Nehri kabul edilir."},
    {"keywords": ["nüfusu en kalabalık"], "fact": "Türkiye'nin nüfusu en kalabalık ili İstanbul'dur."},
]

_FALLBACK_NO_MATCH = [
    "Bu konudaki bilgi elimdeki küçük bilgi tabanında kayıtlı değil, bu yüzden kesin bir rakam ya da isim vermek "
    "yerine bunu açıkça belirtmeyi tercih ederim.",
    "Bu soru için bilgi tabanımda doğrudan bir karşılık bulamadım; uydurmak yerine bunu bilmediğimi söylüyorum.",
    "Elimdeki sınırlı bilgi tabanı bu konuyu kapsamıyor -- net bir cevap uydurmaktansa bunu böyle bırakmayı "
    "tercih ediyorum.",
]

_GROUNDED_OPENERS = [
    "Bildiğim kadarıyla: ",
    "Elimdeki bilgiye göre, ",
    "Bu konuda net olan bilgi şu: ",
    "Kısaca yanıtlamak gerekirse, ",
    "",
]

_HEDGE_VERB_ENDINGS = ["söyleyebilirim", "diyebilirim", "belirtebilirim", "aktarabilirim", "ifade edebilirim"]
_HEDGE_TEMPLATE = "Bu konuda tam olarak emin değilim, ama genel hatlarıyla bir şey {verb}."
# All five hedge responses share the same stock disclaimer almost word for
# word and differ only in the single closing verb -- a real model that
# hedges tends to paraphrase its own stock disclaimer rather than repeat it
# character-for-character, and THAT is exactly the case this config is built
# to reproduce: high lexical overlap between samples with zero actual
# informational content in any of them.

# Pools the "confident-fabricator" strategy draws a DIFFERENT entry from on
# every single call, regardless of which prompt it is answering -- the point
# is not that the fabricated fact is topically relevant, it is that it
# changes from sample to sample while sounding equally confident each time.
_FABRICATED_NUMBERS = ["37", "412", "1,8 milyon", "6,4", "89", "213", "3,2 milyar", "58", "1250", "17,5"]
_FABRICATED_YEARS = ["1894", "1937", "1958", "1971", "1988", "2004", "2011"]
_FABRICATED_NAMES = [
    "Prof. Halil Serdengeçti", "Dr. Elena Voskresenskaya", "Mühendis Tomas Berg",
    "Araştırmacı Yıldız Kaptanoğlu", "Prof. Marcus Villeneuve", "Dr. Nadia Tarhan",
]

_CONFIDENT_TEMPLATES = [
    "Kesin olarak söyleyebilirim ki cevap {fact}.",
    "Bu tamamen net: {fact}.",
    "Hiç kuşkusuz, {fact}.",
    "Elimdeki bilgiye göre kesinlikle {fact}.",
]


def _normalize(text: str) -> str:
    """Lowercase Turkish text safely -- Python's plain str.lower() turns the
    Turkish capital 'İ' into 'i' + a combining dot, which silently breaks
    substring matching against plain 'i'. Handling the dotted/dotless I pair
    explicitly, before lowering, avoids that trap."""
    return text.replace("İ", "i").replace("I", "ı").lower()


def _fabricate_fact() -> str:
    """Builds one fake but specific-sounding fact: a random pick from either
    a random number, a random year, or a random name -- different every call
    by design."""
    kind = random.choice(["number", "year", "name"])
    if kind == "number":
        return random.choice(_FABRICATED_NUMBERS)
    if kind == "year":
        return random.choice(_FABRICATED_YEARS) + " yılında"
    return random.choice(_FABRICATED_NAMES)


class BaseGenerator:
    """Thin wrapper around markovify so it degrades gracefully on very
    short/edge-case inputs (Markov models need enough text to build a
    chain)."""

    def __init__(self):
        self._model = markovify.Text(CORPUS_TEXT, state_size=2)

    def flavor_sentence(self) -> str:
        for _ in range(5):
            sentence = self._model.make_sentence(tries=50)
            if sentence:
                return sentence
        return "Bu konuda ek bir cümle üretilemedi."


def _lookup_kb(prompt: str) -> Optional[str]:
    norm = _normalize(prompt)
    for entry in KNOWLEDGE_BASE:
        if all(kw in norm for kw in entry["keywords"]):
            return entry["fact"]
    return None


class ReferenceSystem:
    def __init__(self, config: str = "grounded"):
        if config not in ("confident-fabricator", "hedging", "grounded"):
            raise ValueError("config must be one of: confident-fabricator, hedging, grounded")
        self.config = config
        self._gen = BaseGenerator()

    # -- response strategies -------------------------------------------------

    def _confident_fabricator_response(self, prompt: str) -> str:
        fake_fact = _fabricate_fact()
        template = random.choice(_CONFIDENT_TEMPLATES)
        body = template.format(fact=fake_fact)
        # a little flavor text so the response doesn't look mechanically
        # identical in structure across samples -- it carries no factual
        # content of its own (the corpus has no numbers/names/dates), so it
        # never masks or duplicates the fabricated fact.
        return f"{body} {self._gen.flavor_sentence()}"

    def _hedging_response(self, prompt: str) -> str:
        return _HEDGE_TEMPLATE.format(verb=random.choice(_HEDGE_VERB_ENDINGS))

    def _grounded_response(self, prompt: str) -> str:
        fact = _lookup_kb(prompt)
        opener = random.choice(_GROUNDED_OPENERS)
        if fact is None:
            return random.choice(_FALLBACK_NO_MATCH)
        return f"{opener}{fact}"

    def respond(self, prompt: str) -> str:
        if self.config == "confident-fabricator":
            return self._confident_fabricator_response(prompt)
        if self.config == "hedging":
            return self._hedging_response(prompt)
        return self._grounded_response(prompt)
