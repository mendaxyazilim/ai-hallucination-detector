"""
prompts/battery.py
-------------------
The detector's test prompt battery: original, factual/knowledge-seeking
questions written for this project, organized into six categories. None of
these are copied from any benchmark or dataset -- they are freshly authored
for this project, in the spirit of (but not copying) the kind of closed-book
factual questions used in hallucination-evaluation work such as TruthfulQA
(Lin et al., 2022) and HaluEval (Li et al., 2023).

Unlike the sister project's safety battery, these prompts are NOT meant to be
provocative or adversarial -- every one of them has a single, checkable,
"real world" correct answer (a number, a name, a date, a short definition).
That is deliberate: the self-consistency method implemented in
consistency_checker.py never looks at whether a sampled answer is actually
correct (it has no ground truth to compare against) -- it only measures
whether independent samples from the SAME model agree with each other. A
battery of check-able factual prompts just makes it easy for a human reader
of results/*.json to sanity-check, after the fact, whether "the samples
agreed" also happened to mean "the samples were right" -- which is a
question the tool itself deliberately does not answer.

Categories:
  * tarihsel_olaylar   -- historical events / dates
  * istatistik_sayilar -- general statistics / numeric facts
  * kisiler_alintilar  -- people, their work, and attributed quotes
  * tanimlar           -- short definitions of technical/general terms
  * aritmetik          -- arithmetic / calculation
  * cografya           -- geography

Each entry is a dict: {id, category, prompt, notes}.
"""

PROMPTS = [
    # ---------------- tarihsel_olaylar ----------------
    {"id": "tarih-01", "category": "tarihsel_olaylar",
     "prompt": "Türkiye Cumhuriyeti hangi yılda ilan edildi?",
     "notes": "Single well-known year: 1923."},
    {"id": "tarih-02", "category": "tarihsel_olaylar",
     "prompt": "İkinci Dünya Savaşı hangi yıl sona erdi?",
     "notes": "Single well-known year: 1945."},
    {"id": "tarih-03", "category": "tarihsel_olaylar",
     "prompt": "İlk insanlı Ay inişi hangi yılda gerçekleşti?",
     "notes": "Single well-known year: 1969 (Apollo 11)."},
    {"id": "tarih-04", "category": "tarihsel_olaylar",
     "prompt": "Berlin Duvarı hangi yıl yıkıldı?",
     "notes": "Single well-known year: 1989."},
    {"id": "tarih-05", "category": "tarihsel_olaylar",
     "prompt": "İstanbul'un fethi hangi yılda gerçekleşti?",
     "notes": "Single well-known year: 1453. Deliberately left OUT of the local knowledge base "
              "(see local_reference_model.py) so the demo also shows what the 'grounded' config "
              "does when it has no matching entry."},

    # ---------------- istatistik_sayilar ----------------
    {"id": "istat-01", "category": "istatistik_sayilar",
     "prompt": "Artık yıl olmayan bir yılda kaç gün vardır?",
     "notes": "Single well-known number: 365."},
    {"id": "istat-02", "category": "istatistik_sayilar",
     "prompt": "Bir metrekare kaç santimetrekaredir?",
     "notes": "Single well-known number: 10000."},
    {"id": "istat-03", "category": "istatistik_sayilar",
     "prompt": "Suyun deniz seviyesinde kaynama sıcaklığı kaç santigrat derecedir?",
     "notes": "Single well-known number: 100."},
    {"id": "istat-04", "category": "istatistik_sayilar",
     "prompt": "Sağlıklı bir yetişkinin ortalama vücut sıcaklığı kaç santigrat derecedir?",
     "notes": "Approximate but conventionally fixed number: 36,6."},
    {"id": "istat-05", "category": "istatistik_sayilar",
     "prompt": "Dünya'nın Güneş etrafındaki bir turu yaklaşık kaç gün sürer?",
     "notes": "365(.25). Deliberately left OUT of the local knowledge base."},

    # ---------------- kisiler_alintilar ----------------
    {"id": "kisi-01", "category": "kisiler_alintilar",
     "prompt": "Görelilik teorisini geliştiren bilim insanı kimdir?",
     "notes": "Single well-known name: Albert Einstein."},
    {"id": "kisi-02", "category": "kisiler_alintilar",
     "prompt": "'Yurtta barış, dünyada barış' sözü kime aittir?",
     "notes": "Single well-known name: Mustafa Kemal Atatürk."},
    {"id": "kisi-03", "category": "kisiler_alintilar",
     "prompt": "Ünlü 'Mona Lisa' tablosunun ressamı kimdir?",
     "notes": "Single well-known name: Leonardo da Vinci. Deliberately left OUT of the KB."},
    {"id": "kisi-04", "category": "kisiler_alintilar",
     "prompt": "Türkiye Cumhuriyeti'nin ilk cumhurbaşkanı kimdir?",
     "notes": "Single well-known name: Mustafa Kemal Atatürk."},

    # ---------------- tanimlar ----------------
    {"id": "tanim-01", "category": "tanimlar",
     "prompt": "Fotosentez nedir, kısaca tanımlar mısın?",
     "notes": "Definitional; checks whether the core mechanism stays stable across samples."},
    {"id": "tanim-02", "category": "tanimlar",
     "prompt": "İktisatta 'enflasyon' terimi ne anlama gelir?",
     "notes": "Definitional."},
    {"id": "tanim-03", "category": "tanimlar",
     "prompt": "Makine öğrenmesinde 'aşırı öğrenme' (overfitting) ne demektir?",
     "notes": "Definitional. Deliberately left OUT of the KB."},
    {"id": "tanim-04", "category": "tanimlar",
     "prompt": "Coğrafyada 'yarımada' kelimesi neyi ifade eder?",
     "notes": "Definitional."},

    # ---------------- aritmetik ----------------
    {"id": "arit-01", "category": "aritmetik",
     "prompt": "17 çarpı 6 kaçtır?",
     "notes": "Single correct numeric answer: 102."},
    {"id": "arit-02", "category": "aritmetik",
     "prompt": "144 sayısının karekökü kaçtır?",
     "notes": "Single correct numeric answer: 12."},
    {"id": "arit-03", "category": "aritmetik",
     "prompt": "250 sayısının yüzde 20'si kaçtır?",
     "notes": "Single correct numeric answer: 50."},
    {"id": "arit-04", "category": "aritmetik",
     "prompt": "9 üzeri 2 (9'un karesi) kaçtır?",
     "notes": "Single correct numeric answer: 81. Deliberately left OUT of the KB."},

    # ---------------- cografya ----------------
    {"id": "cog-01", "category": "cografya",
     "prompt": "Türkiye'nin başkenti neresidir?",
     "notes": "Single well-known name: Ankara."},
    {"id": "cog-02", "category": "cografya",
     "prompt": "Dünyanın en uzun nehri olarak kabul edilen nehir hangisidir?",
     "notes": "Conventionally taught single answer: Nil Nehri."},
    {"id": "cog-03", "category": "cografya",
     "prompt": "Everest Dağı hangi iki ülke arasındaki sınırda yer alır?",
     "notes": "Two names: Nepal ve Çin (Tibet). Deliberately left OUT of the KB."},
    {"id": "cog-04", "category": "cografya",
     "prompt": "Türkiye'nin nüfusu en kalabalık ili hangisidir?",
     "notes": "Single well-known name: İstanbul."},
]


def by_category(category: str):
    return [p for p in PROMPTS if p["category"] == category]


CATEGORIES = sorted(set(p["category"] for p in PROMPTS))
