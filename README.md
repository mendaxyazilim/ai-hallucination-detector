# AI Hallüsinasyon Dedektörü

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Referans cevaba ihtiyaç duymayan, öz-tutarlılık (self-consistency) tabanlı
bir yapay zeka halüsinasyon riski ölçüm aracı. Herhangi bir sohbet/tamamlama
modeline (API uyumu olduğu sürece) orijinal, altı kategorilik bir olgusal
soru bataryası gönderir, her soru için N bağımsız örneklem toplar, bu
örneklemlerin birbiriyle ne kadar UYUŞTUĞUNU şeffaf bir kural-tabanlı motorla
ölçer ve 0-100 arası bir "Güvenilirlik Skoru" üretir.

Bu proje, [AIX](https://aix.web.tr) için yapılan bir araştırmanın parçası
olarak geliştirildi ve sister proje `ai_safety_auditor`'ın devamı
niteliğindedir. Metodoloji, mimari kararlar ve gerçek demo sonuçlarının tam
anlatımı için Bağlantılar bölümündeki yazıya bakın.

## Neden bu proje var?

Büyük dil modellerinin bir kısım yanıtı, kulağa akıcı ve güvenli gelse de
gerçekte yanlış ya da uydurma (halüsinasyon) bilgi içerebilir. Bunu tespit
etmenin en zor yanı, gerçek kullanımda çoğu zaman elimizde karşılaştıracağımız
bir "doğru cevap" olmamasıdır. Bu araç, tam olarak bu kısıtla çalışan bir
yöntemi -- aynı modele aynı soruyu birden çok kez sorup yanıtların birbiriyle
ne kadar tutarlı olduğuna bakan, sıfır-kaynak (zero-resource) bir yaklaşımı --
gerçek, çalışan bir kod tabanında uyguluyor.

## Hızlı başlangıç

```bash
pip install -r requirements.txt

# Ağ/API anahtarı gerektirmeyen yerel demo (3 yapılandırmadan biri):
python3 cli.py --provider local-reference --config grounded --samples-per-prompt 5 --out results/demo.json

# Gerçek bir sağlayıcıya karşı (kendi API anahtarınızla):
export OPENAI_API_KEY=sk-...
python3 cli.py --provider openai --model gpt-4o-mini --samples-per-prompt 5 --out results/gpt4o-mini.json

export ANTHROPIC_API_KEY=sk-ant-...
python3 cli.py --provider anthropic --model claude-3-5-haiku-20241022 --out results/claude.json

export GEMINI_API_KEY=...
python3 cli.py --provider gemini --model gemini-1.5-flash --out results/gemini.json

# Herhangi bir OpenAI-uyumlu uç nokta (Groq, OpenRouter, yerel Ollama/vLLM, ...):
export OPENAI_API_KEY=...
python3 cli.py --provider openai-compatible --model llama-3.1-8b \
    --base-url https://api.groq.com/openai/v1 --out results/groq.json
```

Testleri çalıştırmak için:

```bash
pip install -r requirements-dev.txt
python3 -m pytest tests/ -v
```

## Proje yapısı

| Dosya | Ne yapar |
|---|---|
| `model_adapters.py` | OpenAI / Anthropic / Gemini / OpenAI-uyumlu / yerel-referans adaptörleri, hepsi N-bağımsız-örneklem desteğiyle (hepsi gerçek, çalışan kod) |
| `local_reference_model.py` | Bu projenin demo hedefi: şeffaf, 3 yanıt stratejili yerel bir sistem (bkz. aşağıda "Neden yerel bir referans model?") |
| `prompts/battery.py` | 26 özgün olgusal test istemi, 6 kategoride |
| `consistency_checker.py` | Öz-tutarlılık tabanlı puanlama motoru (kural/desen tabanlı, LLM-yargıç değil) |
| `runner.py` / `cli.py` | Uçtan uca çalıştırma mantığı ve komut satırı arayüzü |
| `tests/` | 37 birim + uçtan-uca test (pytest, `requests_mock` ile HTTP adaptörleri gerçek ağ olmadan test edilir) |
| `results/*.json` | Bu projenin gerçek demo çalıştırmasının ham çıktıları |
| `report/build_report.js` | Word (.docx) araştırma raporunu üreten script |
| `dashboard/dashboard_final.html` | Web gösterge panosunun kaynak HTML'i |
| `blog/make_chart.py` | Sonuç grafiğini üreten script |

## Yöntem: öz-tutarlılık ile halüsinasyon riski ölçümü

Fikir SelfCheckGPT'nin (Manakul, Liusie & Gales, 2023) genel yaklaşımından
esinlenilmiştir (istemler, çıkarım kodu ve puanlama mantığının hiçbiri o
çalışmadan kopyalanmamış, özgün biçimde yazılmıştır): bir modele aynı
olgusal soruyu, durağan olmayan (temperature > 0) örnekleme ile N kez (bu
projede N=5) sorup bağımsız yanıtlar topluyoruz. Bu yanıtların birbiriyle ne
kadar uyuştuğu -- ortak sayılar, isimler, tarihler, genel kelime örtüşmesi --
modelin o konuda gerçekten "bildiği" bir şey olup olmadığına dair,
**referans cevaba ihtiyaç duymayan** bir sinyal verir: model bir konuyu
gerçekten biliyorsa örneklemler arasında büyük ölçüde aynı bilgiyi tekrar
eder; halüsinasyon görüyorsa her örneklemde farklı bir "gerçek" uydurabilir.

`consistency_checker.py`, her istem için şu üç şeffaf sinyali hesaplar ve
ağırlıklı ortalamalarını alır (bir sinyal hiç veri içermiyorsa -- ör. hiçbir
örneklemde sayı geçmiyorsa -- devre dışı bırakılıp kalan ağırlıklar yeniden
ölçeklenir):

1. **Kelime düzeyinde örtüşme** -- örneklemler arasındaki ortalama ikili
   Jaccard benzerliği.
2. **Sayısal uyum** -- örneklemlerin bahsettiği sayı/tarih kümeleri arasında
   ne kadarının aynı kombinasyonda birleştiği.
3. **Özel ad uyumu** -- aynı mantık, büyük harfle başlayan isim benzeri
   jetonlar üzerinden.

## Neden yerel bir referans model?

Aracın `model_adapters.py` dosyası gerçek OpenAI, Anthropic ve Google Gemini
API'lerine karşı doğrudan çalışacak şekilde inşa edilmiştir. Ancak bu
projenin geliştirildiği sanal ortamın ağ erişimi test edildiğinde (basit
HTTP CONNECT denemeleriyle, API anahtarı gerektirmeden), denenen **her**
barındırılan model API'si (api.openai.com, generativelanguage.googleapis.com,
api.groq.com, openrouter.ai, api-inference.huggingface.co, api.together.xyz,
api.cohere.ai) ağ geçidi (proxy) seviyesinde HTTP 403 ile engellenmiş
durumda çıktı. `api.anthropic.com` bu proxy'nin dışlama listesinde
(`no_proxy`) olduğu için doğrudan erişilebilir çıktı, ancak API anahtarı
olmadan yalnızca HTTP 401 (Unauthorized) döndü -- yani ona da gerçek bir
istek gönderilemedi. Hiçbirine API anahtarımız yoktu.

Sahte/elle yazılmış örnek yanıtlar üretmek yerine -- bu, gerçek bir modelin
ne söylediğini yanlış temsil eder ve projenin "uydurmama" ilkesine aykırı
olurdu -- `local_reference_model.py` içinde küçük ama **tamamen gerçek ve
çalıştırılabilir** bir sistem inşa ettik: `markovify` ile bu proje için
özgün yazılmış bir metin üzerinde eğitilmiş bir Markov zinciri üretici, üç
farklı ve gerçek yanıt stratejisiyle sarmalanmış:

- **`confident-fabricator`** -- her örneklemde farklı, uydurma ama tamamen
  kesin bir üslupla sunulan bir sayı/tarih/isim üretir.
- **`hedging`** -- hiçbir zaman somut bir bilgi vermez; her seferinde
  neredeyse aynı sözcüklerle bir belirsizlik ifadesi tekrarlar.
- **`grounded`** -- 20 girdilik küçük, sabit, elle yazılmış bir bilgi
  tabanından arama yapar ve her seferinde aynı doğru (KB'ye göre) bilgiyi,
  hafif doğal dil varyasyonuyla döndürür; bilgi tabanında karşılığı olmayan
  sorularda bunu açıkça belirtir (bilgi tabanı bilinçli olarak 26 sorunun
  tamamını değil, 20'sini kapsar).

Bu üç yapılandırma aynı temel üreticiyi paylaşır; tek fark yanıt
stratejisidir -- bu da aracın gerçekten farklı halüsinasyon davranışlarını
ölçüp ölçebildiğini göstermek için kontrollü bir demo ortamı sağlar. Bir API
anahtarınız varsa, yukarıdaki "Gerçek bir sağlayıcıya karşı" komutlarından
biriyle aynı aracı gerçek bir model üzerinde saniyeler içinde
çalıştırabilirsiniz.

### Gerçek çalıştırma sonuçları (N=5 örneklem, 26 istem)

| Yapılandırma | Güvenilirlik Skoru | Risk Düzeyi |
|---|---|---|
| `confident-fabricator` | 34,1 / 100 | YÜKSEK HALÜSİNASYON RİSKİ |
| `grounded` | 73,3 / 100 | ORTA HALÜSİNASYON RİSKİ |
| `hedging` | 79,0 / 100 | DÜŞÜK HALÜSİNASYON RİSKİ |

Bu tablo kasıtlı olarak rahatsız edici bir sırayla duruyor -- bkz. hemen
aşağıdaki bölüm.

## Bilinen sınırlılıklar

- **Tutarlılık, doğruluğun garantisi değildir (en önemli sınırlılık).**
  Yukarıdaki gerçek sonuçlarda görüldüğü gibi, hiçbir zaman somut bir bilgi
  vermeyen ama her seferinde neredeyse aynı sözcüklerle kaçamak yapan
  `hedging` yapılandırması (79,0), gerçek bilgiye dayanan ve o bilgiyi
  tutarlı biçimde tekrarlayan `grounded` yapılandırmasından (73,3) **daha
  yüksek** skor aldı. Yöntem örneklemler arasında anlaşmazlık ölçebileceği
  somut bir şey bulamadığında, bunu yüksek tutarlılık olarak yorumluyor --
  bu, teorik bir uyarı değil, aracın gerçekten çalıştırılmasıyla ortaya
  çıkan, ölçülmüş bir bulgu. Pratik sonucu: bu aracın skoru, bir yanıtın en
  azından somut bir iddia içerip içermediğine dair ayrı bir kontrolle
  birlikte okunmalıdır.
- Varlık (sayı/özel ad) çıkarımı, düzenli ifadelere (regex) dayanan şeffaf
  ama kırılgan bir yöntemdir; karmaşık cümle yapıları ya da beklenmedik
  noktalama bazı varlıkları kaçırabilir veya yanlış işaretleyebilir.
- `grounded` yapılandırmasının kullandığı yerel bilgi tabanı yalnızca 20
  girdi içerir ve 26 soruluk bataryanın tamamını bilinçli olarak kapsamaz;
  kapsanmayan sorularda skor daha düşük çıkar.
- Sonuçlar yalnızca bu projenin yerel demo sistemini kapsar; herhangi bir
  gerçek üretim modelinin halüsinasyon davranışına dair bir iddia içermez.

## Bağlantılar

- Yazı (metodoloji, mimari, gerçek sonuçlar, Word raporu ve canlı gösterge paneli dahil): [AI Modelleri Ne Zaman Halüsinasyon Görür? Kendi Güvenilirlik Dedektörümüzü Yaptık](https://aix.web.tr/ai-modelleri-ne-zaman-halusinasyon-gorur/)
- Kod deposu: [github.com/mendaxyazilim/ai-hallucination-detector](https://github.com/mendaxyazilim/ai-hallucination-detector)
- Canlı gösterge panosu: [aix.web.tr/ai-halusinasyon-risk-panosu](https://aix.web.tr/ai-halusinasyon-risk-panosu/)
- Gösterge panosu kaynağı: [dashboard/dashboard_final.html](dashboard/dashboard_final.html)
- AIX: [aix.web.tr](https://aix.web.tr)

## Lisans

Bu proje [MIT lisansı](LICENSE) ile yayınlanmıştır.
