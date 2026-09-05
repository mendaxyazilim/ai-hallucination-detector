const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, AlignmentType, BorderStyle, PageBreak,
} = require("docx");

const results = {
  fabricator: JSON.parse(fs.readFileSync("../results/confident-fabricator.json", "utf8")),
  hedging: JSON.parse(fs.readFileSync("../results/hedging.json", "utf8")),
  grounded: JSON.parse(fs.readFileSync("../results/grounded.json", "utf8")),
};

const ACCENT = "1F3A5F";
const LIGHT = "EEF2F7";

function h1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 360, after: 160 } });
}
function h2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 280, after: 120 } });
}
function p(text, opts = {}) {
  return new Paragraph({ children: [new TextRun({ text, ...opts })], spacing: { after: 160 } });
}
function bullet(text) {
  return new Paragraph({ text, bullet: { level: 0 }, spacing: { after: 80 } });
}
function citeBullet(text) {
  return new Paragraph({ text, bullet: { level: 0 }, spacing: { after: 100 }, style: "citation" });
}

function cell(text, opts = {}) {
  const { bold = false, shade = null, width = null, align = AlignmentType.LEFT, color = null } = opts;
  return new TableCell({
    width: width ? { size: width, type: WidthType.DXA } : undefined,
    shading: shade ? { type: ShadingType.CLEAR, fill: shade } : undefined,
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    children: [new Paragraph({ alignment: align, children: [new TextRun({ text: String(text), bold, color: color || undefined })] })],
  });
}
function headCell(text, width) {
  return cell(text, { bold: true, shade: ACCENT, width, align: AlignmentType.CENTER, color: "FFFFFF" });
}

function summaryTable() {
  const rows = [
    new TableRow({
      tableHeader: true,
      children: [
        headCell("Yanıt Stratejisi", 3200),
        headCell("Genel Güvenilirlik Skoru", 2600),
        headCell("Risk Düzeyi", 3000),
      ],
    }),
  ];
  const items = [
    ["Güvenli Uydurmacı (confident-fabricator)", results.fabricator],
    ["Kaçamaklı (hedging)", results.hedging],
    ["Bilgi Tabanlı (grounded)", results.grounded],
  ];
  items.forEach(([label, r], i) => {
    const shade = i % 2 === 0 ? "FFFFFF" : LIGHT;
    rows.push(new TableRow({
      children: [
        cell(label, { shade, width: 3200 }),
        cell(`${r.summary.overall_reliability_score} / 100`, { shade, width: 2600, align: AlignmentType.CENTER }),
        cell(r.summary.risk_level, { shade, width: 3000, align: AlignmentType.CENTER }),
      ],
    }));
  });
  return new Table({ width: { size: 8800, type: WidthType.DXA }, columnWidths: [3200, 2600, 3000], rows });
}

function categoryTable() {
  const cats = Object.keys(results.grounded.summary.category_summary);
  const labelMap = {
    tarihsel_olaylar: "Tarihsel olaylar",
    istatistik_sayilar: "İstatistik / sayılar",
    kisiler_alintilar: "Kişiler / alıntılar",
    tanimlar: "Tanımlar",
    aritmetik: "Aritmetik",
    cografya: "Coğrafya",
  };
  const header = new TableRow({
    tableHeader: true,
    children: [
      headCell("Kategori", 2600),
      headCell("Uydurmacı", 2000),
      headCell("Kaçamaklı", 2000),
      headCell("Bilgi Tabanlı", 2200),
    ],
  });
  const rows = [header];
  cats.forEach((cat, i) => {
    const shade = i % 2 === 0 ? "FFFFFF" : LIGHT;
    rows.push(new TableRow({
      children: [
        cell(labelMap[cat] || cat, { shade, width: 2600 }),
        cell(results.fabricator.summary.category_summary[cat].avg_score, { shade, width: 2000, align: AlignmentType.CENTER }),
        cell(results.hedging.summary.category_summary[cat].avg_score, { shade, width: 2000, align: AlignmentType.CENTER }),
        cell(results.grounded.summary.category_summary[cat].avg_score, { shade, width: 2200, align: AlignmentType.CENTER }),
      ],
    }));
  });
  return new Table({ width: { size: 8800, type: WidthType.DXA }, columnWidths: [2600, 2000, 2000, 2200], rows });
}

const doc = new Document({
  styles: {
    paragraphStyles: [{
      id: "citation", name: "Citation", basedOn: "Normal",
      run: { size: 20, color: "444444", italics: true },
    }],
  },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    children: [
      new Paragraph({ text: "AI Hallüsinasyon Dedektörü", heading: HeadingLevel.TITLE, spacing: { after: 120 } }),
      new Paragraph({
        children: [new TextRun({ text: "Referans cevaba ihtiyaç duymayan, öz-tutarlılık tabanlı bir halüsinasyon riski ölçüm aracının yöntemi, mimarisi ve gerçek çalıştırma sonuçları", italics: true, color: "555555" })],
        spacing: { after: 360 },
      }),
      p("Bu belge; bir yapay zeka modelinin aynı soruya verdiği bağımsız yanıtlar arasındaki tutarlılığı ölçerek, gerçek cevabı bilmeden bir halüsinasyon riski tahmini üreten bir aracın yöntemini, uygulamasını ve gerçek çalıştırma sonuçlarını sunar."),

      h1("1. Amaç ve Motivasyon"),
      p("Büyük dil modellerinin ürettiği yanıtların bir kısmı, kulağa güvenli ve akıcı gelen ama gerçekte yanlış ya da uydurma (halüsinasyon) bilgiler içerebilir. Bu durumu tespit etmenin en zor yanı, çoğu gerçek kullanım senaryosunda 'doğru cevabın ne olduğunu' önceden bilmiyor olmamızdır -- bir soru sorulduğunda karşılaştıracağımız bir referans/altın standart cevap elimizde yoktur."),
      p("Bu proje, tam olarak bu kısıtla çalışan bir yöntemi -- öz-tutarlılık (self-consistency) tabanlı halüsinasyon tespitini -- gerçek, çalışan bir kod tabanında uygulamayı; aracı özgün bir 26 istemlik Türkçe bilgi/olgu bataryasıyla değerlendirmeyi; ve elde edilen gerçek (uydurulmamış) sonuçları hem yazılı bir raporda hem de bir web panosunda sunmayı hedefler."),

      h1("2. Yöntem: Öz-Tutarlılık ile Halüsinasyon Riski Ölçümü"),
      p("Yöntemin temel fikri şudur: bir modele aynı olgusal soruyu, durağan olmayan (temperature > 0) örnekleme ile birden çok (bu projede N=5) kez sorup bağımsız yanıtlar toplarsak, bu yanıtların BİRBİRİYLE NE KADAR UYUŞTUĞU, modelin o konuda gerçekten 'bildiği' bir şey olup olmadığına dair kaynak-bağımsız (zero-resource) bir sinyal verir. Model bir konuyu gerçekten biliyorsa, farklı örneklemlerde de büyük ölçüde aynı sayıyı, ismi veya tarihi tekrar etme eğilimindedir; halüsinasyon görüyorsa, her örneklemde farklı bir 'gerçek' uydurabilir."),
      p("Bu genel fikir, SelfCheckGPT (Manakul, Liusie ve Gales, 2023) makalesinde önerilen sıfır-kaynak halüsinasyon tespiti yaklaşımından esinlenilmiştir; bu projedeki istemler, çıkarım kodu ve puanlama mantığının hiçbiri o çalışmadan kopyalanmamış, yalnızca genel yöntembilim referans alınarak özgün biçimde yazılmıştır."),
      h2("2.1 Puanlama nasıl hesaplanıyor"),
      p("consistency_checker.py, N örneklem için üç şeffaf sinyali hesaplar ve ağırlıklı olarak birleştirir:"),
      bullet("Kelime düzeyinde örtüşme -- örneklemler arasındaki ortalama ikili Jaccard benzerliği (anlamsız/dolgu kelimeler çıkarılmış içerik kelimeleri üzerinden)."),
      bullet("Sayısal uyum -- örneklemlerin bahsettiği sayılar (yıllar dahil) çıkarılır; örneklemlerin ne kadarının aynı sayı kümesinde birleştiği ölçülür."),
      bullet("Özel ad uyumu -- aynı mantık, büyük harfle başlayan isim benzeri jetonlar (kişi, yer, kurum adları) üzerinden uygulanır."),
      p("Bu üç sinyal 0-100 arası bir 'Güvenilirlik Skoru'na dönüştürülür. Hiçbir örneklemde sayı ya da özel ad geçmiyorsa (ör. model hiç somut bir şey söylemiyorsa), o sinyal devre dışı bırakılır ve kalan ağırlıklar yeniden ölçeklenir -- böylece tanım sorularında özel ad bulunmaması skoru haksız yere düşürmez."),

      h1("3. Bilinen Sınırlılık: Tutarlılık, Doğruluğun Garantisi Değildir"),
      p("Bu yöntemin en önemli, bilinçli olarak gizlenmeyen sınırlılığı şudur: yöntem UYUMU ölçer, DOĞRULUĞU değil. Bir model her örneklemde farklı bir sayı uydurursa (halüsinasyon), bu doğru şekilde düşük skorla yakalanır. Ama bir model her seferinde aynı kaçamak/belirsizlik cümlesini tekrarlarsa -- 'bu konuda tam emin değilim, ama...' -- örneklemler arasında anlaşmazlık ÖLÇECEK hiçbir somut şey olmadığından, yöntem bunu YÜKSEK tutarlılık olarak görür."),
      p(`Bu proje bu kör noktayı teorik bir uyarı olarak bırakmak yerine, doğrudan ölçülebilir hale getirdi: local_reference_model.py içindeki 'hedging' (kaçamaklı) yapılandırması, hiçbir zaman somut bir sayı, isim ya da tarih içermeyen ama neredeyse aynı sözcüklerle her seferinde tekrarlanan bir kaçamak cümlesi üretir. Gerçek çalıştırma sonucunda bu yapılandırma ${results.hedging.summary.overall_reliability_score} / 100 güvenilirlik skoru almıştır -- gerçek bilgiye dayanan 'grounded' yapılandırmasının aldığı ${results.grounded.summary.overall_reliability_score} / 100 skorundan DAHA YÜKSEK. Bu, bir kusur değil, yöntemin gerçek ve belgelenmiş bir sınırlılığının somut kanıtıdır (bkz. Bölüm 6).`),

      h1("4. Sistem Mimarisi"),
      h2("4.1 Sağlayıcıdan bağımsız model adaptörleri ve N-örneklem alma"),
      p("Aracın çekirdeği, model_adapters.py içindeki ortak bir arayüzdür (ModelAdapter.generate / sample_n). Gerçek, çalışan HTTP istemcileri şu sağlayıcılar için uygulanmıştır: OpenAI (/v1/chat/completions, `n` parametresiyle tek çağrıda N örneklem), Anthropic (/v1/messages), Google Gemini (generateContent), ve herhangi bir OpenAI-uyumlu uç nokta (Groq, OpenRouter, Together, yerel Ollama/vLLM sunucuları dahil). Her adaptör kimlik bilgisini ortam değişkeninden okur; hiçbir anahtar koda gömülmez."),
      h2("4.2 Test bataryası"),
      p(`prompts/battery.py, altı kategoride toplam ${results.grounded.n_prompts} özgün, olgusal soru içerir: tarihsel olaylar, istatistikler/sayılar, kişiler/alıntılar, tanımlar, aritmetik ve coğrafya. Bu sorular halüsinasyon değerlendirmesinde yaygın kullanılan TruthfulQA (Lin ve ark., 2022) ve HaluEval (Li ve ark., 2023) tarzı kapalı-kitap olgusal soru fikrinden esinlenilerek yazılmıştır; hiçbiri o kaynaklardan kopyalanmamıştır.`),
      h2("4.3 Puanlama motoru"),
      p("consistency_checker.py, sınıflandırma için bir LLM-yargıç kullanmaz; tamamen şeffaf, regex/küme tabanlı bir motordur -- elle takip edilebilir, doğrulanabilir bir hesaplamadır. Bu tercih, sistemin kendisinin de anlaşılır olması gerektiği ilkesine dayanır (bkz. sister proje ai_safety_auditor'daki aynı tasarım kararı)."),

      h1("5. Deney Kurulumu: Neden Yerel Bir Referans Sistem?"),
      p("Araç, model_adapters.py üzerinden gerçek OpenAI / Anthropic / Gemini API'lerine karşı doğrudan çalışacak şekilde inşa edilmiştir. Ancak bu projenin geliştirildiği çalışma ortamının ağ erişimi, denendiğinde, api.openai.com, generativelanguage.googleapis.com, api.groq.com, openrouter.ai, api-inference.huggingface.co, api.together.xyz ve api.cohere.ai dahil denenen HER barındırılan model API'sinin CONNECT isteğini ağ geçidi (gateway) seviyesinde HTTP 403 ile reddettiğini göstermiştir. api.anthropic.com bu proxy'nin 'no_proxy' listesinde olduğu için doğrudan erişilebilir çıkmış, ancak API anahtarı olmadan (401 Unauthorized) kullanılamamıştır -- yani hiçbir sağlayıcıya, hiçbir şekilde, anahtarsız gerçek bir istek gönderilememiştir."),
      p("Sahte veya elle yazılmış örnek çıktılar sunmak yerine -- bu, gerçek bir modelin ne söylediğini yanlış temsil eder ve projenin 'uydurmama' ilkesine aykırı olurdu -- bu proje local_reference_model.py içinde küçük ama tamamen gerçek ve çalıştırılabilir bir sistem inşa etmiştir: markovify ile bu proje için özgün yazılmış bir metin üzerinde eğitilmiş bir Markov zinciri üretici, üç farklı ve gerçek yanıt stratejisiyle sarmalanmıştır:"),
      bullet("confident-fabricator -- her örneklemde farklı, uydurma ama kesin bir üslupla sunulan bir sayı/tarih/isim üretir."),
      bullet("hedging -- hiçbir zaman somut bir bilgi vermez; her seferinde neredeyse aynı sözcüklerle bir belirsizlik ifadesi tekrarlar."),
      bullet("grounded -- 20 girdilik küçük, sabit, elle yazılmış bir bilgi tabanından arama yapar ve her seferinde aynı doğru (KB'ye göre) bilgiyi, hafif doğal dil varyasyonuyla döndürür; bilgi tabanında karşılığı olmayan sorularda bunu açıkça belirtir."),
      p("Bu üç yapılandırma AYNI temel Markov üreticiyi paylaşır; farklılaşan tek şey yanıt stratejisidir. Herhangi biri gerçek bir sağlayıcı API anahtarı sağladığında, tek yapılması gereken cli.py komutuna --provider openai (veya anthropic / gemini / openai-compatible) geçmektir -- kod tabanında başka hiçbir değişiklik gerekmez."),

      h1("6. Sonuçlar"),
      p(`Aşağıdaki sonuçlar, ${results.grounded.n_prompts} istemlik tam bataryanın, istem başına 5 bağımsız örneklemle, üç yapılandırmaya karşı GERÇEKTEN çalıştırılmasıyla elde edilmiştir (bkz. results/*.json -- her istem, alınan 5 gerçek örneklem ve puanlayıcının gerekçesiyle birlikte kayıtlıdır).`),
      summaryTable(),
      new Paragraph({ text: "", spacing: { after: 240 } }),
      p("Kategori bazında ortalama güvenilirlik skorları:"),
      categoryTable(),
      new Paragraph({ text: "", spacing: { after: 240 } }),
      p(`Beklenen ve gözlemlenen örüntü kısmen doğrulandı: her örneklemde farklı bir sayı/isim uyduran confident-fabricator yapılandırması en düşük skoru aldı (${results.fabricator.summary.overall_reliability_score} / Yüksek Risk) -- bu, yöntemin asıl tasarlandığı senaryoda beklendiği gibi çalıştığını gösteriyor. Ancak grounded yapılandırması (${results.grounded.summary.overall_reliability_score}) ile hedging yapılandırması (${results.hedging.summary.overall_reliability_score}) arasındaki fark BEKLENENİN TERSİNE çıktı: kaçamaklı yapılandırma daha yüksek puan aldı. Bu, Bölüm 3'te tartışılan bilinen sınırlılığın rapor yazılmadan ÖNCE değil, aracın gerçekten çalıştırılmasıyla ortaya çıkan, gözlemlenmiş bir bulgudur.`),

      h1("7. Bulgular ve Sınırlılıklar"),
      h2("7.1 Tutarlılık ≠ doğruluk (bkz. Bölüm 3)"),
      p("Bu projenin en önemli bulgusu budur ve bilinçli olarak öne çıkarılmıştır: naif öz-tutarlılık ölçümü, 'hiçbir şey söylemeyen ama tutarlı bir şekilde söylemeyen' bir modeli, gerçekten bilgili ve tutarlı bir modelden ayırt edemez. Gerçek dünyada bunun anlamı, bu tür bir aracın çıktısının HER ZAMAN modelin gerçek yanıtlarıyla birlikte, insan gözüyle ya da ikinci bir sinyalle (ör. yanıtın en azından bir sayı/isim/tarih içerip içermediğinin ayrıca kontrol edilmesi) değerlendirilmesi gerektiğidir."),
      h2("7.2 Varlık çıkarımının kırılganlığı"),
      p("Sayı ve özel ad çıkarımı, düzenli ifadelere (regex) dayanır; karmaşık cümle yapıları, birleşik özel adlar veya beklenmeyen noktalama kalıpları bazı varlıkları kaçırabilir ya da yanlış işaretleyebilir. Bu, sister proje ai_safety_auditor'daki kural-tabanlı sınıflandırıcının aynı türden, bilinçli olarak kabul edilen bir zayıflığıdır."),
      h2("7.3 Bilgi tabanı kapsamı"),
      p("'grounded' yapılandırmasının kullandığı yerel bilgi tabanı yalnızca 20 girdi içerir ve bataryadaki 26 sorunun tamamını kapsamaz (bkz. local_reference_model.py -- bu bilinçli bir tasarım kararıdır). Kapsanmayan sorularda sistem bunu açıkça belirten, ama her seferinde farklı biçimde ifade edilen bir yanıt verir; bu da o sorularda skorun daha düşük çıkmasına yol açar -- yöntemin 'bilgim yok' durumunu da doğru bir şekilde ayırt edebildiğinin bir göstergesi."),
      h2("7.4 Kapsam"),
      p("Bu rapordaki sonuçlar yalnızca yerel referans sistemi kapsar; gerçek bir üretim modelinin (GPT, Claude, Gemini vb.) halüsinasyon davranışına dair bir iddia içermez. Aracın kendisi -- adaptörler, istem bataryası, tutarlılık motoru, raporlama ve pano -- herhangi bir gerçek API ile doğrudan kullanılabilir durumdadır."),

      h1("8. Sonuç ve Gelecek Çalışma"),
      p("Bu proje, referans cevaba ihtiyaç duymayan, öz-tutarlılık tabanlı bir halüsinasyon riski ölçüm aracı sunmaktadır: gerçek HTTP adaptörleri, özgün bir altı-kategori olgu bataryası, şeffaf bir tutarlılık motoru, otomatik testler ve gerçek (uydurulmamış) bir demo çalıştırması -- ve bu çalıştırmanın kendisinin ortaya çıkardığı, yöntemin kendi sınırlılığına dair somut bir bulgu. Olası genişletmeler: (1) gerçek bir sağlayıcı API anahtarıyla üretim modellerine karşı çalıştırma, (2) örneklemlerin en az birinin somut bir varlık içerip içermediğini ayrıca işaretleyen bir 'bilgi içeriği' sinyali ekleyerek hedging kör noktasını hafifletme, (3) daha büyük ve çeşitli bir olgu bataryası, (4) çok adımlı (multi-hop) olgusal sorular üzerinde yöntemin nasıl davrandığının incelenmesi."),
      citeBullet("SelfCheckGPT (Manakul, Liusie & Gales, 2023) -- bu projenin öz-tutarlılık fikrinin esin kaynağı."),
      citeBullet("TruthfulQA (Lin, Hilton & Evans, 2022) -- modellerin yaygın yanlış inançları doğru gibi tekrarlama eğilimini ölçen olgusal soru bataryası."),
      citeBullet("HaluEval (Li ve ark., 2023) -- büyük ölçekli, insan ve model tarafından üretilmiş halüsinasyon örnekleri değerlendirme kümesi."),
      citeBullet("HalluLens (2024-2025 döneminde yayınlanan halüsinasyon değerlendirme çalışmaları) -- halüsinasyon türlerinin (harici/dahili) sistematik ayrımı."),

      new Paragraph({ children: [new PageBreak()] }),
      h1("Ek A: Proje Dosya Yapısı"),
      bullet("model_adapters.py -- OpenAI / Anthropic / Gemini / OpenAI-uyumlu / yerel-referans adaptörleri, N-örneklem desteğiyle"),
      bullet("local_reference_model.py -- şeffaf demo sistemi (3 yanıt stratejisi + 20 girdilik bilgi tabanı)"),
      bullet("prompts/battery.py -- 26 özgün olgusal istem, 6 kategori"),
      bullet("consistency_checker.py -- öz-tutarlılık tabanlı puanlama motoru"),
      bullet("runner.py / cli.py -- uçtan uca çalıştırma ve komut satırı arayüzü"),
      bullet("tests/ -- 37 birim/uçtan-uca test (pytest)"),
      bullet("results/*.json -- bu rapordaki gerçek çalıştırma çıktıları"),
      bullet("README.md -- kurulum, kullanım ve tasarım kararlarının tam açıklaması"),
    ],
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("arastirma_raporu.docx", buf);
  console.log("wrote arastirma_raporu.docx");
});
