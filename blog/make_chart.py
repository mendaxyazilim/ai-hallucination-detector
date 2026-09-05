"""
Reads the real results/*.json files (produced by actually running
cli.py --provider local-reference) and renders the results bar chart used in
the aix.web.tr blog post. No numbers are hand-typed here -- every value comes
straight from the JSON files this script loads.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "..", "results")

CONFIGS = [
    ("confident-fabricator.json", "Güvenli Uydurmacı\n(confident-fabricator)"),
    ("hedging.json", "Kaçamaklı\n(hedging)"),
    ("grounded.json", "Bilgi Tabanlı\n(grounded)"),
]

labels, scores, risk_labels = [], [], []
for filename, label in CONFIGS:
    with open(os.path.join(RESULTS_DIR, filename), encoding="utf-8") as f:
        data = json.load(f)
    labels.append(label)
    scores.append(data["summary"]["overall_reliability_score"])
    risk_labels.append(data["summary"]["risk_level"])


def color_for(score):
    if score >= 75:
        return "#1E8E5A"
    if score >= 45:
        return "#B8860B"
    return "#C23B3B"


colors = [color_for(s) for s in scores]

fig, ax = plt.subplots(figsize=(9, 5.2), dpi=200)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

bars = ax.bar(labels, scores, color=colors, width=0.55, zorder=3)

RISK_SHORT = {
    "DÜŞÜK HALÜSİNASYON RİSKİ": "DÜŞÜK RİSK",
    "ORTA HALÜSİNASYON RİSKİ": "ORTA RİSK",
    "YÜKSEK HALÜSİNASYON RİSKİ": "YÜKSEK RİSK",
}

for bar, score, risk in zip(bars, scores, risk_labels):
    ax.text(bar.get_x() + bar.get_width() / 2, score + 3, f"{score}", ha="center",
             fontsize=20, fontweight="bold", color="#151A22", family="DejaVu Sans")
    ax.text(bar.get_x() + bar.get_width() / 2, score - 6, RISK_SHORT.get(risk, risk), ha="center",
             fontsize=10.5, fontweight="bold", color="white", family="DejaVu Sans")

ax.set_ylim(0, 100)
ax.set_ylabel("Genel Güvenilirlik Skoru (0-100)", fontsize=11, color="#333")
ax.set_title("Aynı Sistem, Üç Yanıt Stratejisi: Gerçek Öz-Tutarlılık Sonuçları",
              fontsize=13.5, fontweight="bold", color="#151A22", pad=16)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#ccc")
ax.spines["bottom"].set_color("#ccc")
ax.yaxis.grid(True, color="#eee", zorder=0)
ax.set_axisbelow(True)
ax.tick_params(axis="x", labelsize=10.5, colors="#151A22")
ax.tick_params(axis="y", labelsize=10, colors="#666")

plt.tight_layout()
out_path = os.path.join(HERE, "sonuclar_grafik.png")
plt.savefig(out_path, facecolor="white")
print(f"saved {out_path}")
print(dict(zip(labels, scores)))
