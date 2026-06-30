import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ── Load org → group mapping ──────────────────────────────────────────────────
with open("lmarena-analysis/org_metadata.json") as f:
    org_meta = json.load(f)["orgs"]

ORG_ALIASES = {
    "bytedance": "Bytedance",
    "NexusFlow": "Nexusflow",
    "Zhipu": "Zhipu AI",
    "allenai": "Ai2",
    "microsoft-ai": "Microsoft",
    "microsoft": "Microsoft",
}

def get_group(org):
    org = ORG_ALIASES.get(org, org)
    origin = org_meta.get(org, {}).get("origin", "Other")
    if origin == "US":     return "US"
    if origin == "China":  return "China"
    if origin == "Europe": return "Europe"
    return "Other"

# ── Load & filter data ────────────────────────────────────────────────────────
df = pd.read_parquet("leaderboard-dataset/text/full-00000-of-00001.parquet")
df["leaderboard_publish_date"] = pd.to_datetime(df["leaderboard_publish_date"])
df = df[df["leaderboard_publish_date"] >= "2024-01-01"]
df["organization"] = df["organization"].map(lambda x: ORG_ALIASES.get(x, x))
df = df[df["organization"] != ""]

# Best model per org per date per category
best_all = (
    df.sort_values("rating", ascending=False)
    .groupby(["leaderboard_publish_date", "category", "organization"], as_index=False)
    .first()
)
best_all["group"] = best_all["organization"].map(get_group)

# ── Category labels (ordered) ─────────────────────────────────────────────────
CATEGORY_LABELS = {
    "overall": "Overall", "english": "English", "coding": "Coding",
    "math": "Math", "creative_writing": "Creative Writing",
    "instruction_following": "Instruction Following", "multi_turn": "Multi-Turn",
    "longer_query": "Longer Query", "hard_prompts": "Hard Prompts",
    "hard_prompts_english": "Hard Prompts (English)", "exclude_ties": "Exclude Ties",
    "expert": "Expert", "chinese": "Chinese", "french": "French",
    "german": "German", "japanese": "Japanese", "korean": "Korean",
    "russian": "Russian", "spanish": "Spanish",
    "industry_business_and_management_and_financial_operations": "Industry: Business & Finance",
    "industry_entertainment_and_sports_and_media": "Industry: Entertainment & Sports",
    "industry_legal_and_government": "Industry: Legal & Government",
    "industry_life_and_physical_and_social_science": "Industry: Life & Social Science",
    "industry_mathematical": "Industry: Mathematical",
    "industry_medicine_and_healthcare": "Industry: Medicine & Healthcare",
    "industry_software_and_it_services": "Industry: Software & IT",
    "industry_writing_and_literature_and_language": "Industry: Writing & Literature",
}
CAT_ORDER = list(CATEGORY_LABELS.keys())
available_cats = best_all["category"].unique().tolist()
categories = [c for c in CAT_ORDER if c in available_cats]

# ── Compute group win probabilities per category ──────────────────────────────
GROUPS = ["US", "China", "Europe", "Other"]

def win_prob(ra, rb):
    return 1 / (1 + 10 ** ((rb - ra) / 400))

def compute_for_category(cat):
    best = best_all[best_all["category"] == cat]
    dates_out, results = [], {g: [] for g in GROUPS}
    for date, day in best.groupby("leaderboard_publish_date"):
        group_best = {}
        for g in GROUPS:
            subset = day[day["group"] == g]
            if not subset.empty:
                group_best[g] = subset["rating"].max()
        if len(group_best) < 2:
            continue
        dates_out.append(date.strftime("%Y-%m-%d"))
        for g in GROUPS:
            if g not in group_best:
                results[g].append(None)
                continue
            ra = group_best[g]
            probs = [win_prob(ra, rb) for og, rb in group_best.items() if og != g]
            results[g].append(np.mean(probs) * 100)
    return dates_out, results

print("Computing win probabilities for all categories…")
cat_data = {cat: compute_for_category(cat) for cat in categories}
print("Done.")

# ── Build figure ──────────────────────────────────────────────────────────────
GROUP_LABELS = {"US": "🇺🇸 US", "China": "🇨🇳 China", "Europe": "🇪🇺 Europe", "Other": "Other"}
GROUP_COLORS = {"US": "#4285f4", "China": "#e11d48", "Europe": "#7c3aed", "Other": "#6b7280"}

fig = go.Figure()

# Add data traces for every category
for ci, cat in enumerate(categories):
    dates_out, results = cat_data[cat]
    first = (ci == 0)
    for g in GROUPS:
        xs = [d for d, v in zip(dates_out, results[g]) if v is not None]
        ys = [v for v in results[g] if v is not None]
        label = GROUP_LABELS[g]
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines+markers",
            name=label,
            visible=first,
            showlegend=first,
            line=dict(color=GROUP_COLORS[g], width=2.5),
            marker=dict(size=4),
            hovertemplate=(
                f"<b>{label}</b><br>"
                "%{x|%b %d, %Y}<br>"
                "Win prob: <b>%{y:.1f}%</b><extra></extra>"
            ),
        ))

n = len(GROUPS)
total = len(categories) * n

# Each button sets visible + showlegend for the selected category's traces
buttons = []
for ci, cat in enumerate(categories):
    visibility   = [False] * total
    show_legend  = [False] * total
    for j in range(n):
        visibility[ci * n + j]  = True
        show_legend[ci * n + j] = True
    buttons.append(dict(
        label=CATEGORY_LABELS[cat],
        method="update",
        args=[
            {"visible": visibility, "showlegend": show_legend},
            {"title": {
                "text": f"Win Probability by Geographic Group Over Time<br>"
                        f"<sup>Category: {CATEGORY_LABELS[cat]} · "
                        f"P(best model in group beats best model of each other group)</sup>",
                "font": {"size": 15},
            }},
        ],
    ))

fig.add_hline(y=50, line_dash="dot", line_color="#94a3b8", line_width=1.5,
              annotation_text="50%", annotation_position="right")

fig.update_layout(
    title=dict(
        text=f"Win Probability by Geographic Group Over Time<br>"
             f"<sup>Category: Overall · P(best model in group beats best model of each other group)</sup>",
        font=dict(size=15),
    ),
    updatemenus=[dict(
        type="dropdown",
        direction="up",
        x=0.5, xanchor="center",
        y=-0.12, yanchor="top",
        showactive=True,
        buttons=buttons,
    )],
    xaxis=dict(showgrid=False, zeroline=False, tickformat="%b %Y"),
    yaxis=dict(
        title="Win Probability (%)",
        ticksuffix="%",
        showgrid=True, gridcolor="#f3f4f6",
        zeroline=False, range=[20, 80],
    ),
    hovermode="x unified",
    plot_bgcolor="white",
    paper_bgcolor="white",
    height=540,
    margin=dict(t=100, r=120, b=100, l=70),
    legend=dict(x=1.01, y=1, xanchor="left", font=dict(size=12)),
    font=dict(family='-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif', size=12),
)

fig.write_image("lmarena-analysis/check_legend.pdf")
print("Saved: lmarena-analysis/check_legend.pdf")
fig.show()
