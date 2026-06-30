import pandas as pd
import json
from pathlib import Path

with open("lmarena-analysis/org_metadata.json") as f:
    _meta = json.load(f)
LICENSE_CAT_MAP = _meta["license_categories"]
OPEN_SOURCE_MODELS = set(_meta.get("open_source_models", []))

def license_category(lic, model_name=""):
    if model_name in OPEN_SOURCE_MODELS:
        return "Open-source"
    return LICENSE_CAT_MAP.get(lic, "Unknown")

df = pd.read_parquet("leaderboard-dataset/text/full-00000-of-00001.parquet")

# Normalize org names (deduplicate casing/alias issues)
ORG_ALIASES = {
    "bytedance": "Bytedance",
    "NexusFlow": "Nexusflow",
    "Zhipu": "Zhipu AI",
    "allenai": "Ai2",
    "microsoft-ai": "Microsoft",
    "microsoft": "Microsoft",
}
df["organization"] = df["organization"].map(lambda x: ORG_ALIASES.get(x, x))
df = df[df["organization"] != ""]

df["leaderboard_publish_date"] = pd.to_datetime(df["leaderboard_publish_date"])
df = df[df["leaderboard_publish_date"] >= "2024-01-01"]

# Best model per (date, category, org)
best = (
    df.sort_values("rating", ascending=False)
    .groupby(["leaderboard_publish_date", "category", "organization"], as_index=False)
    .first()
)

all_dates = sorted(df["leaderboard_publish_date"].dt.strftime("%Y-%m-%d").unique().tolist())
date_idx = {d: i for i, d in enumerate(all_dates)}

CATEGORY_LABELS = {
    "overall": "Overall",
    "english": "English",
    "coding": "Coding",
    "math": "Math",
    "creative_writing": "Creative Writing",
    "instruction_following": "Instruction Following",
    "multi_turn": "Multi-Turn",
    "longer_query": "Longer Query",
    "hard_prompts": "Hard Prompts",
    "hard_prompts_english": "Hard Prompts (English)",
    "exclude_ties": "Exclude Ties",
    "expert": "Expert",
    "chinese": "Chinese",
    "french": "French",
    "german": "German",
    "japanese": "Japanese",
    "korean": "Korean",
    "russian": "Russian",
    "spanish": "Spanish",
    "industry_business_and_management_and_financial_operations": "Industry: Business & Finance",
    "industry_entertainment_and_sports_and_media": "Industry: Entertainment & Sports",
    "industry_legal_and_government": "Industry: Legal & Government",
    "industry_life_and_physical_and_social_science": "Industry: Life & Social Science",
    "industry_mathematical": "Industry: Mathematical",
    "industry_medicine_and_healthcare": "Industry: Medicine & Healthcare",
    "industry_software_and_it_services": "Industry: Software & IT",
    "industry_writing_and_literature_and_language": "Industry: Writing & Literature",
}

CAT_ORDER = [
    "overall", "english", "coding", "math", "creative_writing",
    "instruction_following", "multi_turn", "longer_query",
    "hard_prompts", "hard_prompts_english", "exclude_ties", "expert",
    "chinese", "french", "german", "japanese", "korean", "russian", "spanish",
    "industry_business_and_management_and_financial_operations",
    "industry_entertainment_and_sports_and_media",
    "industry_legal_and_government",
    "industry_life_and_physical_and_social_science",
    "industry_mathematical",
    "industry_medicine_and_healthcare",
    "industry_software_and_it_services",
    "industry_writing_and_literature_and_language",
]

available_cats = set(best["category"].unique())
categories = [{"id": c, "label": CATEGORY_LABELS.get(c, c)} for c in CAT_ORDER if c in available_cats]
for c in available_cats:
    if c not in CAT_ORDER:
        categories.append({"id": c, "label": CATEGORY_LABELS.get(c, c)})

# Order orgs by average rank in overall category (best avg rank first)
avg_rank = (
    best[best["category"] == "overall"]
    .groupby("organization")["rank"]
    .mean()
    .sort_values(ascending=True)
)
all_orgs = avg_rank.index.tolist()

data = {}
for cat in best["category"].unique():
    data[cat] = {}
    cat_df = best[best["category"] == cat]
    for org, grp in cat_df.groupby("organization", sort=False):
        grp = grp.sort_values("leaderboard_publish_date")
        dates = grp["leaderboard_publish_date"].dt.strftime("%Y-%m-%d").tolist()
        def clean(series):
            return [None if v != v else round(v, 1) for v in series]

        data[cat][org] = {
            "i": [date_idx[d] for d in dates],
            "r": clean(grp["rating"]),
            "l": clean(grp["rating_lower"]),
            "u": clean(grp["rating_upper"]),
            "m": grp["model_name"].tolist(),
            "k": [license_category(row["license"], row["model_name"]) for _, row in grp.iterrows()],
        }

out = {"dates": all_dates, "categories": categories, "orgs": all_orgs, "data": data}

out_path = Path("lmarena-analysis/dashboard_data.json")
with open(out_path, "w") as f:
    json.dump(out, f, separators=(",", ":"))

size = out_path.stat().st_size
print(f"Done! {len(categories)} categories, {len(all_orgs)} orgs, {len(all_dates)} dates")
print(f"Output: {out_path} ({size/1024:.0f} KB)")
