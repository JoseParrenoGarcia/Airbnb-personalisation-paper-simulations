import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from pretty_table import save_pretty_table_html

pio.renderers.default = "browser"

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

# ---------- Helpers ----------
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return np.nan
    return float(np.dot(a, b) / denom)

# ---------- 1) Mock historical user click embeddings ----------
# 5 clicks from NY (cluster near (1.0, 2.0))
ny_clicks = np.array([
    [0.9, 2.1],
    [1.2, 1.8],
    [1.1, 2.2],
    [0.8, 1.9],
    [1.3, 2.0],
])

# 3 clicks from LA (cluster near (-1.5, -0.5))
la_clicks = np.array([
    [-1.4, -0.6],
    [-1.6, -0.4],
    [-1.5, -0.7],
])

# Market-level centroids
ny_centroid = ny_clicks.mean(axis=0)
la_centroid = la_clicks.mean(axis=0)

# ---------- 2) Candidate London listings (4 candidates) ----------
# Designed so:
# - LDN_1 aligns with NY centroid
# - LDN_2 aligns with LA centroid
# - LDN_3 is closer to NY
# - LDN_4 is far / different direction
candidates = {
    "LDN_1": np.array([1.4, 2.05]),    # close to NY cluster
    "LDN_2": np.array([-1.55, -1]), # close to LA cluster
    "LDN_3": np.array([0.1, 1.6]),     # moderate, closer to NY
    "LDN_4": np.array([2.2, -1.7]),    # far / different direction
}

# ---------- 3) Compute similarities + Equation (10) max ----------
rows = []
for name, v in candidates.items():
    sim_ny = cosine_sim(v, ny_centroid)
    sim_la = cosine_sim(v, la_centroid)
    sim_max = max(sim_ny, sim_la)
    best_market = "NY" if sim_ny >= sim_la else "LA"
    rows.append({
        "candidate_listing": name,
        "cos_sim_with_Hc(NY)": sim_ny,
        "cos_sim_with_Hc(LA)": sim_la,
        "EmbClickSim (max)": sim_max,
        "best_matching_market": best_market
    })

df = pd.DataFrame(rows)
df["rank"] = df["EmbClickSim (max)"].rank(ascending=False, method="min").astype(int)
df = df.sort_values(["rank", "candidate_listing"]).reset_index(drop=True)

# Pretty formatting and show in browser using auxiliary function
rename_cols = {
    "candidate_listing": "Candidate listing",
    "cos_sim_with_Hc(NY)": "cos(ℓᵢ, μₕc(NY))",
    "cos_sim_with_Hc(LA)": "cos(ℓᵢ, μₕc(LA))",
    "EmbClickSim (max)": "EmbClickSim = maxₘ cos(ℓᵢ, μₕc(m))",
    "best_matching_market": "Best matching market",
    "rank": "Rank",
}

path = save_pretty_table_html(
    df=df,
    path="embclicksim_table.html",
    caption="EmbClickSim example (Equation 10): cosine similarity to market-level click centroids",
    footnote="EmbClickSim is computed as the maximum cosine similarity across market-level click centroids (Equation 10).",
    rename_cols=rename_cols,
    emphasise_col="EmbClickSim = maxₘ cos(ℓᵢ, μₕc(m))",
)

print("Saved to:", path)


def plot_embedding_space_plotly(
    ny_clicks,
    la_clicks,
    ny_centroid,
    la_centroid,
    candidates,
    title,
):
    fig = go.Figure()

    # Historical clicks
    fig.add_trace(go.Scatter(
        x=ny_clicks[:, 0],
        y=ny_clicks[:, 1],
        mode="markers",
        name="Clicked listings (NY)",
    ))

    fig.add_trace(go.Scatter(
        x=la_clicks[:, 0],
        y=la_clicks[:, 1],
        mode="markers",
        name="Clicked listings (LA)",
    ))

    # Centroids
    fig.add_trace(go.Scatter(
        x=[ny_centroid[0]],
        y=[ny_centroid[1]],
        mode="markers",
        name="Centroid Hc(NY)",
    ))

    fig.add_trace(go.Scatter(
        x=[la_centroid[0]],
        y=[la_centroid[1]],
        mode="markers",
        name="Centroid Hc(LA)",
    ))

    # Candidate listings
    cand_xy = np.vstack(list(candidates.values()))
    cand_labels = list(candidates.keys())

    fig.add_trace(go.Scatter(
        x=cand_xy[:, 0],
        y=cand_xy[:, 1],
        mode="markers+text",
        text=cand_labels,
        textposition="top center",
        name="Candidate listings (London)",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Embedding dimension 1",
        yaxis_title="Embedding dimension 2",
        width=900,
        height=550,
        legend_title_text="Legend",
    )

    fig.show()

plot_embedding_space_plotly(
    ny_clicks,
    la_clicks,
    ny_centroid,
    la_centroid,
    candidates,
    title="<b>EmbClickSim (Equation 10): market-level centroids vs candidate listings</b><br>Which listing will best match historical clicks?",
)