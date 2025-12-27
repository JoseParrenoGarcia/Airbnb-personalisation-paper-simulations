import numpy as np
import pandas as pd
import plotly.express as px

import plotly.io as pio

pio.renderers.default = "browser"

# -----------------------------
# Mock-up data: 3 clusters in a 2D embedding space
# -----------------------------
rng = np.random.default_rng(42)

clusters = [
    # Cluster 1 (blue): Entire apartment, 2br, £200–£250 (4 listings)
    {
        "cluster": "Cluster 1",
        "group": "Entire apartment, 2 rooms, £200–£250 p/night",
        "property_type": "Entire apartment",
        "rooms": 2,
        "prices": [210, 225, 235, 245],
        "centre": np.array([1.6, 1.1]),
        "n": 4,
    },
    # Cluster 2 (green): Private room, 1br, £50–£60 (3 listings)
    {
        "cluster": "Cluster 2",
        "group": "Room only, £50–£60 p/night",
        "property_type": "Private room",
        "rooms": 1,
        "prices": [52, 55, 59],
        "centre": np.array([-1.2, 1.4]),
        "n": 3,
    },
    # Cluster 3 (red): Entire apartment, 1br, £200–£250 (4 listings)
    {
        "cluster": "Cluster 3",
        "group": "Entire apartment, 1 room, £200–£250 p/night",
        "property_type": "Entire apartment",
        "rooms": 1,
        "prices": [205, 215, 230, 248],
        "centre": np.array([1.2, -1.2]),
        "n": 4,
    },
]

rows = []
lid = 1

for c in clusters:
    cov = np.array([[0.05, 0.0], [0.0, 0.05]])  # keeps clusters tight
    pts = rng.multivariate_normal(mean=c["centre"], cov=cov, size=c["n"])

    for i in range(c["n"]):
        price = c["prices"][i]
        rows.append(
            {
                "listing_id": f"L{lid:02d}",
                "x": float(pts[i, 0]),
                "y": float(pts[i, 1]),
                "group": c["group"],
                "cluster": c["cluster"],
                "property_type": c["property_type"],
                "rooms": c["rooms"],
                "price_gbp": price,
                "label": f"{'Apt' if c['property_type']=='Entire apartment' else 'Room'}, {c['rooms']}br, £{price}",
            }
        )
        lid += 1

df = pd.DataFrame(rows)

# -----------------------------
# Cold-start listing: initialise from nearest neighbours in Cluster 1
# (In reality, 'nearest' would be computed in metadata space; here we illustrate in 2D.)
# -----------------------------
k = 3  # number of neighbours used to initialise the embedding

cluster1 = df[df["cluster"] == "Cluster 1"].copy()

# pick neighbours that are closest to the cluster centre (deterministic, avoids randomness)
cluster1_centre = np.array([1.6, 1.1])
cluster1["dist_to_centre"] = np.sqrt(
    (cluster1["x"] - cluster1_centre[0]) ** 2 + (cluster1["y"] - cluster1_centre[1]) ** 2
)
neighbours = cluster1.nsmallest(k, "dist_to_centre")[["listing_id", "x", "y", "label"]].copy()

# initialise NEW at the mean of those neighbours (simple average)
new_x = float(neighbours["x"].mean())
new_y = float(neighbours["y"].mean())

df_new = pd.DataFrame(
    [
        {
            "listing_id": "NEW",
            "x": new_x,
            "y": new_y,
            "group": "NEW: Entire apartment, 2 rooms, £200–£250 p/night",
            "cluster": "Cold-start",
            "property_type": "Entire apartment",
            "rooms": 2,
            "price_gbp": 238,
            "label": "NEW (cold-start): Apt, 2br, £238",
        }
    ]
)

df_plot = pd.concat([df, df_new], ignore_index=True)

# -----------------------------
# Plotly figure
# -----------------------------
fig = px.scatter(
    df_plot,
    x="x",
    y="y",
    color="group",
    text="listing_id",
    hover_name="listing_id",
    hover_data={
        "label": True,
        "property_type": True,
        "rooms": True,
        "price_gbp": True,
        "x": ":.3f",
        "y": ":.3f",
        "group": False,
        "cluster": True,
    },
    title="Mock-up 2D embedding space for London listings (cold-start initialisation)",
)

fig.update_traces(textposition="top center")

# # dashed lines from NEW to neighbours used in the average
# for _, r in neighbours.iterrows():
#     fig.add_shape(
#         type="line",
#         x0=new_x,
#         y0=new_y,
#         x1=float(r["x"]),
#         y1=float(r["y"]),
#         line=dict(dash="dash", width=1),
#     )

# fig.add_annotation(
#     x=new_x,
#     y=new_y,
#     text=f"Initialise NEW as mean of {k} similar listings",
#     showarrow=True,
#     arrowhead=2,
#     ax=40,
#     ay=-40,
# )

fig.update_layout(
    xaxis_title="Embedding dimension 1",
    yaxis_title="Embedding dimension 2",
    width=950,
    height=450,
    legend_title_text="Listing segment",
)

fig.show()

# Optional: print the neighbours used for initialisation
print(neighbours.reset_index(drop=True))
