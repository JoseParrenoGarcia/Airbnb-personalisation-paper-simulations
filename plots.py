import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"

global_height = 500
global_width = 700

def objective_curve_plot(df, title):
    # Objective curve (pull term only)
    fig_obj = px.line(
        df.dropna(subset=["obj"]),
        x="step",
        y="obj",
        title=title,
    )

    fig_obj.update_layout(xaxis_title="step", yaxis_title="objective (log-sigmoid sum)", width=global_width, height=global_height)
    fig_obj.show()

def dot_product_plots(df, title):
    df_dots = df.copy()
    fig_dots = px.line(df_dots.dropna(subset=["x1", "x2"]), x="step", y=["x1", "x2"],
                       title=title)
    fig_dots.update_layout(xaxis_title="step", yaxis_title="dot product", width=global_width, height=global_height)
    fig_dots.show()

def _make_animation_df(df):
    # Long format: one row per (step, point)
    rows = []
    for _, r in df.iterrows():
        step = int(r["step"])
        rows.append({"step": step, "name": "centre (l)", "x": r["l_x"],  "y": r["l_y"]})
        rows.append({"step": step, "name": "context (c1)", "x": r["c1_x"], "y": r["c1_y"]})
        rows.append({"step": step, "name": "context (c2)", "x": r["c2_x"], "y": r["c2_y"]})
    return pd.DataFrame(rows)

def run_animation(df):
    df_anim = _make_animation_df(df)

    # Fixed axis ranges so the plot does not jump
    x_min, x_max = df_anim["x"].min() - 0.5, df_anim["x"].max() + 0.5
    y_min, y_max = df_anim["y"].min() - 0.5, df_anim["y"].max() + 0.5

    steps = sorted(df_anim["step"].unique())
    step0 = steps[0]

    def frame_for_step(s):
        d = df_anim[df_anim["step"] == s]
        return go.Frame(
            name=str(s),
            data=[
                go.Scatter(
                    x=d["x"],
                    y=d["y"],
                    mode="markers+text",
                    text=d["name"],
                    textposition="top center",
                )
            ],
            layout=go.Layout(
                title_text=f"Pull term only: vectors over optimisation steps (step={s})"
            )
        )

    frames = [frame_for_step(s) for s in steps]

    # initial data
    d0 = df_anim[df_anim["step"] == step0]

    fig = go.Figure(
        data=[
            go.Scatter(
                x=d0["x"],
                y=d0["y"],
                mode="markers+text",
                text=d0["name"],
                textposition="top center",
            )
        ],
        layout=go.Layout(
            title="Pull term only: co-clicked listings are pulled together",
            width=global_width, height=global_height,
            xaxis=dict(range=[x_min, x_max], title="x"),
            yaxis=dict(range=[y_min, y_max], title="y"),
            updatemenus=[{
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 150, "redraw": True}, "fromcurrent": True}]
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]
                    }
                ],
            }],
            sliders=[{
                "steps": [
                    {"method": "animate", "args": [[str(s)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
                     "label": str(s)}
                    for s in steps
                ],
                "currentvalue": {"prefix": "step="}
            }]
        ),
        frames=frames
    )

    fig.show()
