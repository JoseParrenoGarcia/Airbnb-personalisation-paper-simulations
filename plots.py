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

def dot_product_plots(df, y_cols, title):
    df_dots = df.copy()
    fig_dots = px.line(df_dots.dropna(subset=y_cols),
                       x="step",
                       y=y_cols,
                       title=title)
    fig_dots.update_layout(xaxis_title="step", yaxis_title="dot product", width=global_width, height=global_height)
    fig_dots.show()

def three_point_df_to_long(df, labels=("centre (l)", "context (c1)", "context (c2)")):
    """
    Converts the old wide DataFrame (l_x, c1_x, c2_x...) into a long DataFrame
    compatible with run_animation_long_df().
    """
    l_lab, c1_lab, c2_lab = labels

    rows = []
    for _, r in df.iterrows():
        step = int(r["step"])
        rows.append({"step": step, "label": l_lab,  "x": float(r["l_x"]),  "y": float(r["l_y"])})
        rows.append({"step": step, "label": c1_lab, "x": float(r["c1_x"]), "y": float(r["c1_y"])})
        rows.append({"step": step, "label": c2_lab, "x": float(r["c2_x"]), "y": float(r["c2_y"])})
    return pd.DataFrame(rows)

def equation3_long_df_add_labels(df_long):
    df = df_long.copy()
    def make_label(row):
        if row["group"] == "centre":
            return "centre (l)"
        elif row["group"] == "pos":
            return f"pos {int(row['idx'])+1}"
        else:
            return f"neg {int(row['idx'])+1}"
    df["label"] = df.apply(make_label, axis=1)
    return df

def run_animation_long_df(
    df_long,
    title,
    width=global_width,
    height=global_height,
    x_margin=0.5,
    y_margin=0.5,
    frame_duration=150,
):
    """
    Plotly animation for any simulation output in long form.

    Required columns in df_long:
      - step (int)
      - label (str)  # what is printed next to the point
      - x (float)
      - y (float)
    """
    required = {"step", "label", "x", "y"}
    missing = required - set(df_long.columns)
    if missing:
        raise ValueError(f"df_long is missing required columns: {missing}")

    df_anim = df_long.copy()

    # Fixed axis ranges so the plot does not jump
    x_min, x_max = df_anim["x"].min() - x_margin, df_anim["x"].max() + x_margin
    y_min, y_max = df_anim["y"].min() - y_margin, df_anim["y"].max() + y_margin

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
                    text=d["label"],
                    textposition="top center",
                )
            ],
            layout=go.Layout(title_text=f"{title} (step={s})"),
        )

    frames = [frame_for_step(s) for s in steps]

    d0 = df_anim[df_anim["step"] == step0]

    fig = go.Figure(
        data=[
            go.Scatter(
                x=d0["x"],
                y=d0["y"],
                mode="markers+text",
                text=d0["label"],
                textposition="top center",
            )
        ],
        layout=go.Layout(
            title=title,
            width=width,
            height=height,
            xaxis=dict(range=[x_min, x_max], title="x"),
            yaxis=dict(range=[y_min, y_max], title="y"),
            updatemenus=[{
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": frame_duration, "redraw": True}, "fromcurrent": True}],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
            }],
            sliders=[{
                "steps": [
                    {
                        "method": "animate",
                        "args": [[str(s)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
                        "label": str(s),
                    }
                    for s in steps
                ],
                "currentvalue": {"prefix": "step="},
            }],
        ),
        frames=frames,
    )

    fig.show()



