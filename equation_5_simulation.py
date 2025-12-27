import numpy as np
import pandas as pd

from plots import run_animation_long_df, booking_pull_vs_global_and_market_neg_plot

class Equation5Model:
    """
    Equation (5) simulator: Equation (4) + MARKET-AWARE negative samples.

    Objective:
        sum_{pos} log(sigmoid( v_pos^T v_l ))
      + sum_{neg} log(sigmoid( - v_neg^T v_l ))        # global negatives
      + sum_{mn}  log(sigmoid( - v_mn^T  v_l ))        # market-aware negatives
      +           log(sigmoid( v_b^T v_l ))            # booking (single pull)

    This mirrors Airbnb's formulation:
    - Same loss terms
    - Same optimisation
    - More informative negatives
    """

    def __init__(
        self,
        v_l,
        v_pos_list,
        v_neg_list,
        v_mn_list,
        v_b,
        learning_rate=0.25,
        steps=60
    ):
        self.v_l = np.array(v_l, dtype=float)

        self.v_pos = [np.array(v, dtype=float) for v in v_pos_list]
        self.v_neg = [np.array(v, dtype=float) for v in v_neg_list]   # global negatives
        self.v_mn  = [np.array(v, dtype=float) for v in v_mn_list]    # market-aware negatives
        self.v_b   = np.array(v_b, dtype=float)

        self.learning_rate = float(learning_rate)
        self.steps = int(steps)

        self.traj = {
            "step": [],
            "l": [],
            "pos": [],
            "neg": [],
            "mn": [],
            "b": [],
            "x_pos": [],
            "x_neg": [],
            "x_mn": [],
            "x_b": [],
            "obj": []
        }

        self.record_for_animation(step=0)

    # ---- maths helpers ----

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def log_sigmoid(self, x, eps=1e-12):
        s = self.sigmoid(x)
        return np.log(np.clip(s, eps, 1.0))

    def grad_log_sigmoid_dot(self, v_c, v_l):
        """
        Positive pull: log(sigmoid(v_c^T v_l))
        Returns: (grad_v_c, grad_v_l, x)
        """
        x = float(v_c @ v_l)
        s = self.sigmoid(x)
        coeff = 1.0 - s
        grad_v_l = coeff * v_c
        grad_v_c = coeff * v_l
        return grad_v_c, grad_v_l, x

    def grad_log_sigmoid_neg_dot(self, v_c, v_l):
        """
        Negative push: log(sigmoid(-v_c^T v_l))
        Returns: (grad_v_c, grad_v_l, x)
        """
        x = float(v_c @ v_l)
        s = self.sigmoid(-x)
        coeff = -(1.0 - s)
        grad_v_l = coeff * v_c
        grad_v_c = coeff * v_l
        return grad_v_c, grad_v_l, x

    # ---- objective ----

    def objective(self):
        x_pos = [float(v @ self.v_l) for v in self.v_pos]
        x_neg = [float(v @ self.v_l) for v in self.v_neg]
        x_mn  = [float(v @ self.v_l) for v in self.v_mn]
        x_b   = float(self.v_b @ self.v_l)

        obj = (
            sum(self.log_sigmoid(x) for x in x_pos) +
            sum(self.log_sigmoid(-x) for x in x_neg) +
            sum(self.log_sigmoid(-x) for x in x_mn) +
            self.log_sigmoid(x_b)
        )

        return float(obj), x_pos, x_neg, x_mn, x_b

    # ---- recording ----

    def record_for_animation(self, step):
        obj, x_pos, x_neg, x_mn, x_b = self.objective()

        self.traj["step"].append(step)
        self.traj["l"].append(self.v_l.copy())
        self.traj["pos"].append(np.stack([v.copy() for v in self.v_pos]))
        self.traj["neg"].append(np.stack([v.copy() for v in self.v_neg]))
        self.traj["mn"].append(np.stack([v.copy() for v in self.v_mn]))
        self.traj["b"].append(self.v_b.copy())
        self.traj["x_pos"].append(x_pos)
        self.traj["x_neg"].append(x_neg)
        self.traj["x_mn"].append(x_mn)
        self.traj["x_b"].append(x_b)
        self.traj["obj"].append(obj)

    # ---- optimisation ----

    def run_simulation(self):
        """
        Gradient ASCENT on Equation (5).

        Forces acting on centre:
        - pull from positives
        - pull from booking
        - push from global negatives
        - push from market-aware negatives
        """

        for t in range(1, self.steps + 1):
            g_l_total = np.zeros_like(self.v_l)

            # --- positive pulls ---
            for i, v_p in enumerate(self.v_pos):
                g_cp, g_lp, _ = self.grad_log_sigmoid_dot(v_p, self.v_l)
                self.v_pos[i] += self.learning_rate * g_cp
                g_l_total += g_lp

            # --- booking pull ---
            g_cb, g_lb, _ = self.grad_log_sigmoid_dot(self.v_b, self.v_l)
            self.v_b += self.learning_rate * g_cb
            g_l_total += g_lb

            # --- global negatives ---
            for i, v_n in enumerate(self.v_neg):
                g_cn, g_ln, _ = self.grad_log_sigmoid_neg_dot(v_n, self.v_l)
                self.v_neg[i] += self.learning_rate * g_cn
                g_l_total += g_ln

            # --- market-aware negatives ---
            for i, v_mn in enumerate(self.v_mn):
                g_cmn, g_lmn, _ = self.grad_log_sigmoid_neg_dot(v_mn, self.v_l)
                self.v_mn[i] += self.learning_rate * g_cmn
                g_l_total += g_lmn

            # --- centre update ---
            self.v_l += self.learning_rate * g_l_total

            self.record_for_animation(step=t)

        return self.traj

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)  # show all columns
    pd.set_option('display.width', None)  # don't wrap columns to new lines
    pd.set_option('display.max_colwidth', None)  # show full column contents


    def equation5_traj_to_long_df(traj):
        rows = []
        for k in range(len(traj["step"])):
            step = traj["step"][k]

            rows.append({"step": step, "group": "centre", "idx": 0,
                         "x": traj["l"][k][0], "y": traj["l"][k][1]})

            rows.append({"step": step, "group": "booking", "idx": 0,
                         "x": traj["b"][k][0], "y": traj["b"][k][1]})

            for i, v in enumerate(traj["pos"][k]):
                rows.append({"step": step, "group": "pos", "idx": i,
                             "x": v[0], "y": v[1]})

            for i, v in enumerate(traj["neg"][k]):
                rows.append({"step": step, "group": "neg", "idx": i,
                             "x": v[0], "y": v[1]})

            for i, v in enumerate(traj["mn"][k]):
                rows.append({"step": step, "group": "mn", "idx": i,
                             "x": v[0], "y": v[1]})

        return pd.DataFrame(rows)


    def equation5_traj_to_metrics_df(traj):
        """
        Metrics DataFrame for Equation (5):

        Columns:
          step, obj, x_b
          x_pos_mean/min/max
          x_neg_mean/min/max   (global negatives)
          x_mn_mean/min/max    (market-aware negatives)
        """
        return pd.DataFrame({
            "step": traj["step"],
            "obj": traj["obj"],
            "x_b": traj["x_b"],

            "x_pos_mean": [float(np.mean(x)) for x in traj["x_pos"]],
            "x_pos_min": [float(np.min(x)) for x in traj["x_pos"]],
            "x_pos_max": [float(np.max(x)) for x in traj["x_pos"]],

            "x_neg_mean": [float(np.mean(x)) for x in traj["x_neg"]],
            "x_neg_min": [float(np.min(x)) for x in traj["x_neg"]],
            "x_neg_max": [float(np.max(x)) for x in traj["x_neg"]],

            "x_mn_mean": [float(np.mean(x)) for x in traj["x_mn"]],
            "x_mn_min": [float(np.min(x)) for x in traj["x_mn"]],
            "x_mn_max": [float(np.max(x)) for x in traj["x_mn"]],
        })


    def equation5_add_labels(df_long):
        """
        Adds a human-readable 'label' column for animation.

        Expected groups:
          centre, booking, pos, neg, mn
        """
        df = df_long.copy()

        def make_label(row):
            g = row["group"]
            if g == "centre":
                return "centre (l)"
            if g == "booking":
                return "booked (b)"
            if g == "pos":
                return f"pos {int(row['idx']) + 1}"
            if g == "neg":
                return f"neg {int(row['idx']) + 1}"
            if g == "mn":
                return f"mn {int(row['idx']) + 1}"
            return str(g)

        df["label"] = df.apply(make_label, axis=1)
        return df


    rng = np.random.default_rng(7)

    v_l = rng.normal(size=2)
    v_pos_list = [rng.normal(size=2), rng.normal(size=2)]
    v_neg_list = [rng.normal(size=2), rng.normal(size=2)]  # global negatives
    v_mn_list = [rng.normal(size=2), rng.normal(size=2)]  # market negatives
    v_b = rng.normal(size=2)

    model = Equation5Model(
        v_l=v_l,
        v_pos_list=v_pos_list,
        v_neg_list=v_neg_list,
        v_mn_list=v_mn_list,
        v_b=v_b,
        learning_rate=0.1,
        steps=60
    )

    traj = model.run_simulation()

    df_long = equation5_add_labels(equation5_traj_to_long_df(traj))
    df_metrics = equation5_traj_to_metrics_df(traj)

    print(df_long.head())
    print(df_metrics.head())

    booking_pull_vs_global_and_market_neg_plot(
        df_metrics,
        title="Equation (5): booking pull vs global & market-aware negative pushes"
    )

    run_animation_long_df(
        df_long,
        title="Equation (5): market-aware negatives sharpen local separation",
    )
