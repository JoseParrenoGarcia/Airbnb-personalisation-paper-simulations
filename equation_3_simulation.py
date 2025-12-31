import numpy as np
import pandas as pd

from plots import (
    objective_curve_plot,
    dot_product_plots,
    equation3_long_df_add_labels,
    run_animation_long_df
)

class Equation3Model:
    """
    Equation (3) simulator (Airbnb skip-gram objective with negative sampling).

    We optimise (for one centre listing v_l):
        sum_{pos} log(sigmoid( v_pos^T v_l ))  +  sum_{neg} log(sigmoid( - v_neg^T v_l ))

    Airbnb expanded form:
        sum_{pos} log( 1 / (1 + exp(- v_pos^T v_l )) )
      + sum_{neg} log( 1 / (1 + exp(  v_neg^T v_l )) )

    This is intentionally simplified:
    - 2D vectors (so we can visualise)
    - a fixed number of positive and negative samples
    - no batching, no vocab distribution, no regularisation
    """

    def __init__(self, v_l, v_pos_list, v_neg_list, learning_rate=0.25, steps=60):
        self.v_l = np.array(v_l, dtype=float)

        # Lists of vectors
        self.v_pos = [np.array(v, dtype=float) for v in v_pos_list]
        self.v_neg = [np.array(v, dtype=float) for v in v_neg_list]

        self.learning_rate = float(learning_rate)
        self.steps = int(steps)

        # Trajectory: now includes an arbitrary number of pos/neg vectors.
        # We store them as lists per step.
        self.traj = {
            "step": [],
            "l": [],
            "pos": [],
            "neg": [],
            "x_pos": [],
            "x_neg": [],
            "obj": []
        }

        # Record step 0 diagnostics once
        self.record_for_animation(step=0)

    # --- maths helpers ---
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def log_sigmoid(self, x, eps=1e-12):
        s = self.sigmoid(x)
        return np.log(np.clip(s, eps, 1.0))

    def grad_log_sigmoid_dot(self, v_c, v_l):
        """
        Positive term gradient:
            log(sigmoid(v_c^T v_l))
        d/dx log(sigmoid(x)) = 1 - sigmoid(x)
        """
        x = float(v_c @ v_l)
        s = self.sigmoid(x)
        coeff = 1.0 - s
        grad_v_l = coeff * v_c
        grad_v_c = coeff * v_l
        return grad_v_c, grad_v_l, x

    def grad_log_sigmoid_neg_dot(self, v_c, v_l):
        """
        Negative term gradient:
            log(sigmoid(- v_c^T v_l))
        Let x = v_c^T v_l. Then log(sigmoid(-x)).
        d/dx log(sigmoid(-x)) = - (1 - sigmoid(-x))
        """
        x = float(v_c @ v_l)
        s = self.sigmoid(-x)
        coeff = -(1.0 - s)
        grad_v_l = coeff * v_c
        grad_v_c = coeff * v_l
        return grad_v_c, grad_v_l, x

    def objective(self):
        """
        Full Equation (3) objective for the current vectors.
        Returns: (obj_scalar, x_pos_list, x_neg_list)
        """
        x_pos = [float(v @ self.v_l) for v in self.v_pos]
        x_neg = [float(v @ self.v_l) for v in self.v_neg]

        obj_pos = sum(self.log_sigmoid(x) for x in x_pos)          # log(sigmoid(x))
        obj_neg = sum(self.log_sigmoid(-x) for x in x_neg)         # log(sigmoid(-x))

        return float(obj_pos + obj_neg), x_pos, x_neg

    def record_for_animation(self, step):
        obj, x_pos, x_neg = self.objective()

        self.traj["step"].append(step)
        self.traj["l"].append(self.v_l.copy())
        self.traj["pos"].append(np.stack([v.copy() for v in self.v_pos], axis=0))
        self.traj["neg"].append(np.stack([v.copy() for v in self.v_neg], axis=0))
        self.traj["x_pos"].append(x_pos)
        self.traj["x_neg"].append(x_neg)
        self.traj["obj"].append(obj)

    def run_simulation(self, record_every_step=True):
        """
        Gradient ASCENT on Equation (3).

        Per step:
        1) Compute gradients for each positive pair (pull).
        2) Compute gradients for each negative pair (push).
        3) Sum contributions for the centre vector.
        4) Update centre, positives, negatives via ascent.
        5) Record trajectory.
        """
        # If you want to avoid duplicating step 0, set record_every_step=False
        # since __init__ already records step 0.
        if record_every_step and self.traj["step"][-1] != 0:
            self.record_for_animation(step=0)

        for t in range(1, self.steps + 1):
            # -----------------------------
            # Step 1) Gradients from positives (pull term)
            # -----------------------------
            g_l_total = np.zeros_like(self.v_l)
            g_pos = []

            for v_p in self.v_pos:
                g_cp, g_lp, x = self.grad_log_sigmoid_dot(v_p, self.v_l)
                g_pos.append(g_cp)
                g_l_total += g_lp

            # -----------------------------
            # Step 2) Gradients from negatives (push term)
            # -----------------------------
            g_neg = []
            for v_n in self.v_neg:
                g_cn, g_ln, x = self.grad_log_sigmoid_neg_dot(v_n, self.v_l)
                g_neg.append(g_cn)
                g_l_total += g_ln

            # -----------------------------
            # Step 3) Update vectors (gradient ascent)
            # -----------------------------
            self.v_l = self.v_l + self.learning_rate * g_l_total

            for i in range(len(self.v_pos)):
                self.v_pos[i] = self.v_pos[i] + self.learning_rate * g_pos[i]

            for i in range(len(self.v_neg)):
                self.v_neg[i] = self.v_neg[i] + self.learning_rate * g_neg[i]

            # -----------------------------
            # Step 4) Record
            # -----------------------------
            if record_every_step:
                self.record_for_animation(step=t)

        return self.traj

def equation3_traj_to_long_df(traj):
    """
    Converts Equation3Model trajectory to a tidy (long) DataFrame.

    Output columns:
      step, group (centre/pos/neg), idx, x, y

    This format is ideal for Plotly animation with frame='step'.
    """
    rows = []

    for k in range(len(traj["step"])):
        step = int(traj["step"][k])

        # centre
        l = traj["l"][k]
        rows.append({"step": step, "group": "centre", "idx": 0, "x": float(l[0]), "y": float(l[1])})

        # positives
        pos = traj["pos"][k]  # (P,2)
        for i in range(pos.shape[0]):
            rows.append({"step": step, "group": "pos", "idx": i, "x": float(pos[i, 0]), "y": float(pos[i, 1])})

        # negatives
        neg = traj["neg"][k]  # (N,2)
        for i in range(neg.shape[0]):
            rows.append({"step": step, "group": "neg", "idx": i, "x": float(neg[i, 0]), "y": float(neg[i, 1])})

    return pd.DataFrame(rows)

def equation3_traj_to_metrics_df(traj):
    """
    Scalar/metric DataFrame:
      step, obj, x_pos_mean, x_neg_mean, x_pos_min/max, x_neg_min/max
    """
    df = pd.DataFrame({
        "step": traj["step"],
        "obj": traj["obj"],
        "x_pos_mean": [np.mean(x) for x in traj["x_pos"]],
        "x_neg_mean": [np.mean(x) for x in traj["x_neg"]],
        "x_pos_min": [np.min(x) for x in traj["x_pos"]],
        "x_pos_max": [np.max(x) for x in traj["x_pos"]],
        "x_neg_min": [np.min(x) for x in traj["x_neg"]],
        "x_neg_max": [np.max(x) for x in traj["x_neg"]],
    })
    return df



if __name__ == "__main__":
    pd.set_option('display.max_columns', None)  # show all columns
    pd.set_option('display.width', None)  # don't wrap columns to new lines
    pd.set_option('display.max_colwidth', None)  # show full column contents

    rng = np.random.default_rng(7)

    v_l = rng.normal(size=2)
    v_pos_list = [rng.normal(size=2), rng.normal(size=2)]
    v_neg_list = [rng.normal(size=2), rng.normal(size=2)]

    model = Equation3Model(v_l, v_pos_list, v_neg_list, learning_rate=0.25, steps=60)
    traj = model.run_simulation()

    df_long = equation3_traj_to_long_df(traj)
    df_metrics = equation3_traj_to_metrics_df(traj)

    print(df_long.head())
    print(df_metrics.head())

    # Supporting plots
    objective_curve_plot(df_metrics, title="Objective over optimisation steps (pull + push terms combined)")
    dot_product_plots(df_metrics, y_cols=["x_pos_mean", "x_neg_mean"], title="Positive dot products increase while negative dot products decrease")

    df_long_eq3 = equation3_long_df_add_labels(df_long)
    run_animation_long_df(
        df_long_eq3,
        title="Equation (3): pull + push in one embedding space",
    )