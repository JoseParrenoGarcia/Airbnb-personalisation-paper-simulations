import numpy as np
import pandas as pd

class Equation4Model:
    """
    Equation (4) simulator: Equation (3) + ONE booking pull term.

    Objective:
        sum_{pos} log(sigmoid( v_pos^T v_l ))
      + sum_{neg} log(sigmoid( - v_neg^T v_l ))
      +           log(sigmoid( v_b^T v_l ))     # booking term (single, no summation)

    Notes:
    - 2D vectors for visualisation
    - fixed number of positive and negative samples
    - single booked listing vector v_b
    """

    def __init__(self, v_l, v_pos_list, v_neg_list, v_b, learning_rate=0.25, steps=60):
        self.v_l = np.array(v_l, dtype=float)

        self.v_pos = [np.array(v, dtype=float) for v in v_pos_list]
        self.v_neg = [np.array(v, dtype=float) for v in v_neg_list]
        self.v_b = np.array(v_b, dtype=float)

        self.learning_rate = float(learning_rate)
        self.steps = int(steps)

        # Trajectory (long-friendly): centre + arrays of pos/neg + booking
        self.traj = {
            "step": [],
            "l": [],
            "pos": [],
            "neg": [],
            "b": [],
            "x_pos": [],   # list of dot products v_pos^T v_l
            "x_neg": [],   # list of dot products v_neg^T v_l
            "x_b": [],     # scalar dot product v_b^T v_l
            "obj": []      # scalar objective
        }

        self.record_for_animation(step=0)

    # --- maths helpers ---
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def log_sigmoid(self, x, eps=1e-12):
        s = self.sigmoid(x)
        return np.log(np.clip(s, eps, 1.0))

    def grad_log_sigmoid_dot(self, v_c, v_l):
        """
        Gradient of log(sigmoid(v_c^T v_l)) wrt v_c and v_l.
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
        Gradient of log(sigmoid(-v_c^T v_l)) wrt v_c and v_l.
        Let x = v_c^T v_l. Then d/dx log(sigmoid(-x)) = - (1 - sigmoid(-x))
        """
        x = float(v_c @ v_l)
        s = self.sigmoid(-x)
        coeff = -(1.0 - s)
        grad_v_l = coeff * v_c
        grad_v_c = coeff * v_l
        return grad_v_c, grad_v_l, x

    def objective(self):
        x_pos = [float(v @ self.v_l) for v in self.v_pos]
        x_neg = [float(v @ self.v_l) for v in self.v_neg]
        x_b = float(self.v_b @ self.v_l)

        obj_pos = sum(self.log_sigmoid(x) for x in x_pos)   # log(sigmoid(x))
        obj_neg = sum(self.log_sigmoid(-x) for x in x_neg)  # log(sigmoid(-x))
        obj_b = self.log_sigmoid(x_b)                       # booking pull (single)

        return float(obj_pos + obj_neg + obj_b), x_pos, x_neg, x_b

    def record_for_animation(self, step):
        obj, x_pos, x_neg, x_b = self.objective()

        self.traj["step"].append(int(step))
        self.traj["l"].append(self.v_l.copy())
        self.traj["pos"].append(np.stack([v.copy() for v in self.v_pos], axis=0))
        self.traj["neg"].append(np.stack([v.copy() for v in self.v_neg], axis=0))
        self.traj["b"].append(self.v_b.copy())
        self.traj["x_pos"].append(x_pos)
        self.traj["x_neg"].append(x_neg)
        self.traj["x_b"].append(float(x_b))
        self.traj["obj"].append(float(obj))

    def run_simulation(self, record_every_step=True):
        """
        Gradient ASCENT on Equation (4).

        Per step:
        1) Pull from positives
        2) Push from negatives
        3) Pull from booking vector (single additional positive term)
        4) Update centre, pos, neg, booking
        5) Record
        """
        for t in range(1, self.steps + 1):
            g_l_total = np.zeros_like(self.v_l)

            # --- positives (pull) ---
            g_pos = []
            for v_p in self.v_pos:
                g_cp, g_lp, _ = self.grad_log_sigmoid_dot(v_p, self.v_l)
                g_pos.append(g_cp)
                g_l_total += g_lp

            # --- negatives (push) ---
            g_neg = []
            for v_n in self.v_neg:
                g_cn, g_ln, _ = self.grad_log_sigmoid_neg_dot(v_n, self.v_l)
                g_neg.append(g_cn)
                g_l_total += g_ln

            # --- booking (single pull) ---
            g_b, g_lb, _ = self.grad_log_sigmoid_dot(self.v_b, self.v_l)
            g_l_total += g_lb

            # --- ascent updates ---
            self.v_l = self.v_l + self.learning_rate * g_l_total

            for i in range(len(self.v_pos)):
                self.v_pos[i] = self.v_pos[i] + self.learning_rate * g_pos[i]

            for i in range(len(self.v_neg)):
                self.v_neg[i] = self.v_neg[i] + self.learning_rate * g_neg[i]

            # Booking vector also gets updated (it is a trainable embedding in this toy setup)
            self.v_b = self.v_b + self.learning_rate * g_b

            if record_every_step:
                self.record_for_animation(step=t)

        return self.traj


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)  # show all columns
    pd.set_option('display.width', None)  # don't wrap columns to new lines
    pd.set_option('display.max_colwidth', None)  # show full column contents


    def equation4_traj_to_long_df(traj):
        """
        Long/tidy DataFrame for animation:
          step, group (centre/pos/neg/booking), idx, x, y
        """
        rows = []
        for k in range(len(traj["step"])):
            step = int(traj["step"][k])

            l = traj["l"][k]
            rows.append({"step": step, "group": "centre", "idx": 0, "x": float(l[0]), "y": float(l[1])})

            b = traj["b"][k]
            rows.append({"step": step, "group": "booking", "idx": 0, "x": float(b[0]), "y": float(b[1])})

            pos = traj["pos"][k]
            for i in range(pos.shape[0]):
                rows.append({"step": step, "group": "pos", "idx": i, "x": float(pos[i, 0]), "y": float(pos[i, 1])})

            neg = traj["neg"][k]
            for i in range(neg.shape[0]):
                rows.append({"step": step, "group": "neg", "idx": i, "x": float(neg[i, 0]), "y": float(neg[i, 1])})

        return pd.DataFrame(rows)


    def equation4_traj_to_metrics_df(traj):
        """
        Metrics DataFrame:
          step, obj, x_b, x_pos_mean, x_neg_mean, plus min/max if you want them
        """
        return pd.DataFrame({
            "step": traj["step"],
            "obj": traj["obj"],
            "x_b": traj["x_b"],
            "x_pos_mean": [float(np.mean(x)) for x in traj["x_pos"]],
            "x_neg_mean": [float(np.mean(x)) for x in traj["x_neg"]],
            "x_pos_min": [float(np.min(x)) for x in traj["x_pos"]],
            "x_pos_max": [float(np.max(x)) for x in traj["x_pos"]],
            "x_neg_min": [float(np.min(x)) for x in traj["x_neg"]],
            "x_neg_max": [float(np.max(x)) for x in traj["x_neg"]],
        })


    def equation4_add_labels(df_long):
        df = df_long.copy()

        def make_label(row):
            g = row["group"]
            if g == "centre":
                return "centre (l)"
            if g == "booking":
                return "booked (b)"
            if g == "pos":
                return f"pos {int(row['idx']) + 1}"
            return f"neg {int(row['idx']) + 1}"

        df["label"] = df.apply(make_label, axis=1)
        return df


    rng = np.random.default_rng(7)

    v_l = rng.normal(size=2)
    v_pos_list = [rng.normal(size=2), rng.normal(size=2)]
    v_neg_list = [rng.normal(size=2), rng.normal(size=2)]
    v_b = rng.normal(size=2)

    model = Equation4Model(v_l, v_pos_list, v_neg_list, v_b, learning_rate=0.25, steps=60)
    traj = model.run_simulation()

    df_long = equation4_add_labels(equation4_traj_to_long_df(traj))
    df_metrics = equation4_traj_to_metrics_df(traj)

    print(df_long.head())
    print(df_metrics.head())
