import numpy as np
import pandas as pd

from plots import objective_curve_plot, dot_product_plots, run_animation


class PositiveSamplesOnlyModel:
    """
    Minimal skip-gram 'pull term only' simulator.

    We optimise only for 3 listings (v_l being central, and vc1, vc2 being context):
        log(sigmoid(v_c1^T v_l)) + log(sigmoid(v_c2^T v_l))

    This is intentionally simplified:
    - 2D vectors (so we can visualise)
    - no negative sampling yet
    - no batching, no regularisation, no vocab distribution
    """

    def __init__(self, v_l, v_c1, v_c2):
        self.v_l = v_l
        self.v_c1 = v_c1
        self.v_c2 = v_c2

        self.learning_rate = 0.25
        self.steps = 60

        # Record trajectory for plots/animation.
        # We store vectors at every step, plus scalar diagnostics.
        self.traj = {
            "step": [0],
            "l": [v_l.copy()],
            "c1": [v_c1.copy()],
            "c2": [v_c2.copy()],
            "x1": [],  # dot(c1, l) per step (after update)
            "x2": [],  # dot(c2, l) per step (after update)
            "obj": []  # objective per step (after update)
        }

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def log_sigmoid(self, x, eps=1e-12):
        """
        log(sigmoid(x)) is always in (-inf, 0].
        0 is the maximum (achieved as x -> +inf).
        """
        s = self.sigmoid(x)
        return np.log(np.clip(s, eps, 1.0))

    def grad_log_sigmoid_dot(self, v_c, v_l):
        """
        Computes gradients of:
                log(sigmoid(v_c^T v_l))
        with respect to v_c and v_l.

        Pedagogical view:
          - We want to MAXIMISE log(sigmoid(x)), where x = v_c^T v_l.
          - The gradient (the derivative) tells us the direction of steepest increase.
          - d/dx log(sigmoid(x)) = 1 - sigmoid(x)

        This term controls how strong the update should be:
        - If x is small (embeddings poorly aligned), sigmoid(x) is small,
          so (1 - sigmoid(x)) is large -> strong update.
        - If x is large (embeddings already aligned), sigmoid(x) ~ 1,
          so (1 - sigmoid(x)) ~ 0 -> update vanishes.

        Chain rule:
          ∂/∂v_l = (1 - sigmoid(x)) * v_c
          ∂/∂v_c = (1 - sigmoid(x)) * v_l
        """
        x = float(v_c @ v_l)
        s = self.sigmoid(x)
        coeff = (1.0 - s)
        grad_v_l = coeff * v_c
        grad_v_c = coeff * v_l
        return grad_v_c, grad_v_l, x, s

    def objective(self, v_l=None, v_c1=None, v_c2=None):
        # Pull term only:
        # -> sum of log(sigmoid(dot)) for positive pairs
        # -> expanded based on 2 length contexts this is -> log(sigmoid(c1^T l)) + log(sigmoid(c2^T l))
        v_l = self.v_l if v_l is None else v_l
        v_c1 = self.v_c1 if v_c1 is None else v_c1
        v_c2 = self.v_c2 if v_c2 is None else v_c2

        x1 = float(v_c1 @ v_l)
        x2 = float(v_c2 @ v_l)
        obj = self.log_sigmoid(x1) + self.log_sigmoid(x2)

        return float(obj), x1, x2

    def run_simulation(self, record_every_step=True):
        """
        Gradient ASCENT on the pull term only.

        High-level algorithm (per step):
        1) Compute dot-products (similarity scores) for positive pairs.
        2) Compute gradients that *increase* log(sigmoid(dot)).
        3) Update vectors in the gradient direction (ascent).
        4) Record trajectory for plots/animation.
        """

        # Optional: record initial diagnostics at step 0 (after init)
        if record_every_step:
            self.record_for_animation(step=0)

        for t in range(1, self.steps + 1):
            # Step 1) Compute gradients for each positive pair
            # Pair (l, c1): how should l and c1 move to increase log(sigmoid(c1^T l)) ?
            g_c1, g_l1, x1, s1 = self.grad_log_sigmoid_dot(self.v_c1, self.v_l)

            # Pair (l, c2): similarly for the second context listing
            g_c2, g_l2, x2, s2 = self.grad_log_sigmoid_dot(self.v_c2, self.v_l)

            # Step 2) Combine gradients for the centre vector
            # The centre vector l participates in both terms, so it receives
            # the sum of both gradient contributions.
            g_l_total = g_l1 + g_l2

            # Step 3) Gradient ASCENT update (because we are maximising)
            # ascent: v <- v + lr * grad
            self.v_l = self.v_l + self.learning_rate * g_l_total
            self.v_c1 = self.v_c1 + self.learning_rate * g_c1
            self.v_c2 = self.v_c2 + self.learning_rate * g_c2

            # Step 4) Record the state for plotting/animation
            if record_every_step:
                self.record_for_animation(step=t)

        return self.traj

    def record_for_animation(self, step):
        """
        Single place where we compute diagnostics and store them.
        Keeps run_simulation() clean.

        We record:
        - current vectors
        - dot products x1, x2 (how aligned the pairs are)
        - objective value (should move upward towards 0)
        """
        obj, x1, x2 = self.objective()

        self.traj["step"].append(step)
        self.traj["l"].append(self.v_l.copy())
        self.traj["c1"].append(self.v_c1.copy())
        self.traj["c2"].append(self.v_c2.copy())
        self.traj["x1"].append(x1)
        self.traj["x2"].append(x2)
        self.traj["obj"].append(obj)

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)  # show all columns
    pd.set_option('display.width', None)  # don't wrap columns to new lines
    pd.set_option('display.max_colwidth', None)  # show full column contents

    # --- Simulation setup ---
    rng = np.random.default_rng(7)

    # One centre listing (l) and two context listings (c1, c2)
    v_l = rng.normal(size=2)
    v_c1 = rng.normal(size=2)
    v_c2 = rng.normal(size=2)

    simulationObj = PositiveSamplesOnlyModel(v_l, v_c1, v_c2)
    traj = simulationObj.run_simulation()

    df = pd.DataFrame({
        "step": traj["step"],
        "l_x": [v[0] for v in traj["l"]],
        "l_y": [v[1] for v in traj["l"]],
        "c1_x": [v[0] for v in traj["c1"]],
        "c1_y": [v[1] for v in traj["c1"]],
        "c2_x": [v[0] for v in traj["c2"]],
        "c2_y": [v[1] for v in traj["c2"]],
    })

    # These start at step 1 (after first update)
    df.loc[1:, "x1"] = traj["x1"]
    df.loc[1:, "x2"] = traj["x2"]
    df.loc[1:, "obj"] = traj["obj"]
    df = df.dropna()

    print(df.head())

    # Supporting plots
    objective_curve_plot(df, title="Objective over optimisation steps (pull term only)")
    dot_product_plots(df, title="Dot products increase as optimisation pulls embeddings together")

    run_animation(df)





