import numpy as np
import pandas as pd

from plots import (
    objective_curve_plot,
    dot_product_plots,
    three_point_df_to_long,
    run_animation_long_df
)

class NegativeSamplesOnlyModel:
    """
    Minimal skip-gram 'push term only' simulator.

    We optimise only for 3 listings (v_l being central, and vc1, vc2 being NEGATIVE samples):
        log(sigmoid(-v_c1^T v_l)) + log(sigmoid(-v_c2^T v_l))

    Equivalent expanded form (Airbnb-style):
        log( 1 / (1 + exp( v_c1^T v_l )) ) + log( 1 / (1 + exp( v_c2^T v_l )) )

    This is intentionally simplified:
    - 2D vectors (so we can visualise)
    - no positive samples yet
    - no batching, no regularisation, no vocab distribution
    """

    def __init__(self, v_l, v_c1, v_c2, learning_rate=0.25, steps=60):
        self.v_l = np.array(v_l, dtype=float)
        self.v_c1 = np.array(v_c1, dtype=float)
        self.v_c2 = np.array(v_c2, dtype=float)

        self.learning_rate = float(learning_rate)
        self.steps = int(steps)

        # Record trajectory for plots/animation.
        # We store vectors at every step, plus scalar diagnostics.
        self.traj = {
            "step": [0],
            "l":  [self.v_l.copy()],
            "c1": [self.v_c1.copy()],
            "c2": [self.v_c2.copy()],
            "x1": [],   # dot(c1, l) per step (after update)
            "x2": [],   # dot(c2, l) per step (after update)
            "obj": []   # objective per step (after update)
        }

    # --- maths helpers ---
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def log_sigmoid(self, x, eps=1e-12):
        """
        log(sigmoid(x)) is always in (-inf, 0].
        0 is the maximum (achieved as x -> +inf).
        """
        s = self.sigmoid(x)
        return np.log(np.clip(s, eps, 1.0))

    def grad_log_sigmoid_neg_dot(self, v_c, v_l):
        """
        Computes gradients of the NEGATIVE (push) term:
            log(sigmoid(- v_c^T v_l))

        Let x = v_c^T v_l.
        Then the term is log(sigmoid(-x)).

        Key derivative:
            d/dx [ log(sigmoid(-x)) ] = - (1 - sigmoid(-x))

        Interpretation:
        - We still MAXIMISE the objective, so we update via gradient ASCENT.
        - The minus sign means the update tends to *decrease* x = v_c^T v_l,
          pushing the vectors to become less aligned.

        Update strength intuition:
        - If x is large (vectors too aligned), sigmoid(-x) is small,
          so (1 - sigmoid(-x)) is large -> strong push away.
        - If x is very negative already (vectors well separated),
          sigmoid(-x) ~ 1 so (1 - sigmoid(-x)) ~ 0 -> update vanishes.

        Chain rule:
            ∂/∂v_l = -(1 - sigmoid(-x)) * v_c
            ∂/∂v_c = -(1 - sigmoid(-x)) * v_l
        """
        x = float(v_c @ v_l)
        s = self.sigmoid(-x)            # sigmoid(-x)
        coeff = -(1.0 - s)              # negative sign drives "push"
        grad_v_l = coeff * v_c
        grad_v_c = coeff * v_l
        return grad_v_c, grad_v_l, x, s

    def objective(self, v_l=None, v_c1=None, v_c2=None):
        """
        Push term only:
            log(sigmoid(-c1^T l)) + log(sigmoid(-c2^T l))
        """
        v_l = self.v_l if v_l is None else v_l
        v_c1 = self.v_c1 if v_c1 is None else v_c1
        v_c2 = self.v_c2 if v_c2 is None else v_c2

        x1 = float(v_c1 @ v_l)
        x2 = float(v_c2 @ v_l)
        obj = self.log_sigmoid(-x1) + self.log_sigmoid(-x2)
        return float(obj), x1, x2

    def record_for_animation(self, step):
        """
        Records current vectors and scalar diagnostics.
        """
        obj, x1, x2 = self.objective()

        self.traj["step"].append(step)
        self.traj["l"].append(self.v_l.copy())
        self.traj["c1"].append(self.v_c1.copy())
        self.traj["c2"].append(self.v_c2.copy())
        self.traj["x1"].append(x1)
        self.traj["x2"].append(x2)
        self.traj["obj"].append(obj)

    def run_simulation(self, record_every_step=True):
        """
        Gradient ASCENT on the push term only.

        High-level algorithm (per step):
        1) Compute dot-products x = v_c^T v_l for negative pairs.
        2) Compute gradients that *increase* log(sigmoid(-x)),
           which effectively decreases x.
        3) Update vectors in the gradient direction (ascent).
        4) Record trajectory for plots/animation.
        """
        if record_every_step:
            self.record_for_animation(step=0)

        for t in range(1, self.steps + 1):
            # Step 1) Gradients for each negative pair
            g_c1, g_l1, x1, s1 = self.grad_log_sigmoid_neg_dot(self.v_c1, self.v_l)
            g_c2, g_l2, x2, s2 = self.grad_log_sigmoid_neg_dot(self.v_c2, self.v_l)

            # Step 2) Combine gradients for centre vector
            g_l_total = g_l1 + g_l2

            # Step 3) Gradient ASCENT update (maximisation)
            self.v_l  = self.v_l  + self.learning_rate * g_l_total
            self.v_c1 = self.v_c1 + self.learning_rate * g_c1
            self.v_c2 = self.v_c2 + self.learning_rate * g_c2

            # Step 4) Record
            if record_every_step:
                self.record_for_animation(step=t)

        return self.traj

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

    simulationObj = NegativeSamplesOnlyModel(v_l, v_c1, v_c2)
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
    objective_curve_plot(df, title="Objective over optimisation steps (push term only)")
    dot_product_plots(df, y_cols=["x1", "x2"], title="Dot products decrease as optimisation push embeddings apart")

    df_long_neg = three_point_df_to_long(df)
    run_animation_long_df(
        df_long_neg,
        title="Negative samples only: embeddings are pushed apart",
    )
