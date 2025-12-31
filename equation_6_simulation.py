import numpy as np
import pandas as pd

from plots import (
    objective_curve_plot,
    dot_product_plots,
    equation3_long_df_add_labels,
    run_animation_long_df
)

from equation_3_simulation import (
    Equation3Model,
    equation3_traj_to_long_df,
    equation3_traj_to_metrics_df
)

if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    rng = np.random.default_rng(7)

    # ------------------------------------------------------------
    # 1) Toy booking "flattened" sequence (mixed tokens)
    #    U1 -> L5 -> U1 -> L8 -> U1 -> L2
    # ------------------------------------------------------------
    seq_raw = ["U1", "L5", "U1", "L8", "U1", "L2"]

    # Prefix tokens so the type is visible in plots/labels
    def pretty_token(t: str) -> str:
        if t.startswith("U"):
            return f"user:{t}"
        if t.startswith("L"):
            return f"listing:{t}"
        return t

    seq = [pretty_token(t) for t in seq_raw]

    # ------------------------------------------------------------
    # 2) Mixed vocabulary (add extra tokens so negatives are non-trivial)
    # ------------------------------------------------------------
    extra_raw = ["U2", "U3", "L1", "L3", "L6", "L9"]
    extra = [pretty_token(t) for t in extra_raw]
    vocab = sorted(set(seq + extra))

    # ------------------------------------------------------------
    # 3) Initialise embeddings for every token (2D so it is plottable)
    # ------------------------------------------------------------
    emb = {tok: rng.normal(size=2) for tok in vocab}

    # ------------------------------------------------------------
    # Helpers: build pos/neg tokens for a given centre index
    # ------------------------------------------------------------
    def build_pos_tokens(sequence, centre_idx: int, window: int):
        pos = []
        if centre_idx - window >= 0:
            pos.append(sequence[centre_idx - window])
        if centre_idx + window < len(sequence):
            pos.append(sequence[centre_idx + window])
        return pos

    def sample_neg_tokens(vocab_list, centre_tok: str, pos_toks, num_negs: int, rng_):
        forbidden = set([centre_tok] + list(pos_toks))
        candidates = [t for t in vocab_list if t not in forbidden]
        if len(candidates) < num_negs:
            raise ValueError("Not enough negative candidates. Add more tokens to vocab.")
        return list(rng_.choice(candidates, size=num_negs, replace=False))

    # ------------------------------------------------------------
    # Core runner for one case (re-uses Equation3Model as-is)
    # ------------------------------------------------------------
    def run_case(case_name: str, centre_idx: int, window: int = 1, num_negs: int = 3, steps: int = 60):
        centre_tok = seq[centre_idx]
        pos_toks = build_pos_tokens(seq, centre_idx, window)
        neg_toks = sample_neg_tokens(vocab, centre_tok, pos_toks, num_negs, rng)

        v_l = emb[centre_tok]
        v_pos_list = [emb[t] for t in pos_toks]
        v_neg_list = [emb[t] for t in neg_toks]

        print("\n" + "=" * 80)
        print(f"{case_name}")
        print("Centre token:", centre_tok)
        print("Positive context tokens:", pos_toks)
        print("Negative sample tokens:", [str(t) for t in neg_toks])

        model = Equation3Model(
            v_l=v_l,
            v_pos_list=v_pos_list,
            v_neg_list=v_neg_list,
            learning_rate=0.25,
            steps=steps,
        )
        traj = model.run_simulation()

        df_long = equation3_traj_to_long_df(traj)
        df_metrics = equation3_traj_to_metrics_df(traj)

        # ---- stop point you asked for earlier (prints) ----
        print(df_long.head())
        print(df_metrics.head())

        # ------------------------------------------------------------
        # Optional: plots/animation (no changes to plotting functions)
        # ------------------------------------------------------------
        # objective_curve_plot(df_metrics, title=f"{case_name}: objective over optimisation steps")
        # dot_product_plots(
        #     df_metrics,
        #     y_cols=["x_pos_mean", "x_neg_mean"],
        #     title=f"{case_name}: positive dot products increase while negative dot products decrease",
        # )

        df_long_labeled = equation3_long_df_add_labels(df_long)

        # Replace generic labels with semantic token names
        # centre (l) -> centre = <centre_tok>
        df_long_labeled["label"] = df_long_labeled["label"].replace({"centre (l)": f"centre = {centre_tok}"})

        # pos i -> pos = <pos_tok[i]>
        for i, pt in enumerate(pos_toks):
            df_long_labeled["label"] = df_long_labeled["label"].replace({f"pos {i+1}": f"pos = {pt}"})

        # neg i -> neg = <neg_tok[i]>
        for i, nt in enumerate(neg_toks):
            df_long_labeled["label"] = df_long_labeled["label"].replace({f"neg {i+1}": f"neg = {nt}"})

        run_animation_long_df(
            df_long_labeled,
            title=f"{case_name}",
        )

    # ------------------------------------------------------------
    # Run Case A and Case B as separate simulations
    # ------------------------------------------------------------
    # Case A: centre is user type (middle user:U1 at index 2)
    run_case(case_name="Case A (centre=user type)", centre_idx=2, window=1, num_negs=3, steps=60)

    # Case B: centre is listing type (listing:L8 at index 3)
    run_case(case_name="Case B (centre=listing type)", centre_idx=3, window=1, num_negs=3, steps=60)


