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
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    rng = np.random.default_rng(7)

    # -----------------------------
    # 1) Toy booking "flattened" sequence (mixed tokens)
    #    U1 -> L5 -> U1 -> L8 -> U1 -> L2
    # -----------------------------
    seq = ["U1", "L5", "U1", "L8", "U1", "L2"]

    # -----------------------------
    # 2) Create a mixed vocabulary.
    #    In a real system, this would include many user_types and listing_types.
    #    We add a few extra tokens so negative sampling is non-trivial.
    # -----------------------------
    vocab = sorted(set(seq + ["U2", "U3", "L1", "L3", "L6", "L9"]))

    # -----------------------------
    # 3) Initialise embeddings for every token in the vocab (2Dimensions so it is plottable)
    # -----------------------------
    emb = {tok: rng.normal(size=2) for tok in vocab}

    # -----------------------------
    # 4) Choose a centre position and window size
    # -----------------------------
    window = 1
    centre_idx = 2  # this is the middle "U1" in the sequence above
    centre_tok = seq[centre_idx]

    # Positive context tokens from the window
    pos_toks = []
    if centre_idx - window >= 0:
        pos_toks.append(seq[centre_idx - window])
    if centre_idx + window < len(seq):
        pos_toks.append(seq[centre_idx + window])

    # -----------------------------
    # 5) Sample negatives from the mixed vocab
    #    - exclude the centre token itself
    #    - exclude true positive context tokens
    # -----------------------------
    num_negs = 3
    forbidden = set([centre_tok] + pos_toks)
    neg_candidates = [t for t in vocab if t not in forbidden]

    if len(neg_candidates) < num_negs:
        raise ValueError("Not enough negative candidates. Add more tokens to vocab.")

    neg_toks = list(rng.choice(neg_candidates, size=num_negs, replace=False))

    # -----------------------------
    # 6) Build vectors for Equation3Model (reused as-is)
    # -----------------------------
    v_l = emb[centre_tok]
    v_pos_list = [emb[t] for t in pos_toks]
    v_neg_list = [emb[t] for t in neg_toks]

    print("Centre token:", centre_tok)
    print("Positive context tokens:", pos_toks)
    print("Negative sample tokens:", [str(t) for t in neg_toks])

    # -----------------------------
    # 7) Run the existing simulation (Equation 3 == Equations 6/7 in this setting)
    # -----------------------------
    model = Equation3Model(
        v_l=v_l,
        v_pos_list=v_pos_list,
        v_neg_list=v_neg_list,
        learning_rate=0.25,
        steps=60
    )
    traj = model.run_simulation()

    # -----------------------------
    # 8) Stop exactly where you requested:
    #    build the dataframes and print heads.
    # -----------------------------
    df_long = equation3_traj_to_long_df(traj)
    df_metrics = equation3_traj_to_metrics_df(traj)

    print(df_long.head())
    print(df_metrics.head())

    
