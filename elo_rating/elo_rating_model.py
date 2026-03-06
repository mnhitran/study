#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute overall Elo ratings for Kandinsky, Flux, and SDXL.
"""

import argparse
import re
import pandas as pd
import matplotlib.pyplot as plt


MODEL_MAP = {
    "Q01": ("Kandinsky", "Flux"),
    "Q02": ("Kandinsky", "SDXL"),
    "Q03": ("Flux", "SDXL"),
}


def elo_from_matches(matches, k=20, start=1500):
    """
    Compute Elo ratings from a list of (winner, loser) matches.
    """
    ratings = {"Kandinsky": start, "Flux": start, "SDXL": start}
    for winner, loser in matches:
        ra = ratings[winner]
        rb = ratings[loser]
        ea = 1 / (1 + 10 ** ((rb - ra) / 400))
        ratings[winner] = ra + k * (1 - ea)
        ratings[loser] = rb + k * (0 - (1 - ea))
    return ratings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to the LimeSurvey CSV file") # "results_final.csv"
    parser.add_argument("--plot", default=None, help="Output filename for the plot (PNG)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    q_cols = []
    pat = re.compile(r"^(G\d{2})Q0([1-3])$")
    for col in df.columns:
        m = pat.match(col)
        if not m:
            continue
        group_num = int(m.group(1)[1:])
        if 1 <= group_num <= 57:
            q_cols.append(col)

    matches = []
    for col in q_cols:
        qtype = col[-3:]
        model_a, model_b = MODEL_MAP[qtype]
        series = df[col].dropna()
        for v in series:
            if v == "AO01":
                matches.append((model_a, model_b))
            elif v == "AO02":
                matches.append((model_b, model_a))

    ratings = elo_from_matches(matches, k=20, start=1500)

    elo_df = (
        pd.DataFrame(
            [{"Model": m, "Elo": round(r, 1)} for m, r in ratings.items()]
        )
        .sort_values("Elo", ascending=False)
        .reset_index(drop=True)
    )

    print("=== Overall ranking (Elo) ===")
    print(elo_df.to_string(index=False))
    print(f"\nNumber of included pairwise comparisons: {len(matches)}")

    if args.plot:
        plt.figure()
        plt.bar(elo_df["Model"], elo_df["Elo"])
        plt.title("Overall model ranking (Elo)")
        plt.xlabel("Model")
        plt.ylabel("Elo score")
        plt.savefig(args.plot, bbox_inches="tight")
        plt.close()
        print(f"Plot saved to: {args.plot}")


if __name__ == "__main__":
    main()