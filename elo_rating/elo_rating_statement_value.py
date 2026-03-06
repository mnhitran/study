#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute Elo ratings per statement (G01-G57) and per value (V01-V19).
"""

import argparse
import re
import pandas as pd


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


def collect_group_columns(df):
    """
    Collect relevant survey columns for groups G01-G57.
    """
    pattern = re.compile(r"^(G\d{2})Q0([1-3])$")
    group_cols = {}

    for col in df.columns:
        match = pattern.match(col)
        if not match:
            continue

        group_code = match.group(1)
        group_num = int(group_code[1:])

        if 1 <= group_num <= 57:
            group_cols.setdefault(group_code, []).append(col)

    return group_cols


def matches_from_group(df, cols):
    """
    Convert one group's responses into (winner, loser) matches.
    """
    matches = []

    for col in cols:
        qtype = col[-3:]
        model_a, model_b = MODEL_MAP[qtype]
        series = df[col].dropna()

        for value in series:
            if value == "AO01":
                matches.append((model_a, model_b))
            elif value == "AO02":
                matches.append((model_b, model_a))

    return matches


def compute_elo_per_statement(df, group_cols):
    """
    Compute Elo ratings for each statement group.
    """
    rows = []

    for group_code in sorted(group_cols.keys(), key=lambda x: int(x[1:])):
        cols = [c for c in group_cols[group_code] if c.endswith(("Q01", "Q02", "Q03"))]
        matches = matches_from_group(df, cols)

        if not matches:
            continue

        ratings = elo_from_matches(matches, k=20, start=1500)
        best_model = max(ratings.items(), key=lambda x: x[1])[0]

        rows.append({
            "Group": group_code,
            "Kandinsky": round(ratings["Kandinsky"], 1),
            "Flux": round(ratings["Flux"], 1),
            "SDXL": round(ratings["SDXL"], 1),
            "Best Model": best_model,
            "Comparisons Used": len(matches),
        })

    return pd.DataFrame(rows)


def compute_elo_per_value(df, group_cols):
    """
    Compute Elo ratings for each value based on three statement groups.
    """
    rows = []

    for value_num in range(1, 20):
        start_group = (value_num - 1) * 3 + 1
        groups = [f"G{str(i).zfill(2)}" for i in range(start_group, start_group + 3)]

        matches = []
        present_groups = []

        for group_code in groups:
            if group_code not in group_cols:
                continue

            present_groups.append(group_code)
            cols = [c for c in group_cols[group_code] if c.endswith(("Q01", "Q02", "Q03"))]
            matches.extend(matches_from_group(df, cols))

        if not matches:
            continue

        ratings = elo_from_matches(matches, k=20, start=1500)
        best_model = max(ratings.items(), key=lambda x: x[1])[0]

        rows.append({
            "Value": f"V{str(value_num).zfill(2)}",
            "Groups": ", ".join(present_groups),
            "Kandinsky": round(ratings["Kandinsky"], 1),
            "Flux": round(ratings["Flux"], 1),
            "SDXL": round(ratings["SDXL"], 1),
            "Best Model": best_model,
            "Comparisons Used": len(matches),
        })

    return pd.DataFrame(rows).sort_values("Value").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to the LimeSurvey CSV file") # "results_final.csv"
    parser.add_argument(
        "--statement_out",
        default="elo_per_statement.csv",
        help="Output file for statement-level Elo ratings",
    )
    parser.add_argument(
        "--value_out",
        default="elo_per_value.csv",
        help="Output file for value-level Elo ratings",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    group_cols = collect_group_columns(df)

    elo_per_statement = compute_elo_per_statement(df, group_cols)
    elo_per_statement.to_csv(args.statement_out, index=False)
    print(f"Saved statement-level Elo ratings to: {args.statement_out}")

    elo_per_value = compute_elo_per_value(df, group_cols)
    elo_per_value.to_csv(args.value_out, index=False)
    print(f"Saved value-level Elo ratings to: {args.value_out}")


if __name__ == "__main__":
    main()