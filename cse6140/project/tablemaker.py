"""
tablemaker.py

Converts the values in result_table.csv into the corresponding tables for each
given algorithm and provides the necessary parameters to analyze the results.
"""

import pandas as pd

# load results
results_df = pd.read_csv("result_table.csv")

# find best ls solution per dataset and cutoff
best_ls_quality = (
    results_df[results_df["Algorithm"] == "LS"]
    .groupby(["Dataset", "Cutoff"])["Sol.Quality"]
    .min()
)

# calculate relerror for bf
results_df["RelError_BF"] = results_df.apply(
    lambda row: (
        abs(best_ls_quality.loc[(row["Dataset"], row["Cutoff"])] / row["Sol.Quality"])
        if row["Algorithm"] == "BF" and pd.notna(row["Cutoff"])
        else 0.0
    ),
    axis=1,
)

# find best ls solution per dataset and seed for approx
best_ls_quality_approx = (
    results_df[results_df["Algorithm"] == "LS"]
    .groupby(["Dataset", "Seed"])["Sol.Quality"]
    .min()
)

# calculate relerror for approx
results_df["RelError_Approx"] = results_df.apply(
    lambda row: (
        abs(
            best_ls_quality_approx.loc[(row["Dataset"], row["Seed"])]
            / row["Sol.Quality"]
        )
        if row["Algorithm"] == "Approx" and pd.notna(row["Seed"])
        else 0.0
    ),
    axis=1,
)

# calculate relerror for ls
results_df["RelError_LS"] = results_df.apply(
    lambda row: (
        abs(best_ls_quality[row["Dataset"], row["Cutoff"]] - row["Sol.Quality"])
        / best_ls_quality[row["Dataset"], row["Cutoff"]]
        if row["Algorithm"] == "LS" and pd.notna(row["Cutoff"])
        else 0.0
    ),
    axis=1,
)

# separate results into bf, approx, and ls
bf_df = results_df[results_df["Algorithm"] == "BF"]
approx_df = results_df[results_df["Algorithm"] == "Approx"]
ls_df = results_df[results_df["Algorithm"] == "LS"]

# group and aggregate bf results
bf_table = (
    bf_df.groupby(["Dataset", "Cutoff"])
    .agg(
        {
            "Time": "mean",
            "Sol.Quality": "mean",
            "Full Tour": "first",
            "RelError_BF": "mean",
        }
    )
    .reset_index()
)

# group and aggregate approx results
approx_table = (
    approx_df.groupby(["Dataset", "Seed"])
    .agg(
        {
            "Time": "mean",
            "Sol.Quality": "mean",
            "Full Tour": "first",
            "RelError_Approx": "mean",
        }
    )
    .reset_index()
)

# group and aggregate ls results
ls_table = (
    ls_df.groupby(["Dataset", "Cutoff"])
    .agg(
        {
            "Time": "mean",
            "Sol.Quality": "mean",
            "Full Tour": "first",
            "RelError_LS": "mean",
        }
    )
    .reset_index()
)

# save results to csv
bf_table.to_csv("bf_results.csv", index=False)
approx_table.to_csv("approx_results.csv", index=False)
ls_table.to_csv("ls_results.csv", index=False)

# print confirmation
print("CSV files saved: bf_results.csv, approx_results.csv, ls_results.csv")
