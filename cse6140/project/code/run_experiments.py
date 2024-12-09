"""
run_experiments.py

Runs the TSP solver using the executable with various algorithms (BF, Approx, LS), datasets, and cutoff times.
For LS, experiments are repeated with different random seeds. Results are saved in a CSV file with metrics
including solution quality, runtime, and relative error.
"""

import csv
import os
import subprocess
import time
import pandas as pd
from main import parse_data

# paths and settings
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
EXECUTABLE_PATH = os.path.join(PROJECT_DIR, "dist", "exec")
DATASET_PATH = os.path.join(SCRIPT_DIR, "test_cases")
OUTPUT_PATH = os.path.join(PROJECT_DIR, "output")
CSV_FILE = os.path.join(PROJECT_DIR, "result_table.csv")
ALGORITHMS = ["BF", "Approx", "LS"]
CUTOFFS = [5, 10, 15, 20, 25, 50, 75, 100]
SEEDS = range(42, 52)

# ensure output directory exists
os.makedirs(OUTPUT_PATH, exist_ok=True)

# datasets to test
datasets = [file for file in os.listdir(DATASET_PATH) if file.endswith(".tsp")]


# function to parse the output file and extract solution quality
def parse_output_file(output_filepath, points):
    with open(output_filepath, "r") as f:
        lines = f.readlines()

        # solution quality
        sol_quality = float(lines[0].strip())

        # extract tour, split by commas
        tour_nodes = lines[1].strip().split(",")

        # determine if it's a full tour
        expected_num_nodes = len(points)
        full_tour = "Yes" if len(tour_nodes) == expected_num_nodes else "No"

    return sol_quality, full_tour


# function to run a single experiment and gather results
def run_experiment(dataset, alg, cutoff, seed=None):
    instance_name = os.path.splitext(dataset)[0].lower()

    points = parse_data(os.path.join(DATASET_PATH, dataset))

    # set output filename based on algorithm rules
    if alg == "BF":
        output_file = f"{instance_name}_{alg}_{cutoff}.sol"
    elif alg == "Approx":
        output_file = f"{instance_name}_{alg}_{seed}.sol"
    elif alg == "LS":
        output_file = f"{instance_name}_{alg}_{cutoff}_{seed}.sol"
    output_filepath = os.path.join(OUTPUT_PATH, output_file)

    # build the command to run the executable
    command = [
        EXECUTABLE_PATH,
        "-inst",
        os.path.join(DATASET_PATH, dataset),
        "-alg",
        alg,
    ]

    # add the cutoff argument for BF and LS algorithms
    if alg in ["BF", "LS"]:
        command.extend(["-time", str(cutoff)])

    # add the seed argument only for LS and Approx algorithms
    if alg in ["Approx", "LS"] and seed is not None:
        command.extend(["-seed", str(seed)])

    # run the command and time it
    start_time = time.time()
    subprocess.run(command, check=True)
    end_time = time.time()

    # calculate elapsed time
    elapsed_time = end_time - start_time

    # parse output file for solution quality
    sol_quality, full_tour = parse_output_file(output_filepath, points)

    return {
        "Dataset": dataset,
        "Algorithm": alg,
        "Cutoff": cutoff if alg != "Approx" else None,
        "Seed": seed if alg != "BF" else None,
        "Time": round(elapsed_time, 4),
        "Sol.Quality": sol_quality,
        "Full Tour": full_tour,
    }


# function to update or append results in the CSV
def update_or_append_result(new_result):
    try:
        # Load the existing results
        existing_df = pd.read_csv(CSV_FILE)

        # Check if the result exists
        mask = (
            (existing_df["Dataset"] == new_result["Dataset"])
            & (existing_df["Algorithm"] == new_result["Algorithm"])
            & (existing_df["Cutoff"] == new_result["Cutoff"])
            & (existing_df["Seed"] == new_result["Seed"])
        )

        if existing_df[mask].empty:  # if the result doesn't exist, append it
            new_df = pd.DataFrame([new_result])
            existing_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:  # replace the existing result
            existing_df.loc[mask, ["Time", "Sol.Quality", "Full Tour"]] = [
                new_result["Time"],
                new_result["Sol.Quality"],
                new_result["Full Tour"],
            ]

        # Write the updated dataframe back to CSV
        existing_df.to_csv(CSV_FILE, index=False)
        print(f"Result saved or updated in {CSV_FILE}")
    except FileNotFoundError:
        # If the CSV file doesn't exist, create it and write the new result
        pd.DataFrame([new_result]).to_csv(CSV_FILE, index=False)
        print(f"File not found, creating {CSV_FILE} and saving result.")


# main function to run all experiments and save to CSV
def main():
    # run experiments for each dataset, algorithm, and cutoff time
    for dataset in datasets:
        for alg in ALGORITHMS:
            if alg == "Approx":
                for seed in SEEDS:
                    result = run_experiment(dataset, alg, None, seed)
                    update_or_append_result(result)

            elif alg == "LS":
                for cutoff in CUTOFFS:
                    for seed in SEEDS:
                        result = run_experiment(dataset, alg, cutoff, seed)
                        update_or_append_result(result)

            elif alg == "BF":
                for cutoff in CUTOFFS:
                    result = run_experiment(dataset, alg, cutoff)
                    update_or_append_result(result)


if __name__ == "__main__":
    main()
