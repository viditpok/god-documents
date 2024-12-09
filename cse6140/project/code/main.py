# run `pyinstaller --onefile code/main.py --name exec` in the project directory to get the exec file

"""
main.py

Runs the TSP solver with given algorithms (BF, Approx, LS), datasets, and cutoff times.
For LS, experiments are repeated with different random seeds. Results are saved in a .sol file with metrics
including solution quality, and the list of vertices.
"""

import argparse
import os
import time
from exact_solution import exact_tsp
from approx_solution import approximate_tsp
from local_search import simulated_annealing_tsp_with_approximation, generate_2_approximation
from utils import parse_data, validate_points
import random


def main():
    # setup argument parser
    parser = argparse.ArgumentParser(description="TSP Solver")
    parser.add_argument("-inst", required=True, help="filename of the dataset")
    parser.add_argument(
        "-alg", required=True, choices=["BF", "Approx", "LS"], help="algorithm to use"
    )
    parser.add_argument(
        "-time", required=False, type=int, help="cutoff time in seconds"
    )
    parser.add_argument(
        "-seed",
        type=int,
        required=False,
        help="random seed (only used for Approx and LS)",
    )

    args = parser.parse_args()

    # parse the data file
    try:
        points = parse_data(args.inst)
        validate_points(points)
    except Exception as e:
        print(f"Error parsing input file: {e}")
        return

    # set the seed for Approx and LS algorithms only
    if args.alg in ["Approx", "LS"]:
        if args.seed is None:
            raise ValueError("Seed is required for Approx and LS algorithms")
        random.seed(args.seed)

    # determine which algorithm to run
    if args.alg == "BF":
        start_time = time.time()
        total_distance, path = exact_tsp(points, args.time)
        end_time = time.time()

    elif args.alg == "Approx":
        total_distance, path = approximate_tsp(points, seed=args.seed)
        end_time = time.time()
        start_time = end_time

    elif args.alg == "LS":
        # generate a 2-approximation to use as the starting solution
        try:
            approx_tour = generate_2_approximation(points)
        except Exception as e:
            print(f"Error generating 2-approximation: {e}")
            return

        initial_temperature = 10000
        cooling_rate = 0.995
        start_time = time.time()
        total_distance, path = simulated_annealing_tsp_with_approximation(
            approx_tour, initial_temperature, cooling_rate, args.time, seed=args.seed
        )
        end_time = time.time()

    # output directory
    output_dir = os.path.abspath(os.path.join(os.getcwd(), "output"))
    os.makedirs(output_dir, exist_ok=True)

    # construct the output file name based on algorithm-specific rules
    instance_name = os.path.splitext(os.path.basename(args.inst))[0].lower()
    method = args.alg
    cutoff = args.time
    if method == "BF":
        output_filename = f"{instance_name}_{method}_{cutoff}.sol"
    elif method == "Approx":
        output_filename = f"{instance_name}_{method}_{args.seed}.sol"
    elif method == "LS":
        output_filename = f"{instance_name}_{method}_{cutoff}_{args.seed}.sol"

    output_filepath = os.path.join(output_dir, output_filename)

    # write the output to the file
    with open(output_filepath, "w") as f:
        f.write(f"{total_distance}\n")
        f.write(",".join(map(str, path)) + "\n")

    # print runtime info
    print(f"Execution Time: {end_time - start_time:.4f} seconds")
    print(f"Total Distance: {total_distance}")
    print(f"Output saved to: {output_filepath}")


if __name__ == "__main__":
    main()