import pandas as pd
import numpy as np
import logging
import csv
import argparse
import time

from mo_distill_utils import distill
from many_objective import hyperparams_convert_back

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate MORPH models.")
    parser.add_argument("--model", type=str, required=True,
                        help="Model to test energy performance.")
    parser.add_argument("--use-flops", action='store_true',
                        help="Take models from FLOPs optimization instead of energy optimization.")
    parser.add_argument("--pareto", action='store_true',
                        help="Use Pareto front models for RQ1")
    parser.add_argument("--surrogate", action='store_true',
                        help="Measuring energy of surrogate models.")
    args = parser.parse_args()

    model_number = int(args.model)

    seed = 2

    if args.surrogate:
        # Load surrogate data
        hyperparams_data = pd.read_csv('surrogate_data_metamorphic_NEW.csv').to_numpy()
        num_objs = 3
    # Load Pareto front data and convert to array
    elif args.pareto:
        hyperparams_data = pd.read_csv('mo_pareto_front_rq1.csv').to_numpy()
        num_objs = 5
    elif args.use_flops:
        hyperparams_data = pd.read_csv('morph_results_flops.csv').to_numpy()
        num_objs = 5
    else:
        hyperparams_data = pd.read_csv('morph_results_energy.csv').to_numpy()
        num_objs = 5

    hyperparameters = hyperparams_data[:, :hyperparams_data.shape[1] - num_objs] # 2 for surrogate, 4 for final models
    objectives = hyperparams_data[:, hyperparams_data.shape[1] - num_objs:]

    if args.surrogate:
        hyperparameters = [hyperparams_convert_back(row) for row in hyperparameters]

    hyperparams = hyperparameters[model_number]
    objs = objectives[model_number]

    # RQ1: Pareto measurements
    if args.pareto:
        model_name = f"pareto_{model_number}.bin"
    else:
        # RQ2: Final models
        model_name = f"model-{model_number}.bin"


    eval_rounds = 6
    accs, _ = distill([hyperparams], eval=True, surrogate=False, seed=seed, model_name=model_name, eval_rounds=eval_rounds, use_flops=args.use_flops)

    print(accs)

if __name__ == "__main__":
    main()