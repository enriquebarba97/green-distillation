import pandas as pd
import numpy as np
import logging
import csv
import argparse
import time

from mo_distill_utils import distill_codet5, hyperparams_convert_back_codet5

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate MORPH models.")
    parser.add_argument("--model", type=str, required=True,
                        help="Model to test energy performance.")
    args = parser.parse_args()

    model_number = int(args.model)

    seed = 2

    # Load Pareto front data and convert to array
    hyperparams_data = pd.read_csv('surrogate_data_large.csv').to_numpy()

    hyperparameters = hyperparams_data[:, :hyperparams_data.shape[1] - 2]
    objectives = hyperparams_data[:, hyperparams_data.shape[1] - 2:]

    hyperparameters = [hyperparams_convert_back_codet5(row) for row in hyperparameters]

    hyperparams = hyperparameters[model_number]
    objs = objectives[model_number]

    eval_rounds = 10
    for i in range(eval_rounds):
        accs, _ = distill_codet5([hyperparams], eval=True, surrogate=True, seed=seed, model_name=f"model-{model_number}.bin")

    print(accs)

if __name__ == "__main__":
    main()