import pandas as pd
import numpy as np
import logging
import csv
import time
import argparse

from mo_distill_utils import distill_codet5, hyperparams_convert

def main():
    parser = argparse.ArgumentParser(description="Train MORPH models from Pareto front.")
    #parser.add_argument("--model", type=int, default=0,
        #               help="Model index to train from the Pareto front.")
    parser.add_argument("--start", type=int, help="Starting index for training.")
    parser.add_argument("--end", type=int, help="Ending index for training.")
    parser.add_argument("--segment", type=int, help="Results file suffix for parallel execution in HPC.")

    args = parser.parse_args()

    seed = 2

    pareto_front = pd.read_csv('mo_pareto_front_rq2.csv')
    pareto_front = pareto_front.to_numpy()[:, :]

    # Final results file
    results_file = f'pareto_front_training-{args.segment}.csv'
    fieldnames = [ "Num Hidden Layers", "Hidden Activation", "Number Decoder Layers", "Hidden Size", 
                    "Num Attention Heads", "Projection Size", "Intermediate Size", "Relative Attention Buckets",
                    "Relative Attention Max Distance", "Dropout Rate", "Feed Forward Projection",
                    "Learning Rate", "Batch Size", "Model Size", "Rouge Score", "Consumption", "Training Time", "Evaluation Time"]


    for i in range(args.start, args.end):
        hyperparams = pareto_front[i, :pareto_front.shape[1] - 3]
        objectives = pareto_front[i, pareto_front.shape[1] - 3:]


        logging.info(f"Training {i} with Consumption {objectives[2]} and size {objectives[0]}")
        if i != 0:
            start_time = time.time()
            accs, prediction_flips = distill_codet5([hyperparams], eval=False, surrogate=False, seed=seed, weights_file=f"pareto_{i}.bin")
            training_time = time.time()-start_time
            logging.info(f"Training took: {training_time} seconds")
        else:
            training_time = 0.0
        start_time = time.time()
        accs, prediction_flips = distill_codet5([hyperparams], eval=True, surrogate=False, seed=seed, weights_file=f"pareto_{i}.bin")
        evaluation_time = time.time()-start_time
        logging.info(f"Evaluation took: {evaluation_time} seconds")

        # Save final values to file
        with open(results_file, 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            # Write the header only if the file is empty
            if file.tell() == 0:
                writer.writeheader()

            # Create a dictionary from row data
            row_data = {
                "Num Hidden Layers": hyperparams[0],
                "Hidden Activation": hyperparams[1],
                "Number Decoder Layers": hyperparams[2],
                "Hidden Size": hyperparams[3],
                "Num Attention Heads": hyperparams[4],
                "Projection Size": hyperparams[5],
                "Intermediate Size": hyperparams[6],
                "Relative Attention Buckets": hyperparams[7],
                "Relative Attention Max Distance": hyperparams[8],
                "Dropout Rate": hyperparams[9],
                "Feed Forward Projection": hyperparams[10],
                "Learning Rate": hyperparams[11],
                "Batch Size": hyperparams[12],
                "Model Size": objectives[0],  # Assuming objs[0] is the Size
                "Rouge Score": accs[0],  # Assuming accs contains accuracy values
                "Consumption": objectives[2],  # Assuming objs[2] is the FLOPS
                "Training Time": training_time,
                "Evaluation Time": evaluation_time
            }

            # Write the row data to the CSV file
            writer.writerow(row_data)

if __name__ == "__main__":
    main()
