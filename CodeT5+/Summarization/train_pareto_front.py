import pandas as pd
import numpy as np
import logging
import csv
import time
import argparse

from mo_distill_utils import distill, hyperparams_convert

def main():
    #parser = argparse.ArgumentParser(description="Train MORPH models from Pareto front.")
    #parser.add_argument("--model", type=int, default=0,
    #                    help="Model index to train from the Pareto front.")
    
    #args = parser.parse_args()

    seed = 2

    pareto_front = pd.read_csv('mo_pareto_front_rq2.csv')
    pareto_front = pareto_front.to_numpy()[:, :]

    # Final results file
    results_file = 'pareto_front_training.csv'
    fieldnames = [
            "Tokenizer", "Vocab Size", "Num Hidden Layers", "Hidden Size", "Hidden Act", "Hidden Dropout Prob",
            "Intermediate Size", "Num Attention Heads", "Attention Probs Dropout Prob", "Max Sequence Length",
            "Position Embedding Type", "Learning Rate", "Batch Size", "Size", "Accuracy", "FLOPS", "Flips", "Training Time", "Evaluation Time"
    ]

    for i in range(pareto_front.shape[0]):
        hyperparams = pareto_front[i, :pareto_front.shape[1] - 3]
        objectives = pareto_front[i, pareto_front.shape[1] - 3:]


        logging.info(f"Training {i} with Consumption {objectives[2]} and size {objectives[0]}")
        start_time = time.time()
        accs, prediction_flips = distill([hyperparams], eval=False, surrogate=False, seed=seed, model_name=f"pareto_{i}.bin")
        training_time = time.time()-start_time
        logging.info(f"Training took: {training_time} seconds")
        start_time = time.time()
        accs, prediction_flips = distill([hyperparams], eval=True, surrogate=False, seed=seed, model_name=f"pareto_{i}.bin")
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
                "Tokenizer": hyperparams[0],
                "Vocab Size": hyperparams[1],
                "Num Hidden Layers": hyperparams[2],
                "Hidden Size": hyperparams[3],
                "Hidden Act": hyperparams[4],
                "Hidden Dropout Prob": hyperparams[5],
                "Intermediate Size": hyperparams[6],
                "Num Attention Heads": hyperparams[7],
                "Attention Probs Dropout Prob": hyperparams[8],
                "Max Sequence Length": hyperparams[9],
                "Position Embedding Type": hyperparams[10],
                "Learning Rate": hyperparams[11],
                "Batch Size": hyperparams[12],
                "Size": objectives[0],  # Assuming objs[0] is the Size
                "Accuracy": accs[0],  # Assuming accs contains accuracy values
                "Consumption": objectives[2],  # Assuming objs[3] is the FLOPS
                "Training Time": training_time,
                "Evaluation Time": evaluation_time
            }

            # Write the row data to the CSV file
            writer.writerow(row_data)

if __name__ == "__main__":
    main()