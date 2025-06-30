import csv
import logging
import os
import time
import warnings

import numpy as np
import pandas as pd
#from numba.core.errors import NumbaDeprecationWarning
from pyDOE import lhs
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.moo.nsga2 import binary_tournament
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.core.sampling import Sampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions

from mo_distill_utils import distill, distill_codet5
from flops import TransformerHparams, EncoderDecoderHparams
from mo_surrogate import SurrogateModel
from mo_distill_utils import hyperparams_convert_codet5, hyperparams_convert_back_codet5
from utils import set_seed

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)

# Ignore specific warning
#warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)

def hyperparams_convert_back(hyperparams):
    # Reverse mappings
    tokenizer_type_inv = {"BPE": 1, "WordPiece": 2, "Unigram": 3, "Word": 4}
    hidden_act_inv = {"gelu": 1, "relu": 2, "silu": 3, "gelu_new": 4}
    position_embedding_type_inv = {"absolute": 1, "relative_key": 2, "relative_key_query": 3}
    learning_rate_inv = {1e-3: 1, 1e-4: 2, 5e-5: 3}
    batch_size_inv = {8: 1, 16: 2}

    return [
        tokenizer_type_inv[hyperparams[0]],  # Reverse conversion for 'Tokenizer'
        hyperparams[1],  # 'Vocab Size' remains the same (assuming it's numeric)
        hyperparams[2],  # 'Num Hidden Layers' remains the same (assuming it's numeric)
        hyperparams[3],  # 'Hidden Size' remains the same (assuming it's numeric)
        hidden_act_inv[hyperparams[4]],  # Reverse conversion for 'Hidden Act'
        hyperparams[5],  # 'Hidden Dropout Prob' remains the same (assuming it's numeric)
        hyperparams[6],  # 'Intermediate Size' remains the same (assuming it's numeric)
        hyperparams[7],  # 'Num Attention Heads' remains the same (assuming it's numeric)
        hyperparams[8],  # 'Attention Probs Dropout Prob' remains the same (assuming it's numeric)
        hyperparams[9],  # 'Max Sequence Length' remains the same (assuming it's numeric)
        position_embedding_type_inv[hyperparams[10]],  # Reverse conversion for 'Position Embedding Type'
        learning_rate_inv[hyperparams[11]],  # Reverse conversion for 'Learning Rate'
        batch_size_inv[hyperparams[12]]  # Reverse conversion for 'Batch Size'
    ]


def convert_chromosomes(population):
    """
        Converts the chromosomes of a given population into a specific format suitable for surrogate modeling.

        This function iterates through each individual in the population, rounding certain genes to a specified
        precision and ensuring others are integers. Specifically, the genes at indices 5 and 8 are rounded to
        one decimal place, while the rest are rounded to the nearest integer.

        Parameters:
        - population (list of lists): A population where each individual is represented as a list of genes (chromosomes).
                                      Each gene in an individual can be a float or an integer.

        Process:
        1. Iterates through each individual in the population.
        2. For each gene in an individual:
           - If the gene's index is 5 or 8, it is rounded to one decimal place and added to a temporary list.
           - Otherwise, the gene is rounded to the nearest integer and added to the temporary list.
        3. The modified individual is then added to a new list representing the converted population.

        Returns:
        - surrogate_data (list of lists): A new population list where each individual's genes have been converted
                                          according to the specified rules. This format is typically used for
                                          preparing data for surrogate models in optimization tasks.

        Note:
        - The function assumes the population list is well-formed, with each individual containing at least 9 genes.
        - Indices are 0-based, meaning the 6th and 9th genes are at indices 5 and 8, respectively.
    """
    surrogate_data = []
    for each_pop in population:
        typed_chromosome = []
        for idx in range(len(each_pop)):
            if idx == 9:
                typed_chromosome.append(round(each_pop[9], 1))
            else:
                typed_chromosome.append(int(round(each_pop[idx])))

        surrogate_data.append(typed_chromosome)
    return surrogate_data


class ModelCompressionProblem(Problem):

    def __init__(self, lower_bounds, upper_bounds, surrogate_model: SurrogateModel):
        super().__init__(n_var=len(lower_bounds), n_obj=3, n_ieq_constr=0, xl=lower_bounds, xu=upper_bounds)
        self.generation = 0
        self.surrogate_model = surrogate_model

    def _evaluate(self, population, out, *args, **kwargs):
        # Initialize an array to hold the objective values for each solution in X
        F = np.zeros((len(population), 3))  # For 4 objectives

        # Open the file for appending outside the loop to avoid opening and closing it multiple times
        with open("generation_data.csv", "a", newline='') as f:
            writer = csv.writer(f)
            if self.generation == 0:  # Write headers only once
                writer.writerow(["Generation", "Chromosome", "Objective 1", "Objective 2", "Objective 3"])

            for idx in range(0, len(population)):
                candidate_values = population[idx]

                model = EncoderDecoderHparams(candidate_values[3], candidate_values[0], candidate_values[2],
                                              320, 32000, candidate_values[6], candidate_values[4], candidate_values[5], 128)

                size = model.get_params() * 4 / 1e6
                flops = model.get_infer_flops() / 1e9
                accuracy = self.surrogate_model.predict_accuracy([candidate_values])[0]
                gpu_energy = self.surrogate_model.predict_gpu_energy([candidate_values])[0]
                cpu_energy = self.surrogate_model.predict_cpu_energy([candidate_values])[0]

                F[idx, :] = [size, -accuracy, gpu_energy + cpu_energy]

                # Write data for each individual immediately after evaluation
                writer.writerow([self.generation, candidate_values.tolist(), F[idx, 0], F[idx, 1], F[idx, 2]])

        # Reshape for compatibility with pymoo's expected output format
        out["F"] = F.reshape(-1, 1)

        self.generation += 1  # Increment generation counter

class LatinHypercubeSampler(Sampling):
    def _do(self, search_problem: Problem, n_samples, **kwargs):
        # Number of points to generate
        lb, ub = search_problem.xl, search_problem.xu
        n_parameters = len(lb)

        # Generate Latin Hypercube samples
        lhs_samples = lhs(n_parameters, samples=n_samples)

        repair = MyRepairCodeT5()
        problem = ModelCompressionProblem(lb, ub, None)

        # Scale the samples to the provided bounds
        scaled_samples = np.zeros_like(lhs_samples)
        for i in range(n_parameters):
            scaled_samples[:, i] = lb[i] + lhs_samples[:, i] * (ub[i] - lb[i])
        #sampling = FloatRandomSampling()

        #scaled_samples = sampling._do(problem, n_samples)

        # fix the chromosome to satisfy the constraints
        scaled_samples = repair._do(problem, scaled_samples)
        initial_population = []
        for i in range(n_samples):
            initial_population.append(scaled_samples[i, :])
        return initial_population


class MyRepair(Repair):

    def _do(self, search_problem, X, **kwargs):
        for i in range(0, X.shape[0]):
            individual = X[i]

            # Let's fix the chromosome datatype
            for gene_index in range(0, search_problem.n_var):
                if gene_index == 5 or gene_index == 8:
                    individual[gene_index] = round(individual[gene_index], 1)
                else:
                    int(round(individual[gene_index]))

            individual[3] = int(individual[3])
            individual[7] = int(individual[7])
            if individual[3] % individual[7] != 0:
                # Adjust individual[3] to the closest value that is divisible by individual[7]
                # There are two possible closest values:
                # 1. Next multiple of individual[7]
                # 2. Previous multiple of individual[7]

                next_multiple = (individual[3] // individual[7] + 1) * individual[7]
                prev_multiple = (individual[3] // individual[7]) * individual[7]

                # Choose the closest one
                if abs(individual[3] - next_multiple) < abs(individual[3] - prev_multiple):
                    individual[3] = next_multiple
                else:
                    individual[3] = prev_multiple

        return X
    
class MyRepairCodeT5(Repair):

    def _do(self, search_problem, X, **kwargs):
        for i in range(0, X.shape[0]):
            individual = X[i]
            #print("Individual before repair: ", individual)
            # Let's fix the chromosome datatype
            for gene_index in range(0, search_problem.n_var):
                if gene_index == 9:
                    individual[gene_index] = round(individual[gene_index], 1)
                else:
                    individual[gene_index] = int(round(individual[gene_index]))

            # print("Individual after datatype fix: ", individual)
            individual[3] = int(individual[3])
            individual[4] = int(individual[4])
            if individual[3] % individual[4] != 0:
                # Adjust individual[3] to the closest value that is divisible by individual[7]
                # There are two possible closest values:
                # 1. Next multiple of individual[7]
                # 2. Previous multiple of individual[7]

                next_multiple = (individual[3] // individual[4] + 1) * individual[4]
                prev_multiple = (individual[3] // individual[4]) * individual[4]

                # Choose the closest one
                if abs(individual[3] - next_multiple) < abs(individual[3] - prev_multiple):
                    individual[3] = next_multiple
                else:
                    individual[3] = prev_multiple

        return X


def main():
    start_time = time.time()
    lb = [1, 1, 1, 16, 1, 1, 16, 4, 32, 0.1, 1, 1, 1]
    ub = [6, 4, 6, 512, 8, 64, 2048, 32, 128, 0.5, 2, 3, 2]

    logging.info("Initializing initial population")

    # Read the CSV file into a DataFrame
    df = pd.read_csv("surrogate_data_energy.csv")
    

    # Apply the conversion function to each row, excluding the last column
    df.iloc[:, :-11] = df.iloc[:, :-11].apply(lambda row: hyperparams_convert_back_codet5(row.tolist()), axis=1,
                                            result_type='expand')
    features = df.iloc[:, :-11].values
    accs = df['Rouge Score'].tolist()
    gpu_energy = df['evaluation_median_gpu_energy'].tolist()
    cpu_energy = df['evaluation_median_cpu_energy'].tolist()

    surrogate_model = SurrogateModel()
    surrogate_model.fit([features, accs, gpu_energy, cpu_energy])

    problem = ModelCompressionProblem(lb, ub, surrogate_model)

    # set the seed for replicability
    seed = 2
    set_seed(seed)

    algorithm = AGEMOEA(pop_size=50,
                      sampling=LatinHypercubeSampler(),
                      selection=TournamentSelection(func_comp=binary_tournament),
                      crossover=SBX(eta=15, prob=0.9),
                      mutation=PolynomialMutation(eta=30),
                      repair=MyRepairCodeT5(),
                      )

    res = minimize(problem,
                   algorithm,
                   get_termination("n_gen", 100),
                   seed=seed,
                   verbose=True)

    logging.info("Time taken: {}".format(time.time() - start_time))
    logging.info("Number of solutions in the archive: {}".format(len(res.F)))
    logging.info("Saving the archive to the file")

    fieldnames = [
        "Tokenizer", "Vocab Size", "Num Hidden Layers", "Hidden Size", "Hidden Act", "Hidden Dropout Prob",
        "Intermediate Size", "Num Attention Heads", "Attention Probs Dropout Prob", "Max Sequence Length",
        "Position Embedding Type", "Learning Rate", "Batch Size", "Size", "Accuracy", "FLOPS", "Flips"
    ]

    # results_file = "new_pareto.csv"
    # with open(results_file, 'a', newline='') as file:
    #     writer = csv.DictWriter(file, fieldnames=fieldnames)
    #     # Write the header only if the file is empty
    #     if file.tell() == 0:
    #         writer.writeheader()

    #     for index in range(0, len(res.F)):
    #         x = res.X[index, :]
    #         objs = problem.evaluate(res.X[index, :])[0]
    #         converted_sol = convert_chromosomes([x])


    #     # Create a dictionary from row data
    #         row_data = {
    #             "Tokenizer": converted_sol[0][0],
    #             "Vocab Size": converted_sol[0][1],
    #             "Num Hidden Layers": converted_sol[0][2],
    #             "Hidden Size": converted_sol[0][3],
    #             "Hidden Act": converted_sol[0][4],
    #             "Hidden Dropout Prob": converted_sol[0][5],
    #             "Intermediate Size": converted_sol[0][6],
    #             "Num Attention Heads": converted_sol[0][7],
    #             "Attention Probs Dropout Prob": converted_sol[0][8],
    #             "Max Sequence Length": converted_sol[0][9],
    #             "Position Embedding Type": converted_sol[0][10],
    #             "Learning Rate": converted_sol[0][11],
    #             "Batch Size": converted_sol[0][12],
    #             "Size": objs[0],  # Assuming objs[0] is the Size
    #             "Accuracy": objs[1],  # Assuming accs contains accuracy values
    #             "FLOPS": objs[3],  # Assuming objs[3] is the FLOPS
    #             "Flips": res.F[index, 2]  # Assuming prediction_flips contains the flips value
    #         }

    #     # Write the row data to the CSV file
    #         writer.writerow(row_data)

    logging.info("Pareto front \n : {}".format(res.F))
    
    # # Extract the first objective
    # first_objective = res.F[:, 0]

    # # Calculate the absolute difference from 3
    # difference_from_3 = np.abs(first_objective - 3)

    # # Find the index of the minimum difference
    # closest_index = np.argmin(difference_from_3)

    # # Retrieve the solution with the first objective closest to 3
    # closest_solution = res.X[closest_index, :]
    # print("solution = ", closest_solution)

    # Retrieve solution with highest second objective
    max_accuracy_index = np.argmin(res.F[:, 1])
    solution = res.X[max_accuracy_index, :]
    print("solution = ", solution)

    objs = problem.evaluate(res.X[max_accuracy_index, :])
    logging.info("Objs : {}".format(objs))
    converted_sol = convert_chromosomes([solution])
    accs, prediction_flips = distill_codet5(converted_sol, eval=False, surrogate=False, seed=seed)
    accs, prediction_flips = distill_codet5(converted_sol, eval=True, surrogate=False, seed=seed)
    logging.info("Prediction flips : {}".format(res.F[closest_index, 3]))

    # results_file = "results_morph.csv"
    # with open(results_file, 'a', newline='') as file:
    #     writer = csv.DictWriter(file, fieldnames=fieldnames)
    #     # Write the header only if the file is empty
    #     if file.tell() == 0:
    #         writer.writeheader()

    #     # Create a dictionary from row data
    #     row_data = {
    #         "Tokenizer": converted_sol[0][0],
    #         "Vocab Size": converted_sol[0][1],
    #         "Num Hidden Layers": converted_sol[0][2],
    #         "Hidden Size": converted_sol[0][3],
    #         "Hidden Act": converted_sol[0][4],
    #         "Hidden Dropout Prob": converted_sol[0][5],
    #         "Intermediate Size": converted_sol[0][6],
    #         "Num Attention Heads": converted_sol[0][7],
    #         "Attention Probs Dropout Prob": converted_sol[0][8],
    #         "Max Sequence Length": converted_sol[0][9],
    #         "Position Embedding Type": converted_sol[0][10],
    #         "Learning Rate": converted_sol[0][11],
    #         "Batch Size": converted_sol[0][12],
    #         "Size": objs[0],  # Assuming objs[0] is the Size
    #         "Accuracy": accs[0],  # Assuming accs contains accuracy values
    #         "FLOPS": objs[3],  # Assuming objs[3] is the FLOPS
    #         "Flips": res.F[closest_index, 2]  # Assuming prediction_flips contains the flips value
    #     }

    #     # Write the row data to the CSV file
    #     writer.writerow(row_data)

    # # Open the file in append mode and write the data
    # # Define the results file
    # results_file = "mo_pareto_fronts.csv"

    # # Check if the file exists
    # file_exists = os.path.isfile(results_file)

    # # Write the header if the file does not exist
    # with open(results_file, "a", newline='') as f:
    #     writer = csv.writer(f)
    #     # Write the header if the file does not exist
    #     if not file_exists:
    #         writer.writerow([
    #             "Seed", "Algorithm", "Model Size", "Accuracy", "FLOPs", "Prediction Flips"
    #         ])

    #     for index in range(0, len(res.F)):
    #         row_data = [
    #             seed,
    #             "AGEMOEA",
    #             res.F[index, 0],
    #             -res.F[index, 1],
    #             res.F[index, 2],
    #             res.F[index, 3]
    #         ]
    #         writer.writerow(row_data)

if __name__ == "__main__":
    main()