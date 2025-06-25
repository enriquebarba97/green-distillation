import argparse
import csv
from mo_distill_utils import distill, hyperparams_convert, distill_codet5, hyperparams_convert_codet5, hyperparams_convert_back_codet5
from flops import TransformerHparams
from many_objective import convert_chromosomes, MyRepair, ModelCompressionProblem, \
    LatinHypercubeSampler

def main_roberta():
    # Define the lower and upper bounds
    lb = [1, 1000, 1, 16, 1, 0.2, 32, 1, 0.2, 256, 1, 1, 1]
    ub = [4, 46000, 12, 256, 4, 0.5, 3072, 12, 0.5, 512, 3, 3, 2]

    # Number of points to generate
    n_points = 20

    problem = ModelCompressionProblem(lb, ub, None)
    sampler = LatinHypercubeSampler()

    surrogate_data = convert_chromosomes(sampler._do(problem, 20))

    # trains the models
    accs, prediction_flips = distill(surrogate_data, eval=False, surrogate=True)

    print("Create surrogate models")

    with open("surrogate_data_metamorphic.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Tokenizer", "Vocab Size", "Num Hidden Layers", "Hidden Size", "Hidden Act", "Hidden Dropout Prob",
             "Intermediate Size", "Num Attention Heads", "Attention Probs Dropout Prob", "Max Sequence Length",
             "Position Embedding Type", "Learning Rate", "Batch Size", "Model Size","Accuracy", "Prediction Flips"])
        for i in range(0, len(accs)):
            model = TransformerHparams(surrogate_data[i][3], surrogate_data[i][2], surrogate_data[i][9],
                                       surrogate_data[i][1], surrogate_data[i][6], surrogate_data[i][7])
            size = abs(model.get_params() * 4 / 1e6)

            row_data = hyperparams_convert(surrogate_data[i])
            row_data += [size]
            row_data += [accs[i]]
            row_data += [prediction_flips[i]]
            writer.writerow(row_data)

def main_codet5(start_from=0, end_at=80, single=False):
    # Define the lower and upper bounds
    # We use default values as upper bounds since they are smaller than 220m
    lb = [1, 1, 1, 16, 1, 1, 16, 4, 32, 0.1, 1, 1, 1]
    ub = [6, 4, 6, 512, 8, 64, 2048, 32, 128, 0.5, 2, 3, 2]
#    ub = [12, 4, 12, 768, 12, 64, 3072, 32, 128, 0.3, 2, 3, 2]


    if start_from == 0:
    # Number of points to generate
        n_points = 80

        with open("surrogate_data.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["Num Hidden Layers", "Hidden Activation", "Number Decoder Layers", "Hidden Size", 
                "Num Attention Heads", "Projection Size", "Intermediate Size", "Relative Attention Buckets",
                "Relative Attention Max Distance", "Dropout Rate", "Feed Forward Projection",
                "Learning Rate", "Batch Size", "Model Size", "Rouge Score"])

        problem = ModelCompressionProblem(lb, ub, None)
        sampler = LatinHypercubeSampler()

        surrogate_data = convert_chromosomes(sampler._do(problem, n_points))

        with open("surrogate_data_sampling.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["Num Hidden Layers", "Hidden Activation", "Number Decoder Layers", "Hidden Size", 
                "Num Attention Heads", "Projection Size", "Intermediate Size", "Relative Attention Buckets",
                "Relative Attention Max Distance", "Dropout Rate", "Feed Forward Projection",
                "Learning Rate", "Batch Size"])
            for i in range(0, len(surrogate_data)):
                row_data = hyperparams_convert_codet5(surrogate_data[i])
                writer.writerow(row_data)
    else:
        with open("surrogate_data_sampling.csv", "r") as f:
            reader = csv.reader(f)
            surrogate_data = list(reader)[1:]
        surrogate_data = [hyperparams_convert_back_codet5(row) for row in surrogate_data]

    if single:
        
        rouges, sizes = distill_codet5([surrogate_data[start_from]], eval=False, surrogate=True, weights_file=f"model-{start_from}.bin")

        with open("surrogate_data.csv", "a") as f:
                writer = csv.writer(f)
                # model = TransformerHparams(surrogate_data[i][3], surrogate_data[i][2], surrogate_data[i][9],
                #                            surrogate_data[i][1], surrogate_data[i][6], surrogate_data[i][7])
                # size = abs(model.get_params() * 4 / 1e6)

                row_data = hyperparams_convert_codet5(surrogate_data[start_from])
                row_data += [sizes[0]]
                row_data += [rouges[0]]
                row_data += [start_from]
                writer.writerow(row_data)

    else:
        for i in range(start_from, end_at):

        # trains the models
            rouges, sizes = distill_codet5([surrogate_data[i]], eval=False, surrogate=True, weights_file=f"model-{i}.bin")

            with open("surrogate_data.csv", "a") as f:
                writer = csv.writer(f)
                # model = TransformerHparams(surrogate_data[i][3], surrogate_data[i][2], surrogate_data[i][9],
                #                            surrogate_data[i][1], surrogate_data[i][6], surrogate_data[i][7])
                # size = abs(model.get_params() * 4 / 1e6)

                row_data = hyperparams_convert_codet5(surrogate_data[i])
                row_data += [sizes[0]]
                row_data += [rouges[0]]
                row_data += [i]
                writer.writerow(row_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create surrogate models for CodeT5")
    parser.add_argument("--start-from", type=int, default=0, help="Start from the given index in surrogate data sampling")
    parser.add_argument("end-at", type=int, default=80, help="End at the given index in surrogate data sampling")
    parser.add_argument("--single", action='store_true', help="Run single model creation instead of batch")

    args = parser.parse_args()
    main_codet5(args.start_from, args.single)