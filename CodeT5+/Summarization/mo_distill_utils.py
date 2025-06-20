import os
import time
import torch
import logging
import warnings
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from models import Model, distill_loss, distillation_loss_new, distillation_loss_generation, DistillationLoss
from utils import set_seed, DistilledDataset, OnlineDistilledDataset
from sklearn.metrics import recall_score, precision_score, f1_score
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, ConcatDataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, RobertaConfig, RobertaModel, T5Config, T5ForConditionalGeneration, T5Model, AutoModelForSeq2SeqLM
from torch.optim import AdamW
from rouge_score import rouge_scorer

warnings.filterwarnings("ignore")
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def train_codet5(model, teacher_model, tokenizer, train_dataloader, eval_dataloader, epochs, learning_rate, device, surrogate=False, weights_file="model.bin"):
    num_steps = len(train_dataloader) * epochs
    no_decay = ["bias", "LayerNorm.weight"]
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"{total_params:,} total parameters.")
    logger.info(f"{total_params * 4 / 1e6} MB model size")

    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_steps * 0.1,
                                                num_training_steps=num_steps)

    dev_best_acc = 0
    dev_best_flips = float("inf")  # Initialize to a large number
    best_pred = []

    criterion = DistillationLoss()

    teacher_model.eval()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        tr_num = 0
        logger.info(f"Epoch [{epoch + 1}/{epochs}]")
        bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")

        for batch in bar:
            texts = batch[0].to(device)
            attention = batch[1].to(device)
            labels = batch[2].to(device)

            with torch.no_grad():
                teacher_output = teacher_model(input_ids=texts, attention_mask=attention, labels=labels)
                teacher_logits = teacher_output.logits
                # softmax = F.softmax(teacher_logits, dim=-1)
                # teacher_preds = torch.argmax(softmax, dim=-1)
            
            # Forward pass
            student_outputs = model(input_ids=texts, attention_mask=attention, labels=labels)
            student_logits = student_outputs.logits

            #loss = distillation_loss_generation(student_logits, teacher_logits, labels)
            loss = criterion(student_logits, teacher_logits, labels)

            optimizer.zero_grad()
            loss.backward()

            # Optimizer and scheduler step
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            tr_num += 1

        # Validation after training in each epoch
        dev_results = evaluate(model, device, eval_dataloader, tokenizer)

        # Validation metrics
        dev_acc = dev_results["rouge_l"]
        acc_tolerance = 0.005  # 0.5% tolerance for accuracy

        # Check if this is the best model so far
        if dev_acc > dev_best_acc:
            dev_best_acc = dev_acc

            save_path = os.path.join("../checkpoints", "surrogate", weights_file) if surrogate else os.path.join("../checkpoints", "final", weights_file)

            output_dir = os.path.join("../checkpoints", "Morph")
            os.makedirs(output_dir, exist_ok=True)
            model_path = os.path.join(output_dir, weights_file)

            if os.path.exists(model_path):
                os.remove(model_path)

            torch.save(model.state_dict(), model_path)
            logger.info("New best model found and saved.")

        logger.info("Epoch [{}]: Train Loss: {:.4f}, Val ROUGE-L: {:.4f}".format(
            epoch + 1,
            train_loss / tr_num,
            dev_results["rouge_l"],
        ))

        logger.info(f"Epoch {epoch + 1}, loss: {train_loss / tr_num:.4f}, ")

        # Clear memory
        del texts, labels #, soft_knowledge
        torch.cuda.empty_cache()

    return dev_best_acc, best_pred

def train(model, train_dataloader, eval_dataloader, meta_dataloader, epochs, learning_rate, device, surrogate=False, weights_file="model.bin"):
    num_steps = len(train_dataloader) * epochs
    no_decay = ["bias", "LayerNorm.weight"]
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"{total_params:,} total parameters.")
    logger.info(f"{total_params * 4 / 1e6} MB model size")

    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_steps * 0.1,
                                                num_training_steps=num_steps)

    dev_best_acc = 0
    dev_best_flips = float("inf")  # Initialize to a large number
    best_pred = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        tr_num = 0
        logger.info(f"Epoch [{epoch + 1}/{epochs}]")
        bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")

        for batch in bar:
            texts = batch[0].to(device)
            labels = batch[1].to(device)
            soft_knowledge = batch[3].to(device)

            # Forward pass
            preds = model(texts)
            print(preds)
            loss = distillation_loss_new(preds, soft_knowledge, labels)
            loss.backward()

            # Optimizer and scheduler step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            tr_num += 1

        # Validation after training in each epoch
        dev_results, predictions = evaluate(model, device, eval_dataloader)
        meta_results, meta_predictions = evaluate(model, device, meta_dataloader)

        # Calculate flips
        prediction_flips = np.sum(predictions != meta_predictions)

        # Validation metrics
        dev_acc = (dev_results["eval_acc"] + meta_results["eval_acc"])/2.0 #dev_acc = dev_results["eval_acc"]
        acc_tolerance = 0.005  # 0.5% tolerance for accuracy

        # Check if this is the best model so far
        if dev_acc > dev_best_acc or (
                dev_best_acc - dev_acc <= acc_tolerance and prediction_flips < dev_best_flips):
            dev_best_acc = dev_acc
            dev_best_flips = prediction_flips
            best_pred = predictions

            if not surrogate:
                output_dir = os.path.join("../checkpoints", "Morph")
                os.makedirs(output_dir, exist_ok=True)
                model_path = os.path.join(output_dir, weights_file)

                if os.path.exists(model_path):
                    os.remove(model_path)

                torch.save(model.state_dict(), model_path)
                logger.info("New best model found and saved.")

        logger.info("Epoch [{}]: Train Loss: {:.4f}, Val Acc: {:.4f}, Val Precision: {:.4f}, Val Recall: {:.4f}, "
                    "Val F1: {:.4f}, Prediction Flips: {}".format(
            epoch + 1,
            train_loss / tr_num,
            dev_results["eval_acc"],
            dev_results["eval_precision"],
            dev_results["eval_recall"],
            dev_results["eval_f1"],
            prediction_flips
        ))

        # Clear memory
        del dev_results, predictions, texts, labels, soft_knowledge
        torch.cuda.empty_cache()

    return dev_best_acc, best_pred

def evaluate(model, device, eval_dataloader, tokenizer):
    model.eval()
    predict_all = []
    labels_all = []
    target_all = []
    time_count = []

    with torch.no_grad():
        bar = tqdm(eval_dataloader, total=len(eval_dataloader))
        bar.set_description("Evaluation")
        for batch in bar:
            texts = batch[0].to(device)
            label = batch[1].to(device)
            target = list(batch[4])
            time_start = time.time()
            pred = model.generate(texts)
            time_end = time.time()
            pred = tokenizer.batch_decode(pred, skip_special_tokens=True)
            
            
            predict_all += pred
            target_all += target
            labels_all.append(label.cpu().numpy())
            time_count.append(time_end - time_start)

    latency = np.mean(time_count)
    logger.info("Average Inference Time pre Batch: {}".format(latency))
    # Compute ROUGE-L scores
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    rouge_scores = []
    for pred, target in zip(predict_all, target_all):
        score = scorer.score(target, pred)
        rouge_scores.append(score['rougeL'].fmeasure)

    # Calculate average ROUGE-L score
    avg_rouge_l = np.mean(rouge_scores)
    print(f"Average ROUGE-L: {avg_rouge_l:.4f}")

    return {
        "rouge_l": avg_rouge_l,
        "inference_time": latency
    }


def evaluate_old(model, device, eval_dataloader):
    model.eval()
    predict_all = []
    labels_all = []
    time_count = []
    with torch.no_grad():
        bar = tqdm(eval_dataloader, total=len(eval_dataloader))
        bar.set_description("Evaluation")
        for batch in bar:
            texts = batch[0].to(device)
            label = batch[1].to(device)
            time_start = time.time()
            prob = model(texts)
            time_end = time.time()
            prob = F.softmax(prob)
            predict_all.append(prob.cpu().numpy())
            labels_all.append(label.cpu().numpy())
            time_count.append(time_end - time_start)

    latency = np.mean(time_count)
    logger.info("Average Inference Time pre Batch: {}".format(latency))
    predict_all = np.concatenate(predict_all, 0)
    labels_all = np.concatenate(labels_all, 0)

    preds = predict_all[:, 1] > 0.5
    recall = recall_score(labels_all, preds)
    precision = precision_score(labels_all, preds)
    f1 = f1_score(labels_all, preds)
    results = {
        "eval_acc": np.mean(labels_all == preds),
        "eval_precision": float(precision),
        "eval_recall": float(recall),
        "eval_f1": float(f1),
        "inference_time": latency
    }
    return results, preds

def distill(hyperparams_set, eval=False, surrogate=True, seed=1, weights_file="model.bin"):
    data_file = "data.jsonl"
    metamorphic_file = "metamorphic_data_new.jsonl"
    train_data_file = "../data/unlabel_train.txt"
    eval_data_file = "../data/valid_sampled.txt"
    test_data_file = "../data/test_sampled.txt"
    if surrogate:
        epochs = 5
    else:
        epochs = 20
    n_labels = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    set_seed(seed)

    prediction_flips = []
    dev_best_accs = []
    for hyperparams in hyperparams_set:
        tokenizer_type, vocab_size, num_hidden_layers, hidden_size, hidden_act, hidden_dropout_prob, intermediate_size, num_attention_heads, attention_probs_dropout_prob, max_sequence_length, position_embedding_type, learning_rate, batch_size = hyperparams_convert(
            hyperparams)

        config = RobertaConfig.from_pretrained("microsoft/codebert-base")
        config.num_labels = n_labels
        config.vocab_size = vocab_size
        config.num_hidden_layers = num_hidden_layers
        config.hidden_size = hidden_size
        config.hidden_act = hidden_act
        config.hidden_dropout_prob = hidden_dropout_prob
        config.intermediate_size = intermediate_size
        config.num_attention_heads = num_attention_heads
        config.attention_probs_dropout_prob = attention_probs_dropout_prob
        config.max_position_embeddings = max_sequence_length + 2
        config.position_embedding_type = position_embedding_type

        model = Model(RobertaModel(config=config), config)
        if not eval:
            train_dataset = DistilledDataset(tokenizer_type, vocab_size, train_data_file, max_sequence_length, logger,
                                             data_file)
            # make prediction on metamorphic data
            train_dataset2 = DistilledDataset(tokenizer_type, vocab_size, train_data_file, max_sequence_length, logger,
                                             metamorphic_file)

            # Combine datasets using the custom dataset class
            combined_train_dataset = ConcatDataset([train_dataset, train_dataset2])

            # Create a DataLoader for the combined dataset
            train_sampler = RandomSampler(combined_train_dataset)
            train_dataloader = DataLoader(combined_train_dataset, sampler=train_sampler, batch_size=batch_size,
                                          num_workers=8, pin_memory=True)

            eval_dataset = DistilledDataset(tokenizer_type, vocab_size, eval_data_file, max_sequence_length, logger,
                                            data_file)
            eval_sampler = SequentialSampler(eval_dataset)
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=batch_size * 2, num_workers=8,
                                         pin_memory=True)

            model.to(device)
            eval_dataset2 = DistilledDataset(tokenizer_type, vocab_size, eval_data_file, max_sequence_length, logger,
                                             metamorphic_file)
            eval_sampler2 = SequentialSampler(eval_dataset2)
            eval_dataloader2 = DataLoader(eval_dataset2, sampler=eval_sampler2, batch_size=batch_size * 2,
                                          num_workers=8,
                                          pin_memory=True)


            dev_best_acc, pred_original = train(model, train_dataloader, eval_dataloader, eval_dataloader2, epochs, learning_rate, device,
                                                surrogate, weights_file=weights_file)
            dev_best_accs.append(dev_best_acc)

            #eval_dataset2 = DistilledDataset(tokenizer_type, vocab_size, eval_data_file, max_sequence_length, logger,
            #                                  metamorphic_file)
            #eval_sampler2 = SequentialSampler(eval_dataset2)
            #eval_dataloader2 = DataLoader(eval_dataset2, sampler=eval_sampler2, batch_size=batch_size * 2,
            #                              num_workers=8,
            #                              pin_memory=True)
            meta_results, pred_metamorphic = evaluate(model, device, eval_dataloader2)
            prediction_flips.append(np.sum(pred_original != pred_metamorphic))

        else:
            model_dir = os.path.join("../checkpoints", "Morph", weights_file)
            model.load_state_dict(torch.load(model_dir, map_location=device))
            model.to(device)

            test_dataset = DistilledDataset(tokenizer_type, vocab_size, test_data_file, max_sequence_length, logger,
                                            data_file)
            test_sampler = SequentialSampler(test_dataset)
            test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size * 2, num_workers=8,
                                         pin_memory=True)

            test_results, prediction = evaluate(model, device, test_dataloader)

            meta_test_dataset = DistilledDataset(tokenizer_type, vocab_size, test_data_file, max_sequence_length,
                                                 logger,
                                                 metamorphic_file)
            meta_test_sampler = SequentialSampler(meta_test_dataset)
            meta_test_dataloader = DataLoader(meta_test_dataset, sampler=meta_test_sampler, batch_size=batch_size * 2,
                                              num_workers=8,
                                              pin_memory=True)
            meta_results, pred_metamorphic = evaluate(model, device, meta_test_dataloader)
            logger.info(
                "Test Acc: {0}, Test Precision: {1}, Test Recall: {2}, Test F1: {3}".format(test_results["eval_acc"],
                                                                                            test_results[
                                                                                                "eval_precision"],
                                                                                            test_results["eval_recall"],
                                                                                            test_results["eval_f1"],
                                                                                            test_results[
                                                                                                "inference_time"]))
            prediction_flips.append(np.sum(prediction != pred_metamorphic))
            dev_best_accs.append(test_results["eval_acc"])

    return dev_best_accs, prediction_flips

def distill_codet5(hyperparams_set, eval=False, surrogate=True, seed=1, weights_file="model.bin"):
    data_file = "data.jsonl"
    metamorphic_file = "metamorphic_data_new.jsonl"
    train_data_file = "../data/unlabel_train.txt"
    eval_data_file = "../data/valid_sampled.txt"
    test_data_file = "../data/test_sampled.txt"
    if surrogate:
        epochs = 10
    else:
        epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5p-220m")

    teacher_model = load_teacher_model("teacher_model/", device)

    # Teacher inference test
    # dataset = OnlineDistilledDataset(split="train")

    # Teacher inference test
    # with torch.no_grad():
    #     input_ids, attention_mask, labels, source, target = dataset[0]
        
    #     input_ids = input_ids.unsqueeze(0).to(device)
    #     attention_mask = attention_mask.unsqueeze(0).to(device)
    #     labels = labels.unsqueeze(0).to(device)

    #     decoder_input_ids = torch.tensor([[teacher_model.config.decoder_start_token_id]], device=input_ids.device)

    #     print("Labels:", labels)
    #     output = teacher_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    #     #print(output.logits)

    #     # Get tokens from the output logits
    #     logits = output.logits
    #     predicted_tokens = torch.argmax(logits, dim=-1)
    #     print("Predicted Tokens:", predicted_tokens)

    #     preds = teacher_model.generate(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, output_scores=True, return_dict_in_generate=True)
    #     print("Generated Tokens:", preds.sequences)
    #     print("Logits:", len(preds.scores))

    dev_best_rouges = []
    sizes = []
    for hyperparams in hyperparams_set:
        print(hyperparams)
        num_layers, hidden_act, num_decoder_layers, d_model, num_heads, d_kv, d_ff, relative_attention_num_buckets, relative_attention_max_distance, dropout_rate, feed_forward_proj, learning_rate, batch_size = hyperparams_convert_codet5(
            hyperparams)

        config = T5Config.from_pretrained("Salesforce/codet5p-220m")
        config.num_layers = num_layers
        config.num_decoder_layers = num_decoder_layers
        config.d_model = d_model
        config.hidden_act = hidden_act
        config.dropout_rate = dropout_rate
        config.d_kv = d_kv
        config.d_ff = d_ff
        config.num_heads = num_heads
        config.relative_attention_num_buckets = relative_attention_num_buckets
        config.relative_attention_max_distance = relative_attention_max_distance
        config.feed_forward_proj = feed_forward_proj
        
        model = T5ForConditionalGeneration(config=config)

        if not eval:
            train_dataset = OnlineDistilledDataset(split="train", tokenizer=tokenizer, n_samples=100000)

            # Create a DataLoader for the combined dataset
            train_sampler = RandomSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size,
                                          num_workers=8, pin_memory=True)

            eval_dataset = OnlineDistilledDataset(split="validation", tokenizer=tokenizer, n_samples=2000)
            eval_sampler = SequentialSampler(eval_dataset)
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=batch_size * 2, num_workers=8,
                                         pin_memory=True)

            model.to(device)

            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"{total_params:,} total parameters.")
            size = abs(total_params * 4 / 1e6)
            logger.info(f"{size} MB model size")
            sizes.append(size)

            dev_best_rouge, pred_original = train_codet5(model, teacher_model, tokenizer, train_dataloader, eval_dataloader, epochs, learning_rate, device,
                                                surrogate, weights_file=weights_file)
            dev_best_rouges.append(dev_best_rouge)
            del model, teacher_model, train_dataloader, eval_dataloader, train_dataset, eval_dataset
            #eval_dataset2 = DistilledDataset(tokenizer_type, vocab_size, eval_data_file, max_sequence_length, logger,
            #                                  metamorphic_file)
            #eval_sampler2 = SequentialSampler(eval_dataset2)
            #eval_dataloader2 = DataLoader(eval_dataset2, sampler=eval_sampler2, batch_size=batch_size * 2,
            #                              num_workers=8,
            #                              pin_memory=True)
#            meta_results, pred_metamorphic = evaluate(model, device, eval_dataloader2)
        else:
            model_dir = os.path.join("../checkpoints", "Morph", weights_file)
            model.load_state_dict(torch.load(model_dir, map_location=device))
            model.to(device)

            test_dataset = DistilledDataset(tokenizer_type, vocab_size, test_data_file, max_sequence_length, logger,
                                            data_file)
            test_sampler = SequentialSampler(test_dataset)
            test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size * 2, num_workers=8,
                                         pin_memory=True)

            test_results, prediction = evaluate(model, device, test_dataloader)

            meta_test_dataset = DistilledDataset(tokenizer_type, vocab_size, test_data_file, max_sequence_length,
                                                 logger,
                                                 metamorphic_file)
            meta_test_sampler = SequentialSampler(meta_test_dataset)
            meta_test_dataloader = DataLoader(meta_test_dataset, sampler=meta_test_sampler, batch_size=batch_size * 2,
                                              num_workers=8,
                                              pin_memory=True)
            meta_results, pred_metamorphic = evaluate(model, device, meta_test_dataloader)
            logger.info(
                "Test Acc: {0}, Test Precision: {1}, Test Recall: {2}, Test F1: {3}".format(test_results["eval_acc"],
                                                                                            test_results[
                                                                                                "eval_precision"],
                                                                                            test_results["eval_recall"],
                                                                                            test_results["eval_f1"],
                                                                                            test_results[
                                                                                                "inference_time"]))
            prediction_flips.append(np.sum(prediction != pred_metamorphic))
            dev_best_rouges.append(test_results["eval_acc"])

    return dev_best_rouges, sizes

def load_teacher_model(model_name, device):
    teacher = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    teacher.to(device)
    teacher.eval()
    return teacher

def hyperparams_convert(hyperparams):
    tokenizer_type = {1: "BPE", 2: "WordPiece", 3: "Unigram", 4: "Word"}
    hidden_act = {1: "gelu", 2: "relu", 3: "silu", 4: "gelu_new"}
    position_embedding_type = {1: "absolute", 2: "relative_key", 3: "relative_key_query"}
    learning_rate = {1: 1e-3, 2: 1e-4, 3: 5e-5}
    batch_size = {1: 8, 2: 16}

    return [
        tokenizer_type[hyperparams[0]],
        hyperparams[1],
        hyperparams[2],
        hyperparams[3],
        hidden_act[hyperparams[4]],
        hyperparams[5],
        hyperparams[6],
        hyperparams[7],
        hyperparams[8],
        hyperparams[9],
        position_embedding_type[hyperparams[10]],
        learning_rate[hyperparams[11]],
        batch_size[hyperparams[12]]
    ]

def hyperparams_convert_codet5(hyperparams):
    hidden_act = {1: "gelu", 2: "relu", 3: "silu", 4: "gelu_new"}
    feed_forward_proj = {1: "relu", 2: "gated-gelu"}
    learning_rate = {1: 1e-3, 2: 1e-4, 3: 5e-5}
    batch_size = {1: 8, 2: 16}

    return [
        hyperparams[0],
        hidden_act[hyperparams[1]],
        hyperparams[2],
        hyperparams[3],
        hyperparams[4],
        hyperparams[5],
        hyperparams[6],
        hyperparams[7],
        hyperparams[8],
        hyperparams[9],
        feed_forward_proj[hyperparams[10]],
        learning_rate[hyperparams[11]],
        batch_size[hyperparams[12]]
    ]

def hyperparams_convert_back_codet5(hyperparams):
    hidden_act = {"gelu": 1, "relu": 2, "silu": 3, "gelu_new": 4}
    feed_forward_proj = {"relu": 1, "gated-gelu": 2}
    learning_rate = {1e-3: 1, 1e-4: 2, 5e-5: 3}
    batch_size = {8: 1, 16: 2}

    return [
        hyperparams[0],
        hidden_act[hyperparams[1]],
        hyperparams[2],
        hyperparams[3],
        hyperparams[4],
        hyperparams[5],
        hyperparams[6],
        hyperparams[7],
        hyperparams[8],
        hyperparams[9],
        feed_forward_proj[hyperparams[10]],
        learning_rate[hyperparams[11]],
        batch_size[hyperparams[12]]
    ]


if __name__ == "__main__":
    print(hyperparams_convert([1, 27505, 1, 24, 3, 0.2, 1508, 2, 0.1, 512, 1, 2, 2]))
    distill([[1, 27505, 3, 36, 3, 0.3, 1508, 12, 0.2, 358, 1, 2, 2]], eval=True, surrogate=False)
