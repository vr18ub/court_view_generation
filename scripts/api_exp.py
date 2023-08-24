from datasets import load_dataset
import os
import openai
import tiktoken

from scripts.util import get_batch_size
from scripts.util import average_bert_score, average_rouge_scores, export_output, get_val_dataset

import nltk

nltk.download('wordnet')

import os
import torch
from nltk.translate import meteor_score
import numpy as np
import argparse
import wandb
import json
import csv

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from rouge import Rouge
from evaluate import load

import logging

# implement timer
import time
import datetime

### Initialization
def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

def log_test_scores(meteor_score_avg, rouge_score_avg, bleu_score_avg, bert_score_avg):
    wandb.log({
        "METEOR_score/test": meteor_score_avg,
        "ROUGE_score/test/rouge-1/r": rouge_score_avg['rouge-1']['r'],
        "ROUGE_score/test/rouge-1/p": rouge_score_avg['rouge-1']['p'],
        "ROUGE_score/test/rouge-1/f": rouge_score_avg['rouge-1']['f'],
        "ROUGE_score/test/rouge-2/r": rouge_score_avg['rouge-2']['r'],
        "ROUGE_score/test/rouge-2/p": rouge_score_avg['rouge-2']['p'],
        "ROUGE_score/test/rouge-2/f": rouge_score_avg['rouge-2']['f'],
        "ROUGE_score/test/rouge-l/r": rouge_score_avg['rouge-l']['r'],
        "ROUGE_score/test/rouge-l/p": rouge_score_avg['rouge-l']['p'],
        "ROUGE_score/test/rouge-l/f": rouge_score_avg['rouge-l']['f'],
        "BLEU_score/test": bleu_score_avg,
        "BERT_score/test/precision": bert_score_avg['precision'],
        "BERT_score/test/recall": bert_score_avg['recall'],
        "BERT_score/test/f1": bert_score_avg['f1']
    })
    print()
    logger.info(f"Average METEOR score: {meteor_score_avg:.4f}")
    logger.info(f"Average ROUGE score: {rouge_score_avg}")
    logger.info(f"Average BLEU score: {bleu_score_avg:.4f}")
    logger.info(f"Average BERTScore: {bert_score_avg}")
    print()

def tokenize(text, tokenizer = "whitespace"):
    if tokenizer == "tiktoken":
        enc = tiktoken.encoding_for_model("gpt-4")
        return enc.encode(text) ## list of tokens
    elif tokenizer == "whitespace":
        return text.split(" ")

def compute_scores(completion_dataset, num_examples=100):
    scores = {'meteor': [], 'rouge': [], 'bleu': [], 'bert': []}
    rouge = Rouge()
    bertscore = load("bertscore")

    for idx, entry in enumerate(completion_dataset):
        """
        target_text_tokens: list of tokens
        predicted_text_tokens: list of tokens

        predicted_text: string (plain text)
        target_text: string (plain text)

        tokenized_target_text: string (tokens)
        tokenized_predicted_text: string (tokens)
        """
        target_text_tokens = tokenize(entry['target'])
        predicted_text_tokens = tokenize(entry['predicted'])

        predicted_text = entry['predicted']
        target_text = entry['target']

        tokenized_target_text = ' '.join(target_text_tokens)
        tokenized_predicted_text = ' '.join(predicted_text_tokens)

        # Calculate Meteor scores
        meteor = meteor_score.meteor_score([target_text_tokens], predicted_text_tokens)
        scores['meteor'].append(meteor)

        # Calculate Rouge scores
        rouge_scores = rouge.get_scores(predicted_text, target_text)[0]
        scores['rouge'].append(rouge_scores)

        # Calculate Bleu scores
        bleu = sentence_bleu([tokenized_target_text], tokenized_predicted_text, weights=(.25, .25, .25, .25))
        scores['bleu'].append(bleu)

        # Calculate BERTScore
        bert = bertscore.compute(predictions=[predicted_text], references=[target_text],
                                 model_type="bert-base-multilingual-cased", lang=['de', 'fr', 'it'])
        scores['bert'].append(bert)

        output_examples.append({
            'target': target_text,
            'predicted': predicted_text,
            'meteor': meteor,
            'bert-f1': bert['f1'][0],
            'bleu': bleu,
            'rouge-1_f1': rouge_scores['rouge-1']['f'],
            'rouge-2_f1': rouge_scores['rouge-2']['f'],
            'rouge-l_f1': rouge_scores['rouge-l']['f'],
            'bert_full': bert,
            'rouge_full': rouge_scores,
            })

        # Print examples
        if idx < num_examples:
            print("\n", flush=True)
            print("#" * 180, flush=True)
            logger.info(f"Example {idx + 1} of {len(completion_dataset)}")
            logger.info(f"Output: {predicted_text}")
            logger.info("-" * 100)
            logger.info(f"Label: {target_text}")
            logger.info("-" * 100)
            logger.info(f"METEOR Score: {meteor:.4f}")
            logger.info(f"ROUGE Score: {rouge_scores}")
            logger.info(f"BLEU Score: {bleu:.4f}")
            logger.info(f"BERTScore: {bert}")
            print("#" * 180, flush=True)
            print("\n", flush=True)

    return np.mean(scores['meteor']), average_rouge_scores(scores['rouge']), np.mean(
        scores['bleu']), average_bert_score(scores['bert'])


logger = setup_logger()

parser = argparse.ArgumentParser()
parser.add_argument("--finetune", help="Want to finetune model?")
parser.add_argument("--model", help="Model name for finetune / evaluation (depends on finetune flag")
parser.add_argument("--train_size", help="Size of training set", type=int)
parser.add_argument("--eval_size", help="Size of evaluation set", type=int)
parser.add_argument("--test_size", help="Size of test set", type=int)
parser.add_argument("--input_length", help="Input sequence length for training, evaluation and generation", type=int)
parser.add_argument("--output_length", help="Output sequence length for training, evaluation and generation", type=int)
parser.add_argument("--epochs", help="Number of training epochs", type=int)
parser.add_argument("--total_batch_size", help="The total batch size to use", type=int)
parser.add_argument("--gm", help="GPU memory size for batch size", type=int)
parser.add_argument("--origin", help="Use dataset with origin cases")
parser.add_argument("--sum", help="Loads summarization dataset if True")
args = parser.parse_args()

model_name = args.model

if args.origin == "True" and args.sum == "True":
    raise ValueError("Cannot use both origin and sum flags (as True)")

# print all args
logger.info(args)

eval_dataset = get_val_dataset(logger, args.sum, args.origin)

# Update args values with the full lengths of the dataset splits if the args values are -1
if args.eval_size == -1:
    args.eval_size = len(eval_dataset)

# Select subsets of the dataset based on the updated args values
seed = 42
eval_dataset = eval_dataset.shuffle(seed).select(range(args.eval_size))

project_name = "summarization" if args.sum == "True" else "court view generation"
logger.info("Project name: " + project_name)
os.environ["WANDB_PROJECT"] = project_name
os.environ["WANDB_RUN_GROUP"] = f"{model_name}, {len(eval_dataset)}"

# add train size, seq length to output dir
output_dir = f"output/{args.model.split('/')[-1]}_inlen={args.input_length}_outlen={args.output_length}_origin={args.origin}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}"
# set wandb run name
wandb.init(name=output_dir.split('/')[-1]) # means
# log output dir to wandb
wandb.log({"output_dir": output_dir})

logger.info("Model name:" + model_name + " finetune: " + " output_dir: " + output_dir)
logger.info("Eval dataset size: " + str(len(eval_dataset)) )

# log all args to wandb
wandb.config.update(args)

output_examples = []

# TODO: komplett anpassen an dataset format, für api completions
# TODO: trennen in eine funktion für generation und eine für evaluation

def prepare_dataset(dataset, input_len, output_len, sum, origin):
    """
       if args.origin is "True": it takes origin_facts and origin_considerations as input
       else: it takes facts and considerations as input except specific input_col_name and target_col_name are given as arguments
       """
    if sum == "True":
        input_col_name = "text"
        target_col_name = "regeste"
    if sum == "False":
        input_col_name = "facts"
        target_col_name = "considerations"
    if origin == "True":
        raise NotImplementedError("Origin not implemented yet")

    data_list = [{"input": i, "target": t, "lang": lang} for i, t, lang in zip(dataset[input_col_name], dataset[target_col_name], dataset["language"])]

    # truncate input and target to input_len and output_len words:
    # but join input and target first, so that we can truncate them together
    for i in range(len(data_list)):
        data_list[i]["input"] = " ".join(data_list[i]["input"].split()[:int(input_len/3)])
        data_list[i]["target"] = " ".join(data_list[i]["target"].split()[:output_len])


    # we want to return a list of dicts with the keys "input" and "target"
    return data_list

# prepare dataset for generation
eval_data = prepare_dataset(eval_dataset, args.input_length, args.output_length, args.sum, args.origin)

def create_instruction(input, task="court_view_generation", lang='en'):
    """
    Creates the instruction for the API completion
    """
    if task == "court_view_generation":
        if lang == 'en':
            instruction = f"'Given the following facts:'\n'{input}'\n'Write the considerations of the court.'\n'Considerations:'\n'"
        elif lang == 'de':
            instruction = f"'Gegeben sind folgende Sachverhalte:'\n'{input}'\n'Schreiben Sie die Erwägungen des Gerichts.'\n'Erwägungen:'\n'"
        elif lang == 'fr':
            instruction = f"'Étant donné les faits suivants:'\n'{input}'\n'Écrivez les considérations du tribunal.'\n'Considérations:'\n'"
        elif lang == 'it':
            instruction = f"'Dati i seguenti fatti:'\n'{input}'\n'Scrivere le considerazioni del tribunale.'\n'Considerazioni:'\n'"
    else:
        raise NotImplementedError("Task not implemented yet")
    return instruction

def generate_completions(dataset, input_length, output_length, model='gpt-3.5-turbo'):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    dataset_with_predicted = []

    for entry in dataset[:20]:
        instruct = create_instruction(entry["input"], lang=entry["lang"])
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": instruct}
                ],
            max_tokens=output_length,
            )
        print(completion)
        predicted = completion.choices[0].message["content"]
        entry_with_predicted = {"input": entry["input"], "target": entry["target"], "predicted": predicted}
        dataset_with_predicted.append(entry_with_predicted)

    return dataset_with_predicted


# generate output examples using API completions
completion_dataset = generate_completions(eval_data, args.input_length, args.output_length)


# Evaluate model on test dataset
meteor_score_avg, rouge_score_avg, bleu_score_avg, bert_score_avg = compute_scores(completion_dataset)

# Print and log scores to wandb
log_test_scores(meteor_score_avg, rouge_score_avg, bleu_score_avg, bert_score_avg)

try:
    # save output examples to file
    export_output(output_examples, output_dir)
except Exception as e:
    logger.info("Error exporting output examples: " + str(e))
