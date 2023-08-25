from datasets import load_dataset
import csv
import json
import os
import tiktoken


def get_batch_size(model_type, gpu_memory, input_length, output_length):
    # 2048 is the maximum input length for all models
    batch_sizes_same = {
        'mgpt': {
            24: {512: 0, 1024: 0, 1536: 0, 2048: 0},
            48: {512: 1, 1024: 1, 1536: 1, 2048: 1},  # never tested
            80: {512: 8, 1024: 6, 1536: 2, 2048: 2},  # tested
        },
        'mt5-small': {
            24: {512: 4, 1024: 2, 2048: 1},  # tested with 512, 1024 and 2048 seq length
            48: {512: 8, 1024: 4, 2048: 2},  # never tested
            80: {512: 16, 1024: 12, 2048: 4},  # tested 1024 and 2048
        },
        'mt5-base': {
            24: {512: 2, 1024: 1, 2048: 1},  # only tested with 512 seq length
            48: {512: 4, 1024: 2, 2048: 1},  # never tested
            80: {512: 8, 1024: 6, 2048: 2},  # tested 1024 and 2048
        },
        'mt5-large': {
            24: {512: 1, 1024: 0, 2048: 0},  # tested
            48: {512: 1, 1024: 1, 2048: 1},  # never tested
            80: {512: 4, 1024: 2, 2048: 1},  # tested 1024 and 2048
        },
    }

    # for summarization task
    batch_sizes_256_output = {
        'mt5-small': {
            24: {512: 1, 1024: 1, 2048: 1, 3072: 1, 4096: 1},  # never tested
            48: {512: 1, 1024: 1, 2048: 2, 3072: 1, 4096: 1},  # never tested
            80: {512: 16, 1024: 16, 2048: 16, 3072: 12, 4096: 8},  # tested
        },
        'mt5-base': {
            24: {512: 1, 1024: 1, 2048: 1, 3072: 1, 4096: 1},  # never tested
            48: {512: 1, 1024: 1, 2048: 1, 3072: 1, 4096: 0},  # never tested
            80: {512: 16, 1024: 16, 2048: 8, 3072: 4, 4096: 2},  # tested
        },
        'mt5-large': {
            24: {512: 1, 1024: 1, 2048: 0, 3072: 0, 4096: 0},  # never tested
            48: {512: 1, 1024: 1, 2048: 1, 3072: 1, 4096: 1},  # never tested
            80: {512: 14, 1024: 6, 2048: 2, 3072: 1, 4096: 0},  # tested
        },
        'mt5-xl': { # we cannot even run 1024, so disregard this
            24: {512: 1, 1024: 1, 2048: 0, 3072: 0, 4096: 0},  # never tested
            48: {512: 1, 1024: 1, 2048: 1, 3072: 1, 4096: 1},  # never tested
            80: {512: 1, 1024: 0, 2048: 0, 3072: 0, 4096: 0},  # tested
        },
    }

    # for court view generation task
    batch_sizes_512_output = {
        'mt5-small': {
            24: {512: 1, 1024: 1, 2048: 1, 3072: 1, 4096: 1},  # never tested
            48: {512: 1, 1024: 1, 2048: 2, 3072: 1, 4096: 1},  # never tested
            80: {512: 16, 1024: 16, 2048: 14, 3072: 10, 4096: 6},  # tested
        },
        'mt5-base': {
            24: {512: 1, 1024: 1, 2048: 1, 3072: 1, 4096: 1},  # never tested
            48: {512: 1, 1024: 1, 2048: 1, 3072: 1, 4096: 0},  # never tested
            80: {512: 16, 1024: 12, 2048: 6, 3072: 4, 4096: 2},  # tested
        },
        'mt5-large': {
            24: {512: 1, 1024: 1, 2048: 0, 3072: 0, 4096: 0},  # never tested
            48: {512: 1, 1024: 1, 2048: 1, 3072: 1, 4096: 1},  # never tested
            80: {512: 8, 1024: 4, 2048: 2, 3072: 1, 4096: 0},  # tested
        },
    }

    try:
        if input_length == output_length:
            batch_sizes = batch_sizes_same
        elif output_length == 256:
            batch_sizes = batch_sizes_256_output
        elif output_length == 512:
            batch_sizes = batch_sizes_512_output
        else:
            batch_sizes = batch_sizes_same
            #raise ValueError(f"Output length {output_length} not supported")
        batch_size = batch_sizes[model_type][gpu_memory][input_length]
    except KeyError:
        print(f"Batch size not found for "
              f"model type: {model_type}, "
              f"input length: {input_length}, "
              f"gpu memory: {gpu_memory}")
        raise KeyError

    return batch_size

def average_rouge_scores(rouge_scores_list):
    avg_scores = {
        'rouge-1': {'r': 0, 'p': 0, 'f': 0},
        'rouge-2': {'r': 0, 'p': 0, 'f': 0},
        'rouge-l': {'r': 0, 'p': 0, 'f': 0}
    }

    num_scores = len(rouge_scores_list)

    for scores in rouge_scores_list:
        for rouge_type in avg_scores:
            for metric in avg_scores[rouge_type]:
                avg_scores[rouge_type][metric] += scores[rouge_type][metric]

    for rouge_type in avg_scores:
        for metric in avg_scores[rouge_type]:
            avg_scores[rouge_type][metric] /= num_scores

    return avg_scores

def average_bert_score(bert_scores):
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    count = len(bert_scores)

    for bert_score in bert_scores:
        total_precision += sum(bert_score['precision']) / len(bert_score['precision'])
        total_recall += sum(bert_score['recall']) / len(bert_score['recall'])
        total_f1 += sum(bert_score['f1']) / len(bert_score['f1'])

    return {
        'precision': total_precision / count,
        'recall': total_recall / count,
        'f1': total_f1 / count
    }


def export_output(data, output_dir, task_name):
    # Export to CSV
    # if output_dir doesn't exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(f"{output_dir}/{task_name}_{output_dir.split('/')[-1]}.csv", mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=data[0].keys())
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    print(f"Output saved to {output_dir}/output_{output_dir.split('/')[-1]}.csv")

    # Export to JSONL
    with open(f"{output_dir}/{task_name}_{output_dir.split('/')[-1]}.jsonl", mode='w') as jsonl_file:
        for row in data:
            json.dump(row, jsonl_file)
            jsonl_file.write('\n')
    print(f"Output saved to {output_dir}/output_{output_dir.split('/')[-1]}.jsonl")



def get_datasets(logger, sum="False", origin="False"):
    # Load dataset
    if sum == "True":
        logger.info("Loading summarization dataset")
        dataset = load_dataset("rcds/swiss_ruling_summarization")
    else:
        if origin == "True":
            logger.info("Loading origin dataset")
            dataset = load_dataset("rcds/swiss_court_view_generation", "origin")
        else:
            logger.info("Loading cvg dataset")
            dataset = load_dataset("rcds/swiss_court_view_generation", "main")
    return dataset['train'], dataset['validation'], dataset['test']

def get_val_dataset(logger, sum="False", origin="False"):
    # Load dataset
    if sum == "True":
        logger.info("Loading summarization dataset (validation)")
        dataset = load_dataset("rcds/swiss_ruling_summarization", split="validation")
    else:
        if origin == "True":
            logger.info("Loading origin dataset (validation)")
            dataset = load_dataset("rcds/swiss_court_view_generation", "origin", split="validation")
        else:
            logger.info("Loading full dataset (validation)")
            dataset = load_dataset("rcds/swiss_court_view_generation", "main", split="validation")
    return dataset

def truncate_text(text, max_tokens, tokenizer):
    enc = tiktoken.get_encoding("cl100k_base")
    enc = tiktoken.encoding_for_model(tokenizer)
    text_token_ids = enc.encode(text)
    if len(text_token_ids) > max_tokens:
        text_token_ids = text_token_ids[:max_tokens]
    return enc.decode(text_token_ids)

def number_of_tokens(text, tokenizer):
    enc = tiktoken.get_encoding("cl100k_base")
    enc = tiktoken.encoding_for_model(tokenizer)
    return len(enc.encode(text))

def tokenize(text, tokenizer):
    enc = tiktoken.get_encoding("cl100k_base")
    enc = tiktoken.encoding_for_model(tokenizer)
    print("loading tokenizer for model: ", tokenizer)
    text_token_ids = enc.encode(text)
    # we want a list text_tokens with applying enc.decode([x]) for each x in text_token_ids
    text_tokens = [enc.decode([x]) for x in text_token_ids]
    return text_tokens ## list of tokens