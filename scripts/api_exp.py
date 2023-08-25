import openai
import tiktoken

from scripts.util import average_bert_score, average_rouge_scores, export_output, get_val_dataset, truncate_text, number_of_tokens, tokenize

import os
from nltk.translate import meteor_score
import numpy as np
import argparse
import wandb

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
        target_text_tokens = tokenize(entry['target'], model_name)
        predicted_text_tokens = tokenize(entry['predicted'], model_name)

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
            'language': entry['lang'],
            'input': entry['input'],
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
        data_list[i]["input"] = truncate_text(data_list[i]["input"], input_len, model_name)
        data_list[i]["target"] = truncate_text(data_list[i]["target"], output_len, model_name)


    # we want to return a list of dicts with the keys "input" and "target"
    return data_list

def create_instruction(input, lang, task):
    """
    Creates the instruction for the API completion
    """
    if task == "cvg":
        if lang == 'de':
            instruction = f'Ziel: Generiere Erwägungen basierend auf dem gegebenen Sachverhalt eines Schweizer Gerichtsurteils.\nHintergrund: Ein Schweizer Gerichtsurteil besteht aus Rubrum, Sachverhalt, Erwägungen, Dispositiv (Urteilsformel) und Unterschrift. Die Erwägungen sind die rechtliche Würdigung des Geschehens durch das Gericht.\nAnweisung:\n-Sachverhalt Verstehen: Der gegebene Sachverhalt enthält bestrittene und unbestrittene Fakten, die Begehren der Parteien, das Beweisverfahren und die Prozessgeschichte.\n-Beginne mit Prozessvoraussetzungen: Prüfe zunächst, ob die Prozessvoraussetzungen (z.B. Zuständigkeit des Gerichts) erfüllt sind. Wenn nicht strittig, reicht es aus zu bestätigen, dass die Voraussetzungen erfüllt sind.\n-Rechtliche Würdigung:\nEruieren Sie relevante Rechtssätze basierend auf den behaupteten und rechtlich relevanten Tatsachen.\n-Setzen Sie sich mit den rechtlichen Standpunkten der Parteien auseinander.\n-Beachten Sie die Beweislastverteilung und würdigen Sie die Beweise frei, aber berücksichtigen Sie relevante gesetzliche Beweisregeln.\n-Iura novit curia: Ihre rechtliche Würdigung muss nicht zwangsläufig dem rechtlichen Vorbringen der Parteien entsprechen. Berücksichtigen Sie andere mögliche Argumentationslinien.\n-Zusammenfassung: Fassen Sie am Ende Ihrer Erwägungen das Ergebnis Ihrer rechtlichen Würdigung zusammen.\n-Output: Der generierte Text sollte strukturiert, klar und in der Form von typischen Erwägungen eines Schweizer Gerichtsurteils sein.\n\nSachverhalt des Schweizer Gerichtsurteils:\n\n{input}\n\nErwägungen:\n\n'
        elif lang == 'fr':
            instruction = f"But: Générer des considérations basées sur les faits donnés d'un jugement suisse.\nContexte: Un jugement suisse est composé du rubrum, des faits, des considérations, du dispositif (formule du jugement) et de la signature. Les considérations sont l'appréciation juridique des événements par le tribunal.\nInstructions:\n- Comprendre les faits: Les faits donnés contiennent des faits contestés et non contestés, les demandes des parties, la procédure de preuve et l'historique du procès.\n- Commencer par les conditions de procédure: Vérifiez d'abord si les conditions de procédure (par exemple, la compétence du tribunal) sont remplies. Si cela n'est pas contesté, il suffit de confirmer que les conditions sont remplies.\n- Appréciation juridique:\nÉvaluez les dispositions juridiques pertinentes basées sur les faits allégués et juridiquement pertinents.\n- Confrontez-vous aux points de vue juridiques des parties.\n- Tenez compte de la répartition de la charge de la preuve et évaluez les preuves librement, mais tenez compte des règles légales de preuve pertinentes.\n- Iura novit curia: Votre appréciation juridique ne doit pas nécessairement correspondre aux arguments juridiques présentés par les parties. Considérez d'autres lignes d'argumentation possibles.\n- Résumé: Résumez à la fin de vos considérations le résultat de votre appréciation juridique.\n- Résultat: Le texte généré devrait être structuré, clair et sous la forme de considérations typiques d'un jugement suisse.\n\nFaits du jugement suisse:\n\n{input}\n\nConsidérations:\n\n"
        elif lang == 'it':
            instruction = f"Obiettivo: Generare considerazioni basate sui fatti presentati in una sentenza svizzera.\nContesto: Una sentenza svizzera si compone di rubrum, fatti, considerazioni, dispositivo (formula della sentenza) e firma. Le considerazioni rappresentano la valutazione giuridica degli eventi da parte del tribunale.\nIstruzioni:\n- Comprendere i fatti: I fatti presentati includono fatti contestati e non contestati, le richieste delle parti, la procedura probatoria e la storia del processo.\n- Iniziare con le condizioni processuali: Verificare prima di tutto se le condizioni processuali (ad es. la competenza del tribunale) sono soddisfatte. Se non contestate, è sufficiente confermare che le condizioni sono state soddisfatte.\n- Valutazione giuridica:\nValutare le norme giuridiche rilevanti in base ai fatti affermati e giuridicamente rilevanti.\n- Confrontarsi con i punti di vista giuridici delle parti.\n- Tenere conto della distribuzione dell'onere della prova e valutare le prove liberamente, ma considerare le regole di prova legalmente rilevanti.\n- Iura novit curia: La tua valutazione giuridica non deve necessariamente corrispondere alle argomentazioni giuridiche delle parti. Considera altre possibili linee di argomentazione.\n- Riassunto: Riassumere alla fine delle tue considerazioni il risultato della tua valutazione giuridica.\n- Risultato: Il testo generato dovrebbe essere strutturato, chiaro e nella forma di considerazioni tipiche di una sentenza svizzera.\n\nFatti della sentenza svizzera:\n\n{input}\n\nConsiderazioni:\n\n"
    elif task == 'summ':
        instruction = f"Ziel: Generiere eine Regeste basierend auf dem gegebenen Sachverhalt, Erwägungen und Dispositiv eines Schweizer Gerichtsurteils.\nHintergrund: Ein Schweizer Gerichtsurteil besteht aus Sachverhalt, Erwägungen und Dispositiv. Die Regeste dient als Kurzzusammenfassung einer gerichtlichen Entscheidung mit Leitsätzen.\nAnweisung:\n1. Sachverhalt: Überprüfe den Sachverhalt.\n2. Erwägungen: Analysiere die Erwägungen.\n3. Dispositiv: Berücksichtige das Dispositiv.\n4. Erstelle die Regeste: Fasse den Fall in Leitsätzen zusammen.\nOutput: Die Regeste sollte eine klare Kurzzusammenfassung mit Leitsätzen bieten.\n\nGegebener Sachverhalt, Erwägungen und Dispositiv:\n\n{input}\n\nGeneriere nun die kurze Regeste.\n\nRegeste:\n\n"

    else:
        raise NotImplementedError("Task not implemented yet")
    return instruction

def generate_completions(dataset, input_length, output_length, model):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    dataset_with_predicted = []
    logger.info("Using model: " + model)
    if len(dataset) > 50:
        input("Are you sure you want to generate completions for " + str(len(dataset)) + " examples? Press Enter to continue...")
    if len(dataset) > 100:
        input("Are you really sure? Press Enter to continue...")

    for entry in dataset:
        instruct = create_instruction(entry["input"], entry["lang"], task_name)
        # measure time to generate completion
        start = time.time()
        completion = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "user", "content": instruct}
                ],
            max_tokens=output_length,
            )
        end = time.time()
        logger.info("Time to generate completion: " + str(end - start))
        print(completion)
        predicted = completion.choices[0].message["content"]
        entry_with_predicted = {"input": entry["input"], "target": entry["target"], "predicted": predicted, "lang": entry["lang"]}
        dataset_with_predicted.append(entry_with_predicted)

    return dataset_with_predicted


########################################################################################################################

# measure time for whole script
start = time.time()

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

if args.sum == "True":
    task_name = "summ"
elif args.origin == "True":
    task_name = "cvg-origin"
else:
    task_name = "cvg"


logger.info("Project name: " + project_name)
os.environ["WANDB_PROJECT"] = project_name
os.environ["WANDB_RUN_GROUP"] = f"{model_name}, {len(eval_dataset)}"

# add train size, seq length to output dir
output_dir = f"output/{task_name}/{args.model.split('/')[-1]}_evalsize={args.eval_size}_inlen={args.input_length}_outlen={args.output_length}_origin={args.origin}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}"
# set wandb run name
wandb.init(name=output_dir.split('/')[-1]) # means
# log output dir to wandb
wandb.log({"output_dir": output_dir})
# log task name to wandb
wandb.log({"task_name": task_name})

logger.info("Model name:" + model_name + " finetune: " + " output_dir: " + output_dir)
logger.info("Eval dataset size: " + str(len(eval_dataset)) )

# log all args to wandb
wandb.config.update(args)

output_examples = []

# prepare dataset for generation
eval_data = prepare_dataset(eval_dataset, args.input_length, args.output_length, args.sum, args.origin)

# generate output examples using API completions
completion_dataset = generate_completions(eval_data, args.input_length, args.output_length, model_name)

# Evaluate model on test dataset
meteor_score_avg, rouge_score_avg, bleu_score_avg, bert_score_avg = compute_scores(completion_dataset)

# Print and log scores to wandb
log_test_scores(meteor_score_avg, rouge_score_avg, bleu_score_avg, bert_score_avg)

try:
    # save output examples to file
    export_output(output_examples, output_dir, task_name)
except Exception as e:
    logger.info("Error exporting output examples: " + str(e))

# measure time for whole script
end = time.time()
# in readable format
logger.info("Time for whole script: " + str(datetime.timedelta(seconds=end - start)))