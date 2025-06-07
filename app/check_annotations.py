import json
import argparse
import pandas as pd
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gzip

def json_generator(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                yield {"id": str(data["id"]), "response": {"input_labels": int(data["label"])}}
            except Exception as e:
                print(f"JSON decode error: {e}")
                raise StopIteration

def json_generator_gz(filepath: str):
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                data =  json.loads(line)
                yield data
            except Exception as e:
                print(f"JSON decode error: {e}")
                raise StopIteration


def csv_generator(filepath: str):
    csv_df = pd.read_csv(filepath)
    for index, row in csv_df.iterrows():
        yield {"id": str(row["id"]), "response": {"input_labels": row["predicted_label"]}}
   
            
def label_agg(row):
    res = 0
    for x in row:
        if x != 0:
            res+= 1
    return res


def main(args):
    ground_truth_df = pd.read_json(args.dataset_path, lines=False, orient='index', convert_dates=False)
    ground_truth_df['label'] = ground_truth_df['labels'].apply(label_agg) # Assumes 'labels' column contains lists
    ground_truth_df['id'] = ground_truth_df['tweet_url'].str.extract(r'/status/(\d+)')

    confusion_matrix = np.zeros((4, 4))
    with open(f"{args.split}/test_ids.txt", 'r') as f:
        val_ids = set(line.strip() for line in tqdm.tqdm(f, desc="Building test set"))
   
    if args.annotation_path.endswith('.gz'):
        file_generator = json_generator_gz
    elif args.annotation_path.endswith('.csv'):
        file_generator = csv_generator
    else:
        file_generator = json_generator

    num_annotations = sum(1 for _ in file_generator(args.annotation_path))
    num_annotations = max(num_annotations, len(val_ids))
    accuracy = 0
    mae = 0
    rmse = 0
    tp, tn, fp, fn = 0, 0, 0, 0
    f1_confusion_mat = np.zeros((2, 2)) 
    for line in tqdm.tqdm(file_generator(args.annotation_path), total=num_annotations):
        id = line['id']
        if id not in val_ids:
            continue

        label = line['response']['input_labels']        
        ground_truth_label = ground_truth_df[ground_truth_df['id'] == id]['label'].values[0]
        gt_hate_label = ground_truth_label >= 2
        label_hate_label = label >= 2
        # Ground truth labels are going down the rows and annotated labels are going across the columns
        accuracy += int(ground_truth_label == label)
        mae += abs(ground_truth_label - label)
        rmse += (ground_truth_label - label) ** 2

        if gt_hate_label == 0 and label_hate_label == 0:
            f1_confusion_mat[0, 0] += 1
        elif gt_hate_label == 0 and label_hate_label == 1:
            f1_confusion_mat[0, 1] += 1
        elif gt_hate_label == 1 and label_hate_label == 0:
            f1_confusion_mat[1, 0] += 1
        elif gt_hate_label == 1 and label_hate_label == 1:
            f1_confusion_mat[1, 1] += 1

        confusion_matrix[ground_truth_label][label] += 1

    f1_confusion_mat /= num_annotations
    

    accuracy /= num_annotations
    mae /= num_annotations
    rmse = np.sqrt(rmse / num_annotations)
    normalized_confusion_matrix = confusion_matrix / num_annotations
    relative_confusion_matrix = normalized_confusion_matrix / np.sum(normalized_confusion_matrix, axis=1, keepdims=True)

    observed_agreement = np.trace(confusion_matrix) / num_annotations
    row_marginals = np.sum(confusion_matrix, axis=1)
    col_marginals = np.sum(confusion_matrix, axis=0)
    expected_agreement = np.sum((row_marginals * col_marginals)) / (num_annotations ** 2)

    print(f"Confusion Matrix:\n{confusion_matrix}")
    print(f"Normalized Confusion Matrix:\n{normalized_confusion_matrix}")
    print(f"Conditional Confusion Matrix:\n{relative_confusion_matrix}")
    print(f"Accuracy (HateScore): {accuracy}")
    print(f"MAE (HateScore): {mae}")
    print(f"RMSE (HateScore): {rmse}")
    print(f"Soft Accuracy: {np.trace(f1_confusion_mat)}")
    print(f"Precision: {f1_confusion_mat[1, 1] / (f1_confusion_mat[1, 1] + f1_confusion_mat[0, 1])}")
    print(f"Recall: {f1_confusion_mat[1, 1] / (f1_confusion_mat[1, 1] + f1_confusion_mat[1, 0])}")
    print(f"F1 Score: {f1_confusion_mat[1, 1] / (f1_confusion_mat[1, 1] + 0.5*(f1_confusion_mat[0, 1] + f1_confusion_mat[1, 0]))}")
    print(f"Observed Agreement: {observed_agreement}")
    print(f"Expected Agreement: {expected_agreement}")
    print(f"Cohen's Kappa: {(observed_agreement - expected_agreement) / (1-expected_agreement)}")

    plt.figure(figsize=(6, 4))
    sns.heatmap(relative_confusion_matrix, cmap='flare', annot=True, fmt=".2f", cbar=True)
    plt.title("Conditional Confusion Matrix (P(LM-7B | GT))")
    plt.xlabel("Model Prediction")
    plt.ylabel("Ground Truth Label")
    plt.tight_layout()
    plt.savefig(f"{args.graph_name}.png", dpi=300)
    plt.show()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, help="Path to the MMHS150K_GT.json file")
    parser.add_argument("--annotation-path", type=str, help="Path to the output JSONL file")
    parser.add_argument("--graph-name", type=str, help="Name of the graph")
    parser.add_argument("--split", type=str)
    args = parser.parse_args()
    main(args)