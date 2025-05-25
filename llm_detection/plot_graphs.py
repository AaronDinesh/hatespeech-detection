"""
This file is mainly to help plot the graphs. It is not intended to be reused as-is. Just a hacky script to get something
plotted for the poster + report 
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
from tqdm import tqdm


def label_agg(row):
    res = 0
    for x in row:
        if x != 0:
            res+= 1
    return res

Allowed_labels = ["NotHate", "Racist", "Sexist", "Homophobe", "Religion", "OtherHate"]

r1r2 = np.zeros((6, 6))
r1r3 = np.zeros((6, 6))
r2r3 = np.zeros((6, 6))

def main(args):
    # Load ground truth

    ground_truth_df = pd.read_json(args.dataset_json_path, lines=False, orient='index', convert_dates=False)
    ground_truth_df['label'] = ground_truth_df['labels'].apply(label_agg) # Assumes 'labels' column contains lists
    ground_truth_df['id'] = ground_truth_df['tweet_url'].str.extract(r'/status/(\d+)')
    counts = len(ground_truth_df)    

    wrong_count = 0
    for idx, row in tqdm(ground_truth_df.iterrows(), total=counts):
        labels = row['labels']
        if len(labels) < 3:
            for _ in range(3 - len(labels)):
                labels.append(labels[-1])
        
        labels = labels[:3]    
        r1r2[labels[0]][labels[1]] += 1
        r1r3[labels[0]][labels[2]] += 1
        r2r3[labels[1]][labels[2]] += 1

    normalized_r1r2 = r1r2 / counts
    normalized_r1r3 = r1r3 / counts
    normalized_r2r3 = r2r3 / counts

    relative_r1r2 = normalized_r1r2 / np.sum(normalized_r1r2, axis=0, keepdims=True)
    relative_r1r3 = normalized_r1r3 / np.sum(normalized_r1r3, axis=0, keepdims=True)
    relative_r2r3 = normalized_r2r3 / np.sum(normalized_r2r3, axis=0, keepdims=True)

    plt.figure(figsize=(6, 4))
    sns.heatmap(relative_r1r2, annot=True, xticklabels=Allowed_labels, yticklabels=Allowed_labels, cmap="flare", cbar=True, fmt=".2f")
    plt.title("Conditional Agreement between Researcher 1 and 2 (P(R2 | R1))")
    plt.xlabel("Researcher 2")
    plt.ylabel("Researcher 1")
    plt.tight_layout()
    plt.savefig("r1r2.png")
    plt.show()

    sns.heatmap(relative_r1r3, annot=True, xticklabels=Allowed_labels, yticklabels=Allowed_labels, cmap="flare", cbar=True, fmt=".2f")
    plt.title("Conditional Agreement between Researcher 1 and 3 (P(R3 | R1))")
    plt.xlabel("Researcher 3")
    plt.ylabel("Researcher 1")
    plt.tight_layout()
    plt.savefig("r1r3.png")
    plt.show()


    sns.heatmap(relative_r2r3, annot=True, xticklabels=Allowed_labels, yticklabels=Allowed_labels, cmap="flare", cbar=True, fmt=".2f")
    plt.title("Conditional Agreement between Researcher 2 and 3 (P(R3 | R2))")
    plt.xlabel("Researcher 3")
    plt.ylabel("Researcher 2")
    plt.tight_layout()
    plt.savefig("r2r3.png")
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-json-path", type=str, required=True, help="Path to the dataset json file")
    main(parser.parse_args())