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

    r1r2_agreement = 0
    r1r3_agreement = 0
    r2r3_agreement = 0  

    r1r2_agreement_wout_zero = 0
    r1r3_agreement_wout_zero = 0
    r2r3_agreement_wout_zero = 0   

    r1r2_agreement_hate_present = 0
    r1r3_agreement_hate_present = 0
    r2r3_agreement_hate_present = 0


    count_wout_zero = 0
    for idx, row in tqdm(ground_truth_df.iterrows(), total=counts):
        labels = row['labels']
        if len(labels) < 3:
            for _ in range(3 - len(labels)):
                labels.append(labels[-1])
        
        labels = labels[:3]    
        r1r2[labels[0]][labels[1]] += 1
        r1r3[labels[0]][labels[2]] += 1
        r2r3[labels[1]][labels[2]] += 1

        if labels[0] == labels[1]:
            r1r2_agreement += 1

        if labels[0] == labels[1] and labels[0] != 0:
            r1r2_agreement_wout_zero += 1
            count_wout_zero += 1
        
        if labels[0] == labels[2]:
            r1r3_agreement += 1

        if labels[0] == labels[2] and labels[0] != 0:
            r1r3_agreement_wout_zero += 1
            count_wout_zero += 1

        if labels[1] == labels[2]:
            r2r3_agreement += 1

        if labels[1] == labels[2] and labels[1] != 0:
            r2r3_agreement_wout_zero += 1
            count_wout_zero += 1

        if (labels[0] > 0 and labels[1] > 0) or (labels[0] == labels[1]):
            r1r2_agreement_hate_present += 1

        if (labels[0] > 0 and labels[2] > 0) or (labels[0] == labels[2]):
            r1r3_agreement_hate_present += 1

        if (labels[1] > 0 and labels[2] > 0) or (labels[1] == labels[2]):
            r2r3_agreement_hate_present += 1
        


    normalized_r1r2 = r1r2 / counts
    normalized_r1r3 = r1r3 / counts
    normalized_r2r3 = r2r3 / counts

    prob_agreement_r1r2 = r1r2_agreement / counts
    prob_agreement_r1r3 = r1r3_agreement / counts
    prob_agreement_r2r3 = r2r3_agreement / counts

    prob_agreement_r1r2_wout_zero = r1r2_agreement_wout_zero / count_wout_zero
    prob_agreement_r1r3_wout_zero = r1r3_agreement_wout_zero / count_wout_zero
    prob_agreement_r2r3_wout_zero = r2r3_agreement_wout_zero / count_wout_zero

    prob_agreement_r1r2_hate_present = r1r2_agreement_hate_present / counts
    prob_agreement_r1r3_hate_present = r1r3_agreement_hate_present / counts
    prob_agreement_r2r3_hate_present = r2r3_agreement_hate_present / counts

    relative_r1r2 = normalized_r1r2 / np.sum(normalized_r1r2, axis=0, keepdims=True)
    relative_r1r3 = normalized_r1r3 / np.sum(normalized_r1r3, axis=0, keepdims=True)
    relative_r2r3 = normalized_r2r3 / np.sum(normalized_r2r3, axis=0, keepdims=True)


    # === SETTINGS FOR LATEX-READY PLOT ===
    plt.rcParams.update({
        'text.usetex': True,                 # Use LaTeX for rendering (requires LaTeX installation)
        'font.family': 'serif',
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 18,
        'lines.linewidth': 2,
        'lines.markersize': 4,              # Smaller markers
        'figure.dpi': 300,
        'figure.figsize': (6, 4)            # A4 report friendly
    })

    plt.figure()
    sns.heatmap(relative_r1r2, annot=True, xticklabels=Allowed_labels, yticklabels=Allowed_labels, cmap="flare", cbar=True, fmt=".2f")
    plt.xlabel("Researcher 2")
    plt.ylabel("Researcher 1")
    plt.tight_layout()
    plt.savefig("./plots/r1r2_cond_aggrement.pdf")
    
    plt.figure()
    sns.heatmap(relative_r1r3, annot=True, xticklabels=Allowed_labels, yticklabels=Allowed_labels, cmap="flare", cbar=True, fmt=".2f")
    plt.xlabel("Researcher 3")
    plt.ylabel("Researcher 1")
    plt.tight_layout()
    plt.savefig("./plots/r1r3_cond_agreement.pdf")

    plt.figure()
    sns.heatmap(relative_r2r3, annot=True, xticklabels=Allowed_labels, yticklabels=Allowed_labels, cmap="flare", cbar=True, fmt=".2f")
    plt.xlabel("Researcher 3")
    plt.ylabel("Researcher 2")
    plt.tight_layout()
    plt.savefig("./plots/r2r3_cond_aggrement.pdf")

    # Use seaborn's theme for styling
    sns.set_theme(style="whitegrid")

    # Plot the agreement segment
    plt.figure()
    plt.bar([r'\textbf{A1-A2}', r'\textbf{A1-A3}', r'\textbf{A2-A3}'], [prob_agreement_r1r2, prob_agreement_r1r3, prob_agreement_r2r3], label='Agreement')
    # Plot the disagreement segment stacked on top
    plt.bar([r'\textbf{A1-A2}', r'\textbf{A1-A3}', r'\textbf{A2-A3}'], [1 - prob_agreement_r1r2, 1 - prob_agreement_r1r3, 1 - prob_agreement_r2r3], bottom=[prob_agreement_r1r2, prob_agreement_r1r3, prob_agreement_r2r3], label='Disagreement')
    plt.ylabel(r'\textbf{Proportion}')
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig("./plots/agreement_disagreement.pdf")

    plt.figure()
    plt.bar([r'\textbf{A1-A2}', r'\textbf{A1-A3}', r'\textbf{A2-A3}'], [prob_agreement_r1r2_wout_zero, prob_agreement_r1r3_wout_zero, prob_agreement_r2r3_wout_zero], label='Agreement')
    plt.ylabel(r'\textbf{Proportion}')
    plt.bar([r'\textbf{A1-A2}', r'\textbf{A1-A3}', r'\textbf{A2-A3}'], [1 - prob_agreement_r1r2_wout_zero, 1 - prob_agreement_r1r3_wout_zero, 1 - prob_agreement_r2r3_wout_zero], bottom=[prob_agreement_r1r2_wout_zero, prob_agreement_r1r3_wout_zero, prob_agreement_r2r3_wout_zero], label='Disagreement')
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig("./plots/agreement_disagreement_wout_zero.pdf")

    plt.figure()
    plt.bar([r'\textbf{A1-A2}', r'\textbf{A1-A3}', r'\textbf{A2-A3}'], [prob_agreement_r1r2_hate_present, prob_agreement_r1r3_hate_present, prob_agreement_r2r3_hate_present], label='Agreement')
    plt.ylabel(r'\textbf{Proportion}')
    plt.bar([r'\textbf{A1-A2}', r'\textbf{A1-A3}', r'\textbf{A2-A3}'], [1 - prob_agreement_r1r2_hate_present, 1 - prob_agreement_r1r3_hate_present, 1 - prob_agreement_r2r3_hate_present], bottom=[prob_agreement_r1r2_hate_present, prob_agreement_r1r3_hate_present, prob_agreement_r2r3_hate_present], label='Disagreement')
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig("./plots/agreement_disagreement_hate_present.pdf")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-json-path", type=str, required=True, help="Path to the dataset json file")
    main(parser.parse_args())