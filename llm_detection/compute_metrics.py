import argparse
import json
import gzip
import pandas as pd
from tqdm import tqdm
import os
import pydantic
import typing
import ast
import numpy as np


Allowed_labels = typing.Literal[
    "NotHate", "Racist", "Sexist", "Homophobe", "Religion", "OtherHate", "HateSpeech"
]

# class Response_schema(pydantic.BaseModel):
#     input_labels: pydantic.conlist(Allowed_labels, min_length=3, max_length=3)

# class Response_schema(pydantic.BaseModel):
#     input_labels: pydantic.conlist(Allowed_labels, min_length=1, max_length=1)

class Response_schema(pydantic.BaseModel):
    """
    Response Schema to use when the LLM just outputs a number indicating the hatefulness score
    """
    input_labels: int

class Output_schema(pydantic.BaseModel):
    id: str
    response: Response_schema


# All the allowed labels
labels = ["NotHate", "Racist", "Sexist", "Homophobe", "Religion", "OtherHate", "HateSpeech"]


# The label mapping (whether the label is hate or not)
labels_mapping = {
    "NotHate": -1,
    "Racist": 1,
    "Sexist": 1,
    "Homophobe": 1,
    "Religion": 1,
    "OtherHate": 1,
    "HateSpeech": 1
}


class LabelMatcher:
    """
    This class is used to compute the intersection between the ground truth and the LLM output. Ie if the ground truth
    output is ["Religion", "NotHate", "Racist"] and the LLM output is ["Religion", "Homophobe", "NotHate"] then the
    intersection is ["Hateful", "NotHate"] so the score it has will be 2, and "2-Match" in the results dict will be
    incremented by 1.
    
    """
    def __init__(self):
        self.results = {
            "3-Match": 0,
            "2-Match": 0,
            "1-Match": 0,
            "0-Match": 0,
            "Avg-IoU": 0,
            "Total_count": 0
        }
    def update_scores(self, ground_truth: tuple[str, list[str]], llm_output: tuple[str, list[str]]):
        assert ground_truth[0] == llm_output[0], "The ids of the ground truth and llm output must be the same"

        # Create a set for easy intersection and union calculations
        ground_truth_set = set(ground_truth[1])
        llm_output_set = set(llm_output[1])

        # Compute intersection and union
        intersection = ground_truth_set.intersection(llm_output_set)
        union = ground_truth_set.union(llm_output_set)

        self.results["Total_count"] += 1

        if len(intersection) == 3:
            self.results["3-Match"] += 1
        elif len(intersection) == 2:
            self.results["2-Match"] += 1
        elif len(intersection) == 1:
            self.results["1-Match"] += 1
        elif len(intersection) == 0:
            self.results["0-Match"] += 1

        curr_IoU = len(intersection) / len(union)
        self.results["Avg-IoU"] += (curr_IoU - self.results["Avg-IoU"]) / self.results["Total_count"] 
    
    def get_results(self):
        return self.results
    
class HatefulMatcher:
    """
    This class will compute the 'hatefulness' of the LLM output. If the ground truth contains a majority of 'hateful'
    labels i.e. ["Religion", "Homophobe", "NotHate"] then the hateful score will be 1, whereas if it is ["NotHate",
    "NotHate", "Homophobe"] then the hateful score will be 0. We do this for both the LLM output and the ground truth
    and then calculat ethe FP, TP, FN and TN.
    """

    def __init__(self):
        self.results = {
            "False-Positive": 0,
            "True-Positive": 0,
            "False-Negative": 0,
            "True-Negative": 0,
            "Count": 0
        }
    
    def compute_majority(self, labels: list[str]):
        num_hate = sum(1 for label in labels[1] if labels_mapping[label] == 1)
        num_total = len(labels)
        return num_hate > num_total / 2

    def update_scores(self, ground_truth: tuple[str, list[str]], llm_output: tuple[str, list[str]]):
        assert ground_truth[0] == llm_output[0], "The ids of the ground truth and llm output must be the same"

        # This will be 1 if it is hateful, 0 otherwise        
        ground_truth_hate = self.compute_majority(ground_truth)
        llm_output_hate = self.compute_majority(llm_output)
    
        self.results["Count"] += 1
        # Compute the scores for analysis later
        if ground_truth_hate and llm_output_hate:
            self.results["True-Positive"] += 1
        elif ground_truth_hate and not llm_output_hate:
            self.results["False-Negative"] += 1
        elif not ground_truth_hate and llm_output_hate:
            self.results["False-Positive"] += 1
        elif not ground_truth_hate and not llm_output_hate:
            self.results["True-Negative"] += 1

    
    def get_results(self):
        return self.results
    
    def get_metrics(self):
        """
        Computes the Precision, Accuracy, Recall and F1 score
        """
        return {
            "Accuracy": (self.results["True-Positive"] + self.results["True-Negative"]) / self.results["Count"],
            "Recall": self.results["True-Positive"] / (self.results["True-Positive"] + self.results["False-Negative"]),
            "Precision": self.results["True-Positive"] / (self.results["True-Positive"] + self.results["False-Positive"]),
            "F1": self.results["True-Positive"] / (self.results["True-Positive"] + 0.5*(self.results["False-Positive"] + self.results["False-Negative"])),
            "Cohen's Kappa": 2*(self.results["True-Positive"] * self.results["True-Negative"] - self.results["False-Negative"] * self.results["False-Positive"]) / ((self.results["True-Positive"] + self.results["False-Positive"]) * (self.results["True-Positive"] + self.results["False-Negative"]) * (self.results["False-Negative"] + self.results["True-Negative"]))
        }
    
    def plot_confusion(self, save_dir: str):
        raise NotImplementedError
    
class HateNotHateMatcher():
    def __init__(self):
        self.results = {
            "False-Positive": 0,
            "True-Positive": 0,
            "False-Negative": 0,
            "True-Negative": 0,
            "Count": 0
        }

        self.true_hate = 0
        self.true_not_hate = 0
    
    def compute_majority(self, labels: list[str]):
        num_hate = sum(1 for label in labels[1] if labels_mapping[label] == 1)
        num_total = len(labels)
        return num_hate > num_total / 2


    def update_scores(self, ground_truth: tuple[str, list[str]], llm_output: tuple[str, list[str]]):
        assert ground_truth[0] == llm_output[0], "The ids of the ground truth and llm output must be the same"

        # This will be 1 if it is hateful, 0 otherwise
        ground_truth_hate = self.compute_majority(ground_truth)
        llm_output_hate = 1 if "HateSpeech" in llm_output[1] else 0

        if ground_truth_hate:
            self.true_hate += 1
        else:
            self.true_not_hate += 1

        self.results["Count"] += 1
        # Compute the scores for analysis later
        if ground_truth_hate and llm_output_hate:
            self.results["True-Positive"] += 1
        elif ground_truth_hate and not llm_output_hate:
            self.results["False-Negative"] += 1
        elif not ground_truth_hate and llm_output_hate:
            self.results["False-Positive"] += 1
        elif not ground_truth_hate and not llm_output_hate:
            self.results["True-Negative"] += 1

    
    def get_results(self):
        print(f"GT NotHate: {self.true_not_hate}, GT Hate: {self.true_hate}")
        return self.results
    
    def get_metrics(self):
        """
        Computes the Precision, Accuracy, Recall and F1 score
        """
        return {
            "Accuracy": (self.results["True-Positive"] + self.results["True-Negative"]) / self.results["Count"],
            "Recall": self.results["True-Positive"] / (self.results["True-Positive"] + self.results["False-Negative"]),
            "Precision": self.results["True-Positive"] / (self.results["True-Positive"] + self.results["False-Positive"]),
            "F1": self.results["True-Positive"] / (self.results["True-Positive"] + 0.5*(self.results["False-Positive"] + self.results["False-Negative"])),
            "Cohen's Kappa": 2*(self.results["True-Positive"] * self.results["True-Negative"] - self.results["False-Negative"] * self.results["False-Positive"]) / ((self.results["True-Positive"] + self.results["False-Positive"]) * (self.results["True-Positive"] + self.results["False-Negative"]) * (self.results["False-Negative"] + self.results["True-Negative"]))
        }
    
    def plot_confusion(self, save_dir: str):
        raise NotImplementedError
    

class HatefulScore:
    def __init__(self):
        self.mae = 0
        self.mse = 0
        self.count = 0
        self.confusion_matrix = np.zeros((4, 4))
        pass
    
    def compute_hate_score(self, labels: list[str]):
        return sum(1 for label in labels[1] if labels_mapping[label] == 1)
    
    def update_scores(self, ground_truth: tuple[str, list[str]], llm_output: tuple[str, int]):
        assert ground_truth[0] == llm_output[0], "The ids of the ground truth and llm output must be the same"
        self.count += 1
        gt_hate = self.compute_hate_score(ground_truth)
        self.mae += abs(gt_hate - llm_output[1])
        self.mse += (gt_hate - llm_output[1])**2
        self.confusion_matrix[gt_hate, llm_output[1]] += 1
        
        
    def get_metrics(self):
        observed_agreement = np.trace(self.confusion_matrix) / self.count
        row_marginals = np.sum(self.confusion_matrix, axis=1)
        col_marginals = np.sum(self.confusion_matrix, axis=0)
        expected_agreement = np.sum((row_marginals * col_marginals)) / (self.count ** 2)
        self.normalized_confusion_matrix = self.confusion_matrix / self.count
         
        return {
            "MAE": self.mae / self.count,
            "MSE": self.mse / self.count,
            "Cohen's Kappa": (observed_agreement - expected_agreement) / (1-expected_agreement),
            "Observed Agreement": observed_agreement,
            "Expected Agreement": expected_agreement,
            "Confusion Matrix": self.normalized_confusion_matrix,
            "Relative Confusion Matrix": self.normalized_confusion_matrix / np.sum(self.normalized_confusion_matrix, axis=1, keepdims=True),
            "Confusion Matrix Row": "Ground Truth",
            "Confusion Matrix Col": "LLM Output"
        }


def json_generator(filepath: str):
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                yield json.loads(line)
            except Exception as e:
                print(f"JSON decode error: {e}")
                raise StopIteration



def main(parser: argparse.ArgumentParser):
    args = parser.parse_args()
    ground_truth = args.ground_truth
    llm_output = args.llm_output
    train_test_split = args.train_test_split
    binary_labels = args.binary_labels
    plot_save_dir = args.plot_save_dir
    text_labels = args.text_labels
    hateful_score = args.hateful_score
    
    if binary_labels: 
        hate_not_hate_matcher = HateNotHateMatcher()
    elif text_labels:
        hateful_matcher = HatefulMatcher()
        label_matcher = LabelMatcher()
    elif hateful_score:
        hateful_score = HatefulScore()
    else:
        print("No valid LLM output format specified. Please specify --binary-labels, --text-labels or --hateful-score")
        return

    if not os.path.exists(llm_output):
        print(f"Error: {llm_output} does not exist")
        return
    
    if not os.path.exists(ground_truth):
        print(f"Error: {ground_truth} does not exist")
        return
    
    if train_test_split is not None:
        if not os.path.exists(train_test_split):
            print(f"Error: {train_test_split} does not exist")
            return

    # Load the ground truth
    ground_truth_df = pd.read_json(ground_truth, lines=False, orient='index', convert_dates=False, convert_axes=False, dtype=str)
    id_to_labels = dict(zip(ground_truth_df.index, ground_truth_df["labels_str"]))

    # Loading the LLM output
    for line in tqdm(json_generator(llm_output), desc="Comparing LLM with GT...", total=len(ground_truth_df), unit=" prompts", leave=False):
        #Parse the LLM output into a Python object
        parsed_llm_output = Output_schema.model_validate(line)

        #Parse the ground truth into a Python list
        ground_truth_labels = ast.literal_eval(id_to_labels[str(parsed_llm_output.id)])

        if binary_labels:
            hate_not_hate_matcher.update_scores((parsed_llm_output.id, ground_truth_labels), (parsed_llm_output.id, parsed_llm_output.response.input_labels))
        elif text_labels:
            hateful_matcher.update_scores((parsed_llm_output.id, ground_truth_labels), (parsed_llm_output.id, parsed_llm_output.response.input_labels))
            label_matcher.update_scores((parsed_llm_output.id, ground_truth_labels), (parsed_llm_output.id, parsed_llm_output.response.input_labels))
        else:
            hateful_score.update_scores((parsed_llm_output.id, ground_truth_labels), (parsed_llm_output.id, parsed_llm_output.response.input_labels))

    print("Results:")
    if binary_labels:
        print(hate_not_hate_matcher.get_results())
        print(hate_not_hate_matcher.get_metrics())
        [print("{}: {}".format(k, v)) for k, v in hate_not_hate_matcher.results.items()]
        hate_not_hate_matcher.plot_confusion(plot_save_dir)
    elif text_labels:
        print(hateful_matcher.get_results())
        print(hateful_matcher.get_metrics())
        [print("{}: {}".format(k, v)) for k, v in hateful_matcher.results.items()]
        print(label_matcher.get_results())
        hateful_matcher.plot_confusion(plot_save_dir)
    else:
        print(hateful_score.get_metrics())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", type=str, required=True, help="Path to the MMHS150K_GT.json file")
    parser.add_argument("--llm-output", type=str, required=True, help="Path to the output .jsonl.gz file")
    parser.add_argument("--train-test-split", type=str, help="Path to the train-test-split.csv file")
    parser.add_argument("--binary-labels", action='store_true', help="Whether the output has binary labels or not")
    parser.add_argument("--text-labels", action='store_true', help="Whether the output has text labels or not")
    parser.add_argument("--hateful-score", action='store_true', help="Whether the output uses hateful score or not")
    parser.add_argument("--plot-save-dir", type=str, help="Path to save the plots")
    main(parser)


