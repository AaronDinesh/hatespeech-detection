import argparse
import json
import gzip
import pandas as pd
from tqdm import tqdm
import os
import pydantic
import typing
import ast

Allowed_labels = typing.Literal[
    "NotHate", "Racist", "Sexist", "Homophobe", "Religion", "OtherHate"
]

class Response_schema(pydantic.BaseModel):
    input_labels: pydantic.conlist(Allowed_labels, min_length=3, max_length=3)

class Output_schema(pydantic.BaseModel):
    id: str
    response: Response_schema


# All the allowed labels
labels = ["NotHate", "Racist", "Sexist", "Homophobe", "Religion", "OtherHate"]


# The label mapping (whether the label is hate or not)
labels_mapping = {
    "NotHate": -1,
    "Racist": 1,
    "Sexist": 1,
    "Homophobe": 1,
    "Religion": 1,
    "OtherHate": 1
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

    def update_scores(self, ground_truth: tuple[str, list[str]], llm_output: tuple[str, list[str]]):
        assert ground_truth[0] == llm_output[0], "The ids of the ground truth and llm output must be the same"

        # This will be 1 if it is hateful, 0 otherwise
        ground_truth_hate = sum([labels_mapping[label] for label in ground_truth[1]]) > 0
        llm_output_hate = sum([labels_mapping[label] for label in llm_output[1]]) > 0


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
            "F1": self.results["True-Positive"] / (self.results["True-Positive"] + 0.5*(self.results["False-Positive"] + self.results["False-Negative"]))
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

    hateful_matcher = HatefulMatcher()
    label_matcher = LabelMatcher()

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

        hateful_matcher.update_scores((parsed_llm_output.id, ground_truth_labels), (parsed_llm_output.id, parsed_llm_output.response.input_labels))
        label_matcher.update_scores((parsed_llm_output.id, ground_truth_labels), (parsed_llm_output.id, parsed_llm_output.response.input_labels))

    print("Results:")
    print(hateful_matcher.get_results())
    print(hateful_matcher.get_metrics())
    print(label_matcher.get_results())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", type=str, required=True, help="Path to the MMHS150K_GT.json file")
    parser.add_argument("--llm-output", type=str, required=True, help="Path to the output .jsonl.gz file")
    parser.add_argument("--train-test-split", type=str, help="Path to the train-test-split.csv file")
    main(parser)


