import pandas as pd
import json
import ast

"""
Finds all the unique labels in the dataset.
"""

df = pd.read_json("../data/MMHS150K/MMHS150K_GT.json", lines=False, orient='index', convert_dates=False, convert_axes=False, dtype=str)


labels = [ast.literal_eval(x) for x in df["labels_str"]]
print("Ground Truth Labels")
print(set().union(*labels))