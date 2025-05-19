import json
import gzip
import argparse


def json_generator(filepath: str):
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                yield json.loads(line)
            except Exception as e:
                print(f"JSON decode error: {e}")
                raise StopIteration


def main():
    hate = 0
    not_hate = 0
    for line in json_generator("./llm_responses/MMHS150K/iteration_2/Llama-4-Scout-17B-16E-Instruct/results_FIXED.jsonl.gz"):
        if "HateSpeech" in line["response"]["input_labels"]:
            hate += 1
        else:
            not_hate += 1

    print("Hate:", hate)
    print("Not Hate:", not_hate)



if __name__ == "__main__":
    main()