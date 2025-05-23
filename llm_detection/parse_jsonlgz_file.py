"""
A script that can print the contents of the jsonl.gz file.
"""

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
            

def main(args):
    filepath = args.file_path
    counter = 0
    for line in json_generator(filepath):
        print(line)
        if args.limit != -1:
            counter += 1
            if counter == args.limit:
                break


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--file-path", type=str, required=True, help="Path to the .jsonl.gz file to validate")
    argparser.add_argument("--limit", type=int, required=True, default=-1, help="Limit the number of lines to process. Use -1 for no limit") 
    main(argparser.parse_args())