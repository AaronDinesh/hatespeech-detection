import gzip
import json
import pydantic
import pydantic
import argparse
import typing
import os

from tqdm import tqdm

Allowed_labels = typing.Literal[
    "NotHate", "Racist", "Sexist", "Homophobe", "Religion", "OtherHate"
]

class Response_schema(pydantic.BaseModel):
    input_labels: pydantic.conlist(Allowed_labels, min_length=3, max_length=3)

class Output_schema(pydantic.BaseModel):
    id: str
    response: Response_schema

def json_generator(filepath: str):
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                yield json.loads(line)
            except Exception as e:
                print(f"JSON decode error: {e}")
                raise StopIteration

def main(args):

    total_length = sum(1 for _ in tqdm(json_generator(args.file_path), desc="Enumerating prompts", unit=" prompts"))
    print(f"Found {total_length} prompts.")

    if not args.fix_errors:
        results = {
            "total": 0,
            "valid": 0,
            "invalid": 0,
            "errors": []
        }

        for line in tqdm(json_generator(args.file_path), desc="Parsing file", total=total_length, unit=" prompts"):
            results["total"] += 1
            try:
                Output_schema.model_validate(line)
                results["valid"] += 1
            except pydantic.ValidationError as e:
                print(f"ID {line['id']} failed because {e}")
                results["invalid"] += 1
                results["errors"].append(f"ID {line["id"]} failed because {e}")
    
        print(f"Total: {results["total"]}")
        print(f"Valid: {results['valid']}")
        print(f"Invalid: {results['invalid']}")
    else:
        if not args.file_path:
            print("Error: --file-path is required when using --fix-errors")
            return
        
        if not os.path.exists(args.file_path):
            print("Error: --file-path does not exist")
            return


        with gzip.open(args.fixed_file_path, "at") as g:
            for line in tqdm(json_generator(args.file_path), desc="Fixing Errors", total=total_length, unit=" prompts"):
                try:
                    Output_schema.model_validate(line)
                    g.write(json.dumps(line) + "\n")
                except pydantic.ValidationError as e:
                    #We assumming that if the model outputs less than 3 labels, it means that it wants to duplicate the
                    #last label. Meaning that if we have a label like ['NotHate', 'Racist'], it means that the model
                    #also thought the last judge would say racist and therefore compressed the returned labels. In the
                    #case of ['NotHate'] it means that all 3 judges would have said 'NotHate'. In the case where the
                    #model output more then 3 labels, we just truncate to the first 3
                    
                    ##print(f"ID {line['id']} failed because {e}\n")
                    prev_length = len(line['response']['input_labels'])
                    if prev_length < 3:
                        for _ in range(prev_length, 3):
                            line['response']['input_labels'].append(line['response']['input_labels'][-1]) 
                        g.write(json.dumps(line) + "\n")
                    else:
                        line['response']['input_labels'] = line['response']['input_labels'][:3]
                        g.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--file-path", type=str, required=True, help="Path to the output .jsonl.gz file")
    argparser.add_argument("--fix-errors", action='store_true', help="Attempts to fix errors if any are found")
    argparser.add_argument("--fixed-file-path", type=str, help="Path to the fixed output .jsonl.gz file")

    main(argparser.parse_args())