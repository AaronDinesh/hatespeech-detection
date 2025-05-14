import gzip
import json
import pydantic
import argparse
import typing
import time
import random
import os
from openai import OpenAI, RateLimitError, APIError, APITimeoutError
from dotenv import load_dotenv
from tqdm import tqdm
from enum import Enum

class labels(str, Enum):
    NotHate = 'NotHate'
    Racist = 'Racist'
    Sexist = 'Sexist'
    Homophobe = 'Homophobe'
    Religion = 'Religion'
    OtherHate = 'OtherHate'

class postDescription(pydantic.BaseModel):
    input_labels: list[labels]


Allowed_labels = typing.Literal[
    "NotHate", "Racist", "Sexist", "Homophobe", "Religion", "OtherHate"
]

class Response_schema(pydantic.BaseModel):
    input_labels: pydantic.conlist(Allowed_labels, min_length=3, max_length=3)

class Output_schema(pydantic.BaseModel):
    id: str
    response: Response_schema



def send_prompt(client: OpenAI, message: tuple[str, dict], model:str, response_format: dict[str, any], max_retries: int = 5):
    """
    Sends a request to the OpenAI endpoint with exponential backoff for retries
    """

    backoff = 1
    num_retries = 0
    bad_json_limit = max_retries * 2
    bad_json_retries = 0
    while num_retries < max_retries:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[message[1]],
                extra_body={"guided_json": response_format}
            )
        except (RateLimitError, APIError, APITimeoutError) as e:
            print("Error:", e)
            time.sleep(backoff + random.uniform(0, 1))
            backoff *= 2
            num_retries += 1
            continue
        try:
            parsed = postDescription.model_validate_json(response.choices[0].message.content)
            return (message[0], parsed.model_dump())
        except pydantic.ValidationError:
            print("Bad JSON response. Retrying...")
            bad_json_retries += 1

            if bad_json_retries > bad_json_limit:
                break

            continue
        
    print(f"Prompt with post id {message[0]} failed") 
    return (message[0], None)



def json_generator(filepath: str):
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                yield json.loads(line)
            except Exception as e:
                print(f"JSON decode error: {e}")
                raise StopIteration

def main(args):
    response_format = postDescription.model_json_schema()
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
        
        load_dotenv()
        RCP_AIaaS_KEY = os.getenv("OPENAI_API_KEY")

        client = OpenAI(api_key=RCP_AIaaS_KEY, base_url="https://inference-dev.rcp.epfl.ch/v1")
        
        
        if not args.file_path:
            print("Error: --file-path is required when using --fix-errors")
            return
        
        if not os.path.exists(args.file_path):
            print("Error: --file-path does not exist")
            return
        

        unique_ids = set()
        for line in tqdm(json_generator(args.file_path), desc="Building Unique ID Set", total=total_length, unit=" prompts"):
            unique_ids.add(line['id'])

        with gzip.open(args.fixed_file_path, "at") as g:
            for line in tqdm(json_generator(args.file_path), desc="Fixing Errors", total=total_length, unit=" prompts"):
                try:
                    Output_schema.model_validate(line)
                    
                    if line['id'] in unique_ids:

                        if line['response']['input_labels'] is None:
                            raise pydantic.ValidationError
                    
                        g.write(json.dumps(line) + "\n")

                        #Only after we write it to the file we want to remove that id from the set. 
                        unique_ids.remove(line['id'])


                except pydantic.ValidationError as e:
                    #We assumming that if the model outputs less than 3 labels, it means that it wants to duplicate the
                    #last label. Meaning that if we have a label like ['NotHate', 'Racist'], it means that the model
                    #also thought the last judge would say racist and therefore compressed the returned labels. In the
                    #case of ['NotHate'] it means that all 3 judges would have said 'NotHate'. In the case where the
                    #model output more then 3 labels, we just truncate to the first 3
                    
                    ##print(f"ID {line['id']} failed because {e}\n")
                    prev_length = len(line['response']['input_labels'])
                    if prev_length < 3 and prev_length != 0:
                        if line['id'] in unique_ids:
                            for _ in range(prev_length, 3):
                                line['response']['input_labels'].append(line['response']['input_labels'][-1]) 
                            g.write(json.dumps(line) + "\n")

                            #Only after we write it to the file we want to remove that id from the set. 
                            unique_ids.remove(line['id'])
        
                    elif prev_length == 0:
                        if line['id'] in unique_ids:
                            # Here we need to dispatch to the model again
                            for prompt in json_generator(args.prompts_file):
                                if prompt['id'] == line['id']:
                                    result = send_prompt(client, (prompt['id'], prompt['prompt']), args.model, response_format)
                                    g.write(json.dumps({"id": result[0], "response": result[1]}) + "\n")
                                    unique_ids.remove(line['id'])
                        pass
                    else:
                        if line['id'] in unique_ids:
                            line['response']['input_labels'] = line['response']['input_labels'][:3]
                            g.write(json.dumps(line) + "\n")
                            unique_ids.remove(line['id'])


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--file-path", type=str, required=True, help="Path to the output .jsonl.gz file")
    argparser.add_argument("--fix-errors", action='store_true', help="Attempts to fix errors if any are found")
    argparser.add_argument("--fixed-file-path", type=str, help="Path to the fixed output .jsonl.gz file")
    argparser.add_argument("--prompts-file", type=str, help="Path to the prompts .jsonl.gz file")
    argparser.add_argument("--model", type=str, help="Model to use in case no labels are found")

    main(argparser.parse_args())