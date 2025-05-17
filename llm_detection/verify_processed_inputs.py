"""
A script that verifies the output of process_inputs_multiprocessing.py. Prints any errors.
"""

import json
import gzip
import argparse
import base64
from tqdm import tqdm
from typing import Tuple, Optional
import pprint

def is_valid_base64_image_url(url: str) -> bool:
    if not url.startswith("data:image/jpeg;base64,"):
        return False
    b64_part = url.split("base64,", 1)[-1]
    try:
        base64.b64decode(b64_part, validate=True)
        return True
    except Exception:
        return False

def is_valid_prompt(entry: dict) -> Tuple[bool, Optional[str]]:
    if not isinstance(entry, dict):
        return False, "Not a dictionary"
    if "id" not in entry or "prompt" not in entry:
        return False, "Missing 'id' or 'prompt' keys"

    prompt = entry["prompt"]
    if not isinstance(prompt, dict):
        return False, "'prompt' is not a dictionary"
    if prompt.get("role") != "user":
        return False, "role is not 'user'"

    content = prompt.get("content")
    if not isinstance(content, list):
        return False, "'content' is not a list"

    has_text = False
    has_valid_image = False

    for item in content:
        if item.get("type") == "text":
            has_text = True
        elif item.get("type") == "image_url":
            url = item.get("image_url", {}).get("url")
            if isinstance(url, str) and is_valid_base64_image_url(url):
                has_valid_image = True

    if not has_valid_image:
        return False, "Missing or invalid base64 input_image"
    if not has_text:
        return False, "Missing text in content"

    return True, None

def json_generator(filepath: str, results: dict):
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            results["total"] += 1
            try:
                yield json.loads(line)
            except Exception as e:
                results["invalid"] += 1
                results["errors"].append((line_num, f"JSON decode error: {e}"))

def main(parser: argparse.ArgumentParser):
    global results
    results = {
        "total": 0,
        "valid": 0,
        "invalid": 0,
        "errors": []
    }

    args = parser.parse_args()
    file_path = args.file_path

    for line_num, entry in tqdm(enumerate(json_generator(file_path, results)), desc="Validating prompts"):
        if isinstance(entry, dict):  # only check structure if JSON load was OK
            valid, error = is_valid_prompt(entry)
            if valid:
                results["valid"] += 1
            else:
                results["invalid"] += 1
                results["errors"].append((line_num, error))

    pprint.pprint(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True, help="Path to the .jsonl.gz file to validate")
    main(parser)
