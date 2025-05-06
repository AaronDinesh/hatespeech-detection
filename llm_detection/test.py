import json
import gzip


def json_generator(filepath: str):
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                yield json.loads(line)
            except Exception as e:
                print(f"JSON decode error: {e}")
                raise StopIteration
            

def main():
    filepath = './llm_responses/MMHS150K/Llama-4-Scout-17B-16E-Instruct/results.jsonl.gz'
    for line in json_generator(filepath):
        print(line)


if __name__ == "__main__":
    main()