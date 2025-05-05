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
    filepath = './processed_prompts/iteration_1/prompts.jsonl.gz'
    for line in json_generator(filepath):
        tweet_idx = line['id']
        print(line['prompt'])
        break


if __name__ == "__main__":
    main()