"""
A script that calls to the AIaaS LLM inference endpoint using a prompts.jsonl.gz file. The prompts should be encoded in
the OpenAI format. The output from the llm is a single number from 0-3 indicating the hatefulness of the prompt
"""

from dotenv import load_dotenv
import os
import requests
import httpx
from openai import AsyncOpenAI, RateLimitError, APIError, APITimeoutError
import argparse
import asyncio
from enum import Enum
from pydantic import BaseModel, ValidationError, conlist
import json
import gzip
from tqdm.asyncio import tqdm
from typing import AsyncGenerator
import random
import time
import typing

############## PROMPT SCHEMAS ##################
# class labels(str, Enum):
#     NotHate = 'NotHate'
#     Racist = 'Racist'
#     Sexist = 'Sexist'
#     Homophobe = 'Homophobe'
#     Religion = 'Religion'
#     OtherHate = 'OtherHate'

class postDescription(BaseModel):
    input_labels: int


################################################

def get_model_choices(API_KEY: str, base_url: str = "https://inference-dev.rcp.epfl.ch/v1"):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}',
    }
    response = requests.get(f'{base_url}/models', headers=headers)
    y = response.json()
    return [x['id'] for x in y['data']]

async def send_ntfy(topic: str, msg: str, headers: dict = {}):
    async with httpx.AsyncClient() as client:
        await client.post(f"https://ntfy.sh/{topic}", content=msg, headers=headers)    

async def generate_prompts_from_file(file_path: str) -> AsyncGenerator[tuple[str, dict], None]:
    """
    Returns a generator that yields tuples of (index, prompt) from a jsonl.gz file
    """
    running_loop = asyncio.get_running_loop()
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            json_obj = await running_loop.run_in_executor(None, json.loads, line)
            yield (json_obj['id'], json_obj['prompt'])


async def send_prompt(client: AsyncOpenAI, message: tuple[str, dict], model:str, response_format: dict[str, any], max_retries: int = 5):
    """
    Sends a request to the OpenAI endpoint with exponential backoff for retries
    """

    backoff = 1
    num_retries = 0
    bad_json_limit = max_retries * 2
    bad_json_retries = 0
    while num_retries < max_retries:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[message[1]],
                extra_body={"guided_json": response_format}
            )
        except (RateLimitError, APIError, APITimeoutError) as e:
            print("Error:", e)
            await asyncio.sleep(backoff + random.uniform(0, 1))
            backoff *= 2
            num_retries += 1
            continue


        try:
            parsed = postDescription.model_validate_json(response.choices[0].message.content)
            return (message[0], parsed.model_dump())
        except ValidationError:
            print("Bad JSON response. Retrying...")
            bad_json_retries += 1

            if bad_json_retries > bad_json_limit:
                break

            continue
        
    print(f"Prompt with post id {message[0]} failed") 
    return (message[0], None)


async def worker_func(queue: asyncio.Queue, model: str, client: AsyncOpenAI, response_format: dict[str, any], pbar: tqdm, result_file_lock: asyncio.Lock, OUTPUT_DIR: str, max_retries: int = 5):
    while True:
        message = await queue.get()
        if message is None:
            queue.task_done()
            break
        
        response = await send_prompt(client, message, model, response_format, max_retries)
        
        if response[1] is not None:
            async with result_file_lock:
                with gzip.open(os.path.join(OUTPUT_DIR, 'results.jsonl.gz'), 'at', encoding='utf-8') as f:
                    f.write(json.dumps({"id": response[0], "response": response[1]}) + "\n")
                    f.flush()
        
        pbar.update(1)
        queue.task_done()



async def process_prompts(client: AsyncOpenAI, input_file_path: str, output_dir: str, model: str, response_format: dict[str, any], max_concurrent_tasks: int, max_retries: int = 5, completed_ids: set = None):
    """Process multiple requests with controlled concurrency."""

    queue = asyncio.Queue(maxsize=max_concurrent_tasks * 2)
    file_lock = asyncio.Lock()
    prog_bar = tqdm(desc="Sending prompts...", unit=" prompts")

    workers = [
        asyncio.create_task(worker_func(queue, model, client, response_format, prog_bar, file_lock, output_dir, max_retries))
        for _ in range(max_concurrent_tasks)
    ]
    
    with gzip.open(input_file_path, 'rt', encoding='utf-8') as f:
        prog_bar.total = sum(1 for _ in tqdm(f, desc="Enumerating prompts", unit=" prompts"))
     
    print(f"Found {prog_bar.total} prompts.")

    async for prompt in generate_prompts_from_file(input_file_path):
        if completed_ids is not None and prompt[0] in completed_ids:
            print(f"Prompt with post id {prompt[0]} already completed. Skipping...")
            prog_bar.update(1)
            continue

        await queue.put(prompt)


    for _ in workers:
        await queue.put(None)
    
    await queue.join()

    for worker in workers:
        await worker

    prog_bar.close()



def main(base_url: str, RCP_AIaaS_KEY: str, parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()    
    input_file = args.input_file
    output_dir = args.output_dir
    model = args.model
    max_concurrent_tasks = args.max_concurrent_tasks
    num_retries = args.num_retries
    restart = args.restart
    results_file = args.results_file

    async_client = AsyncOpenAI(api_key=RCP_AIaaS_KEY, base_url=base_url)

    if not restart:
        with gzip.open(os.path.join(output_dir, 'results.jsonl.gz'), "wt") as f:
            pass

    response_format = postDescription.model_json_schema()
    
    completed_ids = None
    if restart:
        if not results_file:
            print("Error: --results-file file path is required when using --restart")
            return

        if not os.path.exists(results_file):
            print(f"Error: {results_file} does not exist")
            return

        print(f"Restarting from {results_file}")
        completed_ids = set()
        with gzip.open(results_file, 'rt', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading completed ids...", leave=False, unit=" prompts"):
                try:
                    json_line = json.loads(line)
                    if json_line['response'] is not None:
                        completed_ids.add(json_line['id'])
                except Exception as e:
                    print(f"JSON decode error: {e}")
                    raise Exception
        
    
    asyncio.run(process_prompts(async_client, input_file, output_dir, model, response_format, max_concurrent_tasks, num_retries, completed_ids))


if __name__ == "__main__":
    print(f"Script Start time: {time.ctime()}")
    load_dotenv()
    RCP_AIaaS_KEY = os.getenv("OPENAI_API_KEY")

    base_url = "https://inference-dev.rcp.epfl.ch/v1"
    
    try:
        model_choices = get_model_choices(RCP_AIaaS_KEY, base_url)
    except Exception:
        model_choices = []
    
    parser = argparse.ArgumentParser(prog='llm-inference', description='LLM inference on Images for EE-559 Project')
    parser.add_argument("--input-file", type=str, required=True, help="Input .jsonl.gz prompts file")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory path")
    parser.add_argument("--model", type=str, choices=model_choices, required=True, help="The LLM to use")
    parser.add_argument("--max-concurrent-tasks", type=int, default=7, help="Maximum number of concurrent tasks when sending requests to the API. The script will send and await these replies. Do not set this number too large as it can fill up the KV cache of the model")
    parser.add_argument("--num-retries", type=int, default=10, help="Number of retries when running into API errors. Implements exponential backoff with base 2.")
    parser.add_argument("--restart", action='store_true', help="Restart from a previous run")
    parser.add_argument("--results-file", type=str, help="Path to the previous run's results file")
    main(base_url, RCP_AIaaS_KEY, parser)