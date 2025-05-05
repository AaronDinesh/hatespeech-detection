from dotenv import load_dotenv
import os
import requests
import httpx
from openai import AsyncOpenAI, RateLimitError, APIError, Timeout
import argparse
import asyncio
from enum import Enum
from pydantic import BaseModel, ValidationError
import json
import gzip
from tqdm.asyncio import tqdm
from typing import AsyncGenerator
import random
import time

class labels(str, Enum):
    NotHate = 'NotHate'
    Racist = 'Racist'
    Sexist = 'Sexist'
    Homophobe = 'Homophobe'
    Religion = 'Religion'
    OtherHate = 'OtherHate'

class postDescription(BaseModel):
    input_labels: list[labels]

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
        except (RateLimitError, APIError, Timeout) as e:
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
        
        async with result_file_lock:
            with gzip.open(os.path.join(OUTPUT_DIR, 'results.jsonl.gz'), 'at', encoding='utf-8') as f:
                f.write(json.dumps({"id": response[0], "response": response[1]}) + "\n")
        
        pbar.update(1)
        queue.task_done()



async def process_prompts(client: AsyncOpenAI, input_file_path: str, output_dir: str, model: str, response_format: dict[str, any], max_concurrent_tasks: int, max_retries: int = 5):
    """Process multiple requests with controlled concurrency."""

    queue = asyncio.Queue(maxsize=max_concurrent_tasks * 2)
    file_lock = asyncio.Lock()
    prog_bar = tqdm(desc="Sending prompts...", unit=" prompt")

    workers = [
        asyncio.create_task(worker_func(queue, model, client, response_format, prog_bar, file_lock, output_dir, max_retries))
        for _ in range(max_concurrent_tasks)
    ]


    temp_counter = 0
    temp_max = 5
    print(f"\nTesting with {temp_max} prompts\n")
    async for prompt in generate_prompts_from_file(input_file_path):
        await queue.put(prompt)
        temp_counter+=1
        if temp_counter > temp_max:
            break


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

    async_client = AsyncOpenAI(api_key=RCP_AIaaS_KEY, base_url=base_url)

    gzip.open(os.path.join(output_dir, 'results.jsonl.gz'), "wt").close()

    response_format = postDescription.model_json_schema()

    asyncio.run(process_prompts(async_client, input_file, output_dir, model, response_format, 10, 10))

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
    parser.add_argument("--input_file", type=str, required=True, help="Input .jsonl.gz prompts file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory path")
    parser.add_argument("--model", type=str, choices=model_choices, required=True, help="The LLM to use")
    asyncio.run(main(base_url, RCP_AIaaS_KEY, parser))