import pandas as pd
from dotenv import load_dotenv
import os
import aiofiles
import requests
import httpx
from openai import AsyncOpenAI, RateLimitError, APIError, Timeout
import argparse
import asyncio
from enum import Enum
from pydantic import BaseModel
import json
import gzip
from tqdm.asyncio import tqdm
from typing import AsyncGenerator, Tuple

class labels(Enum):
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

async def send_request(message: tuple[str, dict], model: str, sem: asyncio.Semaphore, client: AsyncOpenAI, response_format: dict[str, any]):
    """Send a single request to xAI with semaphore control."""
    # The 'async with sem' ensures only a limited number of requests run at once
    async with sem:
        res =  await client.chat.completions.create(
            model=model,
            messages=[message[1]],
            response_format=response_format
        )
        return (message[0], res)

async def process_requests(message_templates: list[dict], model: str, client: AsyncOpenAI, response_format: dict[str, any], max_async_requests: int):
    """Process multiple requests with controlled concurrency."""
    # Create a semaphore that limits how many requests can run at the same time
    # Think of it like having only 2 "passes" to make requests simultaneously
    sem = asyncio.Semaphore(max_async_requests)
    tasks = [send_request(message, model, sem, client, response_format) for message  in message_templates]
    return await asyncio.gather(*tasks)


async def main(base_url: str, RCP_AIaaS_KEY: str, parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()    

    async_client = AsyncOpenAI(api_key=RCP_AIaaS_KEY, base_url=base_url)
    
    #TODO: FINISH THE ASYNC IMPLEMENTATION
    



if __name__ == "__main__":
    global NTFY_TOPIC
    load_dotenv()
    RCP_AIaaS_KEY = os.getenv("OPENAI_API_KEY")
    NTFY_TOPIC = os.getenv("NTFY_TOPIC")
    OUTPUT_DIR = None

    base_url = "https://inference-dev.rcp.epfl.ch/v1"
    
    try:
        model_choices = get_model_choices(RCP_AIaaS_KEY, base_url)
    except Exception:
        model_choices = []
    
    parser = argparse.ArgumentParser(prog='llm-inference', description='LLM inference on Images for EE-559 Project')
    
    asyncio.run(main(base_url, RCP_AIaaS_KEY, LLM_PROMPT, parser))