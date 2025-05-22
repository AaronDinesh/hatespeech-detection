"""
A script that uses multiprocessing to convert the raw data into prompts that can be used with the RCP AIaaS LLM
inference endpoint or any OpenAI compatible endpoint.
"""

import os
import base64
import requests
import multiprocessing
import pandas as pd
from dotenv import load_dotenv
import argparse
import sys
import gzip
import json
import argcomplete
from tqdm import tqdm
import time
from itertools import islice


def send_ntfy(topic: str, msg: str, headers: dict = {}):
    try:
        requests.post(f"https://ntfy.sh/{topic}", data=msg, headers=headers, timeout=10)
    except requests.exceptions.RequestException as e:
        print(f"ntfy notification failed: {e}", file=sys.stderr)


def encode_image(image_path: str, use_ntfy: bool = False, topic: str = ""):
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_string
    except FileNotFoundError:
        msg = f"File {image_path} not found."
        if use_ntfy:
            send_ntfy(topic, msg, {"Tags": "exclamation"})
        print(msg)
        return None
    except Exception as e:
        msg = f"Error during reading {image_path}: {e}"
        if use_ntfy:
            send_ntfy(topic, msg, {"Tags": "exclamation"})
        print(msg)
        return None


def construct_prompts(LLM_PROMPT: str, image_data: str, image_text: str, tweet_text: str):
    message_template = {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_data}",
                    "detail": "high",
                },
            },
            {
                "type": "text",
                "text": (
                    f"{LLM_PROMPT} "
                    f"Image Text: {image_text.strip() if image_text else ''}. "
                    f"Tweet text: {tweet_text.strip() if tweet_text else ''}"
                )
            },
        ],
    }
    return message_template


def process_single_item(idx: str, row: pd.Series, base_data_dir: str, use_ntfy: bool, topic: str, LLM_PROMPT_text: str):
    image_path = os.path.join(base_data_dir, f"img_resized/{idx}.jpg")
    image_txt_path = os.path.join(base_data_dir, f"img_txt/{idx}.json")

    image_data = encode_image(image_path, use_ntfy, topic)
    if image_data is None:
        return None

    image_text_content = ""
    try:
        with open(image_txt_path, "r") as f:
            image_text_content = json.load(f)["img_text"]
    except FileNotFoundError:
        msg = f"Image text file {image_txt_path} not found for {idx}."
        if use_ntfy:
            send_ntfy(topic, msg, {"Tags": "exclamation"})
    except json.JSONDecodeError as e:
        msg = f"Error decoding JSON from {image_txt_path} for {idx}: {e}"
        print(msg)
        if use_ntfy:
            send_ntfy(topic, msg, {"Tags": "exclamation"})
    except Exception as e:
        msg = f"Error reading image text {image_txt_path} for {idx}: {e}"
        print(msg)
        if use_ntfy:
            send_ntfy(topic, msg, {"Tags": "exclamation"})

    tweet_txt = row.get('tweet_text', "")

    prompt = construct_prompts(LLM_PROMPT_text, image_data, image_text_content, tweet_txt)
    return {"id": idx, "prompt": prompt}


# Worker process function
def worker_process_loop(task_queue: multiprocessing.Queue,
                        result_queue: multiprocessing.Queue,
                        base_data_dir: str,
                        use_ntfy_flag: bool,
                        current_topic: str,
                        LLM_PROMPT_text: str,
                        worker_id: int):
    """
    Worker process: fetches tasks from task_queue, processes them, and puts results on result_queue.
    """
    print(f"--- Worker {worker_id} STARTING ---", flush=True)
    sys.stdout.flush()
    
    while True:
        try:
            # Get a task (idx, row_data) from the queue with timeout
            # A timeout prevents hanging if the queue is empty
            try:
                task = task_queue.get(timeout=5)  # 5 second timeout
            except multiprocessing.queues.Empty:
                continue  # Try again if queue is empty
                
            # Check for exit signal
            if task is None:
                print(f"Worker {worker_id} received exit signal", flush=True)
                break
                
            idx, row_data = task
            # Process item and handle result
            item_result = process_single_item(idx, row_data, base_data_dir, use_ntfy_flag, current_topic, LLM_PROMPT_text)
            
            # Mark task as done
            task_queue.task_done()
            
            # Put result in queue if valid
            if item_result:
                # Use a retry mechanism for putting results
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        result_queue.put(item_result, timeout=5)
                        break
                    except multiprocessing.queues.Full:
                        if retry < max_retries - 1:
                            time.sleep(0.5)  # Short sleep before retry
                        else:
                            print(f"Worker {worker_id}: Result queue full after retries for item {idx}", flush=True)
                            
        except Exception as e:
            print(f"Error in worker {worker_id}: {str(e)}", file=sys.stderr, flush=True)
            if 'task_queue' in locals() and 'task' in locals():
                try:
                    task_queue.task_done()  # Mark task as done even on error
                except ValueError:  # In case task_done was already called
                    pass
            
    print(f"Worker {worker_id} exiting", flush=True)


def process_results_from_queue(result_queue, output_path, total_items_to_process):
    """Process results from the result queue and write to file."""
    items_processed_count = 0
    
    with gzip.open(output_path, 'at', encoding='utf-8') as fout, \
         tqdm(total=total_items_to_process, desc="Writing results", position=2, leave=False, unit=" items") as pbar:
        
        while items_processed_count < total_items_to_process:
            try:
                # Use shorter timeout to be more responsive
                result = result_queue.get(timeout=2)
                
                if result:
                    fout.write(json.dumps(result) + "\n")
                    fout.flush()  # Flush after each write to ensure data is written
                    items_processed_count += 1
                    pbar.update(1)
                
                # Mark task as done
                result_queue.task_done()
                
            except multiprocessing.queues.Empty:
                # Check if we should exit (coordinator will handle this)
                continue
            except Exception as e:
                print(f"Error processing result: {str(e)}", file=sys.stderr, flush=True)
                try:
                    result_queue.task_done()  # Mark as done even on error
                except ValueError:
                    pass
    
    return items_processed_count


def batch_iterable(iterable, batch_size):
    """Yield batches from an iterable."""
    iterator = iter(iterable)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch


def main(LLM_PROMPT: str, cli_parser: argparse.ArgumentParser):
    argcomplete.autocomplete(cli_parser)
    args = cli_parser.parse_args()
    use_ntfy_main = args.use_ntfy
    topic_main = args.topic
    base_data_dir_main = args.data_dir
    output_dir_main = args.output_dir
    batch_size = args.batch_size

    # Determine optimal number of worker processes (using fewer than CPU count)
    num_worker_processes = min(max(os.cpu_count() - 1, 2), args.num_workers)
    print(f"Using {num_worker_processes} worker processes.")

    # Create queues with appropriate sizes
    task_queue = multiprocessing.JoinableQueue(maxsize=batch_size * 2)
    result_queue = multiprocessing.JoinableQueue(maxsize=batch_size * 2)

    # --- Master Process (Process 0) ---
    print("Master process started. Reading data...")
    try:
        # Read the data in chunks to reduce memory usage
        mmhs_dataframe = pd.read_json(
            os.path.join(base_data_dir_main, 'MMHS150K_GT.json'),
            lines=False, orient='index', convert_dates=False,
            convert_axes=False, dtype=str
        )
    except FileNotFoundError:
        print(f"Error: MMHS150K_GT.json not found at {os.path.join(base_data_dir_main, 'MMHS150K_GT.json')}")
        return
    except Exception as e:
        print(f"Error reading MMHS150K_GT.json: {e}")
        return

    if mmhs_dataframe.empty:
        print("Input DataFrame is empty. Exiting.")
        return

    total_items_to_process = len(mmhs_dataframe)
    if total_items_to_process == 0:
        print("No items to process in the DataFrame. Exiting.")
        return

    os.makedirs(output_dir_main, exist_ok=True)
    with open(os.path.join(output_dir_main, 'llm_prompt.txt'), 'w') as f:
        f.write(LLM_PROMPT)

    output_path = os.path.join(output_dir_main, 'prompts.jsonl.gz')

    # Start worker processes
    workers = []
    for i in range(num_worker_processes):
        worker = multiprocessing.Process(
            target=worker_process_loop,
            args=(task_queue, result_queue, base_data_dir_main, use_ntfy_main, topic_main, LLM_PROMPT, i)
        )
        worker.daemon = True  # Set as daemon to terminate if main process exits
        workers.append(worker)
        worker.start()
    print(f"{num_worker_processes} worker processes launched.")

    # Start result processor in a separate process
    result_processor = multiprocessing.Process(
        target=process_results_from_queue,
        args=(result_queue, output_path, total_items_to_process)
    )
    result_processor.daemon = True
    result_processor.start()

    # Process the dataframe in batches to reduce memory pressure
    total_batches = (total_items_to_process + batch_size - 1) // batch_size
    print(f"Processing {total_items_to_process} items in {total_batches} batches...")
    
    try:
        # Process in batches to avoid overloading the queue
        batched_data = batch_iterable(mmhs_dataframe.iterrows(), batch_size)
        
        for batch_idx, batch in tqdm(enumerate(batched_data), total=total_batches, desc="Processing batches", position=1, leave=False, unit=" batches"):
            #print(f"Dispatching batch {batch_idx+1}/{total_batches}", flush=True)
            
            # Put batch items in task queue
            for idx, row in batch:
                # Use timeout and retry to prevent hanging on a full queue
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        task_queue.put((idx, row), timeout=10)
                        break
                    except multiprocessing.queues.Full:
                        if retry < max_retries - 1:
                            print(f"Task queue full, retrying ({retry+1}/{max_retries})...", flush=True)
                            time.sleep(1)
                        else:
                            print(f"Failed to add task for {idx} after {max_retries} retries", flush=True)
            
            # Wait for batch to be processed before adding more
            task_queue.join()
            
            # Add a small delay between batches to allow results to be processed
            time.sleep(0.1)
            
            # Periodically check if workers are alive
            if batch_idx % 10 == 0:
                for i, worker in enumerate(workers):
                    if not worker.is_alive():
                        print(f"Worker {i} died unexpectedly. Restarting...", flush=True)
                        worker = multiprocessing.Process(
                            target=worker_process_loop,
                            args=(task_queue, result_queue, base_data_dir_main, use_ntfy_main, topic_main, LLM_PROMPT, i)
                        )
                        worker.daemon = True
                        workers[i] = worker
                        worker.start()

    except KeyboardInterrupt:
        print("\nInterrupted by user. Cleaning up...", flush=True)
    except Exception as e:
        print(f"Error during batch processing: {str(e)}", file=sys.stderr, flush=True)
    finally:
        # Send sentinel values (None) to tell workers to stop
        print("Sending termination signals to workers...", flush=True)
        for _ in range(num_worker_processes):
            task_queue.put(None)
        
        # Wait for workers to finish (with timeout)
        for worker in workers:
            worker.join(timeout=10)
            if worker.is_alive():
                print(f"Worker {worker.pid} did not terminate gracefully, forcing termination.", flush=True)
                worker.terminate()
        
        # Wait for result processor to finish
        result_processor.join(timeout=10)
        if result_processor.is_alive():
            print("Result processor did not terminate gracefully, forcing termination.", flush=True)
            result_processor.terminate()
        
        print(f"Processing complete. Output written to {output_path}", flush=True)


if __name__ == "__main__":
    load_dotenv()
    CURRENT_NTFY_TOPIC = os.getenv("NTFY_TOPIC", "default_app_topic")

    DEFAULT_MMHS_DATA_DIR = "../data/MMHS150K"
    DEFAULT_OUTPUT_DIR = "./processed_prompts/iteration_3/"

    cli_parser = argparse.ArgumentParser(prog='llm-prompt-generator', 
                                         description='Generate prompts for LLM inference from MMHS150K dataset')
    cli_parser.add_argument('--use-ntfy', action='store_true', 
                           help='Use NTFY service for error notifications')
    cli_parser.add_argument('--topic', type=str, default=CURRENT_NTFY_TOPIC, 
                           help='NTFY topic for notifications')
    cli_parser.add_argument('--data-dir', type=str, default=DEFAULT_MMHS_DATA_DIR, 
                           help='Directory of MMHS150K dataset')
    cli_parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR, 
                           help='Directory to save processed prompts')
    cli_parser.add_argument('--batch-size', type=int, default=50, 
                           help='Number of items to process in each batch')
    cli_parser.add_argument('--num-workers', type=int, default=os.cpu_count(), 
                           help='Number of worker processes')

    LLM_PROMPT = "You are a language model trained to detect hate speech in memes posted to Twitter. You will be provided with the image, any text that could be found in the image (Preceded with 'Image Text:') as well as any text in the tweet body (Preceded with 'Tweet text:'). An Image will always be provided, however there may not be any image text or tweet text. In this case those fields will be empty e.g 'Image Text: . Tweet text:'. You must respond with a number from '0', '1', '2', '3'. '0' means there is no hate. '3' is the maximum hatefulness score you can give. You must only respond with integers in the range 0 to 3. Do not give a number lower than 0 or a number larger than 3."

    main(LLM_PROMPT, cli_parser)
