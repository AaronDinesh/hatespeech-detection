import os
import base64
import requests
from mpi4py import MPI
import pandas as pd
from dotenv import load_dotenv
import argparse
import numpy as np
from tqdm import tqdm
import json
import sys
import gzip

# put this somewhere but before calling the asserts
sys_excepthook = sys.excepthook
def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)
#This will kill all processes when an assertion error is triggered
sys.excepthook = mpi_excepthook

def send_ntfy(topic: str, msg: str, headers: dict = {}):
    requests.post(f"https://ntfy.sh/{topic}", data=msg, headers=headers)    

def encode_image(image_path: str, use_ntfy: bool = False, NTFY_TOPIC: str = ""):
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_string
    except FileNotFoundError:
        if use_ntfy:
            send_ntfy(NTFY_TOPIC, f"File {image_path} not found.", {"Tags": "exclamation"})
        print(f"File {image_path} not found.")
        return None
    except Exception as e:
        if use_ntfy: 
            send_ntfy(NTFY_TOPIC, f"Error during reading {image_path}: {e}", {"Tags": "exclamation"})
        print(f" Error during reading {image_path}: {e}")
        return None

def construct_prompts(LLM_PROMPT: str, image_data: base64, image_text: str, tweet_text: str):
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


def process_df_row(idx: str, row: pd.Series, base_data_dir: str, use_ntfy: bool, topic: str):
    image_path = os.path.join(base_data_dir, f"img_resized/{idx}.jpg")
    image_txt_path = os.path.join(base_data_dir, f"img_txt/{idx}.json")
    image_data = encode_image(image_path, use_ntfy, topic)
    
    try:
        with open(image_txt_path, "r") as f:
            image_text = json.load(f)["img_text"]
    except Exception:
        if use_ntfy:
            send_ntfy(topic, f"File {image_txt_path} not found.", {"Tags": "exclamation"})
        #print(f"File {image_txt_path} not found.")
        image_text = ""

    tweet_txt = row['tweet_text']

    return (image_data, image_text, tweet_txt)


def main(LLM_PROMPT: str, parser: argparse.ArgumentParser, OUTPUT_DIR: str):
    args = parser.parse_args()
    use_ntfy = args.use_ntfy
    topic = args.topic
    base_data_dir = args.data_dir

    BATCH_SIZE = 100

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    mmhs_dataframe = None
    local_df = None
    chunks = None

    if rank == 0:
        mmhs_dataframe = pd.read_json(os.path.join(base_data_dir, 'MMHS150K_GT.json'), lines=False, orient='index', convert_dates=False, convert_axes=False, dtype=str) 
        
        with open(os.path.join(OUTPUT_DIR, 'llm_prompt.txt'), 'w') as f:
            f.write(LLM_PROMPT)

        
        chunks = np.array_split(mmhs_dataframe, size)

     # Distribute DataFrame
    local_df = comm.scatter(chunks, root=0)
    len_local_df = local_df.shape[0]
    BATCH_SIZE = 100
    buffer = []

    output_path = os.path.join(OUTPUT_DIR, 'prompts.jsonl.gz')

    if rank == 0:
        # Open in append mode and write directly
        fout = gzip.open(output_path, 'at', encoding='utf-8')
        
        num_finished = 0
        local_iter = iter(local_df.iterrows())

        # Start processing loop
        while True:
            # Process own data first if available
            try:
                for _ in range(BATCH_SIZE):
                    idx, row = next(local_iter)
                    result = process_df_row(idx, row, base_data_dir, use_ntfy, topic)
                    if result is None:
                        continue
                    prompt = construct_prompts(LLM_PROMPT, result[0], result[1], result[2])
                    fout.write(json.dumps({"id": idx, "prompt": prompt}) + "\n")
            except StopIteration:
                break  # done with local work

            # Check for incoming batches
            while comm.Iprobe(source=MPI.ANY_SOURCE):
                data = comm.recv(source=MPI.ANY_SOURCE)
                if data == "DONE":
                    num_finished += 1
                else:
                    for entry in data:
                        fout.write(json.dumps(entry) + "\n")

        # After local data, drain remaining messages
        while num_finished < size - 1:
            data = comm.recv(source=MPI.ANY_SOURCE)
            if data == "DONE":
                num_finished += 1
            else:
                for entry in data:
                    fout.write(json.dumps(entry) + "\n")

        fout.close()

    else:
        # Worker ranks send in batches
        for i, (idx, row) in enumerate(tqdm(local_df.iterrows(), total=len_local_df, desc=f"Rank {rank}", position=rank)):
            result = process_df_row(idx, row, base_data_dir, use_ntfy, topic)
            if result is None:
                continue
            prompt = construct_prompts(LLM_PROMPT, result[0], result[1], result[2])
            buffer.append({"id": idx, "prompt": prompt})

            if len(buffer) >= BATCH_SIZE:
                comm.send(buffer, dest=0)
                buffer = []

        if buffer:
            comm.send(buffer, dest=0)

        comm.send("DONE", dest=0)



if __name__ == "__main__":
    global NTFY_TOPIC
    load_dotenv()
    NTFY_TOPIC = os.getenv("NTFY_TOPIC")
    MMHS_DATA_DIR = "/home/ubuntu/Coding/hatespeech-detection/data/MMHS150K"
    OUTPUT_DIR = "./processed_prompts/iteration_1/"

    
    parser = argparse.ArgumentParser(prog='llm-inference', description='LLM inference on Images for EE-559 Project')
    parser.add_argument('--use-ntfy', action='store_true', help='Use this option to send error messages as push notifications to your phone using the NTFY service')
    parser.add_argument('--topic', type=str, default=NTFY_TOPIC, help='NTFY topic to send error messages')
    parser.add_argument('--data_dir', type=str, default=MMHS_DATA_DIR, help='Directory containing the MMHS150K dataset')

    LLM_PROMPT = "You are a language model trained to detect hate speech in images posted to Twitter. You will be provided with the image, any text that could be found in the image (Preceded with 'Image Text:') as well as any text in the tweet body (Preceded with 'Tweet text:'). An Image will always be provided, however there may not be any image text or tweet text. In this case those fields will be empty e.g 'Image Text: . Tweet text:'. You should carefully look at everything provided to you and determine which tags or labels to assign. The labels you can choose from are 'NotHate', 'Racist', 'Sexist', 'Homophobe', 'Religion', 'OtherHate'. NotHate means that you do not think this is hateful. Racist means that you think the inputs are hateful against a particular group of humans e.g. Indians or Chinese or a groups of races e.g black people and brown people. Sexist means that you think the inputs are hateful against gender e.g males or females. Homophobe means that you think the inputs are hateful against a group of people with a particular sexual orientation e.g. Gay or Lesbian. If the input is hateful against multiple sexual orientations, you should still choose Homophobe e.g if the input is hateful against gays and transgender people, you should still choose Homophobe. Religion means that you think this input is hateful against a particular religious group e.g.Christians or Muslims or Hindus. OtherHate is a tag reserved for any form of hate that does not fit into the previous categories. You can also assign multiple labels to one input. So if an input made a hateful comment about a group of gay muslim men targeting their sexual orientation, their religion and their gender then you should assign the 'Homophobe', 'Religion' and 'Sexist' labels to the post. Return your answer as JSON." 

    main(LLM_PROMPT, parser, OUTPUT_DIR)
