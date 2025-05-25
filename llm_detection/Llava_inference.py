#!/usr/bin/env python
import argparse
import os
import json
import re
from functools import partial
from tqdm import tqdm
from PIL import Image
import json
import pandas as pd
import gzip

import torch
from torch.utils.data import DataLoader
from functools import partial

from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaNextForConditionalGeneration
)
from peft import PeftModel


def label_agg(row):
    res = 0
    for x in row:
        if x != 0:
            res+= 1
    return str(int(res))


def load_img_text(row_id, json_folder):
    file_path = os.path.join(json_folder, f"{row_id}.json")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("img_text", None)
    except Exception as e:
        return None


class MMHS150K:
    def __init__(self, image_path: str, image_text_path: str, dataset_json_path: str, data_split_ids_path: str):
        self.image_path = image_path
        self.image_text_path = image_text_path
        self.dataset_json_path = dataset_json_path
        self.data_split_ids_path = data_split_ids_path
        self.df = pd.read_json(self.dataset_json_path, lines=False, orient='index', convert_dates=False)
        self.df = self.df.reset_index()
        self.df['id'] = self.df['tweet_url'].str.extract(r'/status/(\d+)')
        self.df['label'] = self.df['labels'].apply(label_agg)

        with open(self.data_split_ids_path, 'r') as f:
            self.split_image_ids_str = set(line.strip() for line in f)

        self.df = self.df[self.df['id'].isin(self.split_image_ids_str)]
        self.df['img'] = f'{self.image_path}/' + self.df['id'] + '.jpg'

        self.load_img_text = lambda row_id: load_img_text(row_id, self.image_text_path)
        self.df['img_text'] = self.df['id'].apply(self.load_img_text)
        
        self.data_length = len(self.df)

    def __len__(self):
        return self.data_length
    
    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        tweet_id = sample['id']
        image_path = sample['img']
        img_text = sample['img_text']
        tweet_text = sample['tweet_text']

        #Remember this is a score of how hateful the image is.
        #I.e if 2 researchers label it as hateful, the score will be 2.
        #Scale is from 0 - 3
        labels = sample['label']
        image = Image.open(image_path).convert('RGB')
        other_data = {
            "tweet_id": tweet_id,
            "tweet_text": tweet_text,
            "img_text": img_text,
            "labels": labels
        }
        return image, other_data
    

def eval_collate_fn(inputs, processor):
    def create_chat_prompt(img_text, tweet_text): 
        return [
            {"role": "user", "content": f"<image>\nYou are a helpful vision assistant used to measure the hatefulness in memes. The user will give you an image and sometimes they might give you some text that was found in the image and the tweet. You should consider the image and the text given to determine a hatefulness score from 0 to 3. You can choose from the following integers for the score 0, 1, 2, 3. Do not give a score lower than 0 or higher than 3. A score of 3 is the maximum hate score possible, while 0 means no hate. Do not be overly sensitive in your judgement. You must be accurate in your judgement. Score the hatefulness in this meme.  Image Text: {img_text}. Tweet text: {tweet_text}"}
        ]
    
    images = []
    prompts = []
    answers = []
    twitter_ids = []

    for _input in inputs:
        image, other_data = _input
        images.append(image)
        tweet_text = other_data['tweet_text']
        img_text = other_data['img_text']
        labels = other_data['labels']

        chat = create_chat_prompt(img_text, tweet_text)
        prompt = processor.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
        prompts.append(prompt)
        answers.append(labels)
        twitter_ids.append(other_data['tweet_id'])

    batch = processor(text=prompts, images=images, return_tensors="pt", padding=True)

    # input_ids = batch["input_ids"]
    # attention_mask = batch["attention_mask"]
    # pixel_values = batch["pixel_values"]
    # image_sizes = batch["image_sizes"]


    return batch, answers, twitter_ids,



def main(args):

    processor = AutoProcessor.from_pretrained(args.base_model_path)
    processor.tokenizer.padding_side = "right"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = LlavaNextForConditionalGeneration.from_pretrained(
        args.base_model_path,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2"
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PeftModel.from_pretrained(base_model, args.adapter_path, torch_dtype=torch.float16)
    
    
    if args.checkpoint_file:
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        # Lightning prefixes everything with "model."
        state = {
            k.replace("model.", ""): v 
            for k, v in ckpt["state_dict"].items() 
            if k.startswith("model.")
        }
        model.load_state_dict(state, strict=False)
        print(f"Loaded LoRA weights from checkpoint {args.ckpt_path}")
    
    
    model.eval().to(device)
    test_ds = MMHS150K(
        image_path=args.image_path,
        image_text_path=args.image_text_path,
        dataset_json_path=args.dataset_json_path,
        data_split_ids_path=os.path.join(args.splits_path, "test_ids.txt")
    )
    collate_fn = partial(eval_collate_fn, processor=processor)
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True
    )

    total, correct, total_mae = 0, 0, 0
    results = []
    with torch.no_grad():
        for batch, answers, twitter_ids, in tqdm(test_loader, desc="Testing", total=len(test_loader)):
            batch = {k: v.to(device) for k, v in batch.items()}

            # generate hate‐score predictions
            gen_ids = model.generate(
                **batch,
                max_new_tokens=5
            )

            # decode only the newly generated tokens
            preds = processor.batch_decode(
                gen_ids[:, batch["input_ids"].size(1):],
                skip_special_tokens=True
            )
            
            for idx, (pred_str, ans_str) in enumerate(zip(preds, answers)):
                pred_str = pred_str.strip()
                match = re.search(r"\b[0-3]\b", pred_str)
                pred_num = int(match.group()) if match else None
                ans_num = int(ans_str)

                acc = 1 if pred_num == ans_num else 0
                mae = abs(pred_num - ans_num) if pred_num is not None else 3
                
                with gzip.open(args.llm_output, 'at', encoding='utf-8') as f:
                    # {'id': '1114558534635618305', 'response': {'input_labels': 3}} <--- Example output
                    f.write(json.dumps({"id": twitter_ids[idx], "response": {"input_labels": pred_num}}) + "\n")

                correct += acc
                total_mae += mae
                total += 1

                results.append({
                    "prediction": pred_num,
                    "ground_truth": ans_num,
                    "acc": acc,
                    "mae": mae
                })

    test_acc = correct / total
    test_mae = total_mae / total
    print("\n=== Test Results ===")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"MAE     : {test_mae:.4f}")

    with open(args.output_metrics, "w") as f:
        json.dump({
            "overall": {"accuracy": test_acc, "mae": test_mae},
            "samples": results
        }, f, indent=2)

    print(f"Per-sample results saved to {args.output_results}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference & validation on MMHS150K test set with saved LoRA weights"
    )
    parser.add_argument("--dataset-json-path", type=str, required=True, help="Path to MMHS150K_GT.json")
    parser.add_argument("--image-path", type=str, required=True, help="Directory with MMHS150K images")
    parser.add_argument("--image-text-path", type=str, required=True, help="Directory with per-image OCR JSONs")
    parser.add_argument("--splits-path", type=str, required=True, help="Directory containing train/val/test id txt files")
    parser.add_argument("--base-model-path", type=str, required=True, help="Path to the original Llava checkpoint (before fine-tuning)")
    parser.add_argument("--adapter-path", type=str, required=True, help="Directory where you saved LoRA adapters + processor")
    parser.add_argument("--output-metrics", type=str, required=True, help="File to write metrics")
    parser.add_argument("--llm-output", type=str, required=True, help="Path to the output .jsonl.gz file")
    parser.add_argument("--checkpoint-file", type=str, default=None, required=False, help="Path to the checkpoint file to resume from")
    main(parser.parse_args())
