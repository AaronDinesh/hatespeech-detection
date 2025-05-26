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
from bitsandbytes.optim import Adam8bit

from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaNextForConditionalGeneration
)
from peft import PeftModel

import lightning as L
from typing import Dict, Any
import math

MAX_LENGTH = 3000

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

def train_collate_fn(inputs, processor):
    """
    Prepares a batch of training data for a model that processes images and text.

    Args:
        inputs (list): A list of tuples where each tuple contains an image and the corresponding ground truth.

    Returns:
        tuple: A tuple containing:
            - input_ids (torch.Tensor): Tokenized input IDs of the text prompts.
            - attention_mask (torch.Tensor): Attention mask for the input IDs.
            - pixel_values (torch.Tensor): Preprocessed pixel values of the images.
            - image_sizes (list): List of original sizes of the images.
            - labels (torch.Tensor): Labels for training, with padding token IDs replaced by -100.
    """

    def create_chat_prompt(img_text, tweet_text, ground_truth): 
        return [
            {"role": "user", "content": f"<image>\nYou are a helpful vision assistant used to measure the hatefulness in memes. The user will give you an image and sometimes they might give you some text that was found in the image and the tweet. You should consider the image and the text given to determine a hatefulness score from 0 to 3. You can choose from the following integers for the score 0, 1, 2, 3. Do not give a score lower than 0 or higher than 3. A score of 3 is the maximum hate score possible, while 0 means no hate. Do not be overly sensitive in your judgement. You must be accurate in your judgement. Score the hatefulness in this meme.  Image Text: {img_text}. Tweet text: {tweet_text}"},
            {"role": "assistant", "content": ground_truth}
        ]
    
    images = []
    prompts = []
    for _input in inputs:
        image, other_data = _input
        images.append(image)
        tweet_text = other_data['tweet_text']
        img_text = other_data['img_text']
        labels = other_data['labels']

        chat = create_chat_prompt(img_text, tweet_text, labels)
        prompt = processor.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
        prompts.append(prompt)
    
    batch = processor(text=prompts, images=images, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values = batch["pixel_values"]
    image_sizes = batch["image_sizes"]
    labels = batch["labels"]

    return input_ids, attention_mask, pixel_values, image_sizes, labels


class LlavaModelPLModule(L.LightningModule):
    """
    A PyTorch Lightning module for training and validating a multimodal model that processes images and text.

    Attributes:
        config (dict): Configuration dictionary containing model hyperparameters and settings.
        processor (object): A processor object for handling text and image pre-processing.
        model (torch.nn.Module): The model to be trained and evaluated.

    Methods:
        training_step(batch, batch_idx):
            Executes a single training step, computing the loss and logging it.
        
        validation_step(batch, batch_idx, dataset_idx=0):
            Executes a single validation step, generating predictions, comparing them to ground truth, and logging the normalized edit distance.
        
        configure_optimizers():
            Sets up the optimizer and optionally, learning rate scheduler for the training process.
        
        train_dataloader():
            Returns a DataLoader for the training dataset.
        
        val_dataloader():
            Returns a DataLoader for the validation dataset.
    """
    def __init__(self, config, processor, model, training_dataset, validation_dataset, train_collate_fn, val_collate_fn):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset 
        self.train_collate_fn = train_collate_fn
        self.val_collate_fn = val_collate_fn

        self.batch_size = config.get("batch_size")


    
    def on_load_checkpoint(self, checkpoint: Dict[str,Any]) -> None:
        """
        Lightning is about to call load_state_dict(checkpoint['state_dict'], strict=True).
        We filter out any of the BitsAndBytes quantization keys so that only the real
        model + LoRA weights remain.
        """
        state = checkpoint["state_dict"]
        filtered = {
            k: v
            for k, v in state.items()
            # drop any param that belongs to bitsandbytes quant state / absmax / quant_map
            if not (
                "quant_state"   in k or
                "absmax"        in k or
                "quant_map"     in k or
                "nested"        in k
            )
        }
        checkpoint["state_dict"] = filtered

    def training_step(self, batch, batch_idx):
        """
        Performs a single step of training.

        Args:
            batch (tuple): A tuple containing input_ids, attention_mask, pixel_values, image_sizes, and labels.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """

        input_ids, attention_mask, pixel_values, image_sizes, labels = batch

        outputs = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            image_sizes=image_sizes,
                            labels=labels
                          )
        loss = outputs.loss

        self.log("train_loss", loss, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx, dataset_idx=0):
        """
        Performs a single step of validation, generating predictions and computing the normalized edit distance.

        Args:
            batch (tuple): A tuple containing input_ids, attention_mask, pixel_values, image_sizes, and answers.
            batch_idx (int): The index of the current batch.
            dataset_idx (int, optional): Index of the dataset in case of multiple datasets. Defaults to 0.

        Returns:
            list: A list of normalized edit distances between predictions and ground truth answers.
        """

        input_ids, attention_mask, pixel_values, image_sizes, answers = batch

        # autoregressively generate token IDs
        generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                       pixel_values=pixel_values, image_sizes=image_sizes, max_new_tokens=5)
        # turn them back into text, chopping of the prompt
        # important: we don't skip special tokens here, because we want to see them in the output
        predictions = self.processor.batch_decode(generated_ids[:, input_ids.size(1):], skip_special_tokens=True)
        
        accs = []
        maes = []
        rmse = []
        for pred, ans in zip(predictions, answers):
            pred = pred.strip()
            ans = ans.strip()
            ans = int(ans)
            matches = re.search(r'\b[0-3]\b', pred)
            number = int(matches.group()) if matches else None
            if number is not None:
                accs.append(1 if number == ans else 0)
                maes.append(abs(number - ans))
                rmse.append((number - ans)**2)
            else:
                accs.append(0)
                maes.append(3)
                rmse.append(9)
        batch_acc = sum(accs) / len(accs)
        batch_mae = sum(maes) / len(maes)
        batch_rmse = math.sqrt(sum(rmse) / len(rmse))

        self.log('val_acc', batch_acc, sync_dist=True)
        self.log('val_mae', batch_mae, sync_dist=True)
        self.log('val_rmse', batch_rmse, sync_dist=True)
        return {'val_acc': batch_acc, 'val_mae': batch_mae, 'val_rmse': batch_rmse}
            
    def configure_optimizers(self):
        """
        Configures the optimizer for training.

        Returns:
            torch.optim.Optimizer: The optimizer for training.
        """
        # you could also add a learning rate scheduler if you want
        trainable = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = Adam8bit(trainable, lr=self.config.get("lr"))
        #optimizer = DeepSpeedCPUAdam(self.parameters(), lr=self.config.get("lr"))

        return optimizer

    def train_dataloader(self):
        """
        Returns the DataLoader for the training dataset.

        Returns:
            DataLoader: The DataLoader for the training dataset.
        """


        return DataLoader(self.training_dataset, collate_fn=self.train_collate_fn, batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    
    def val_dataloader(self):
        """
        Returns the DataLoader for the validation dataset.

        Returns:
            DataLoader: The DataLoader for the validation dataset.
        """
        return DataLoader(self.validation_dataset, collate_fn=self.val_collate_fn, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)



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
    
    config = {"max_epochs": 10,
            "val_check_interval": 0.25, # how many times we want to validate during an epoch
            "check_val_every_n_epoch": 1,
            "gradient_clip_val": 1.0,
            "accumulate_grad_batches": 8,
            "lr": 1e-4,
            "batch_size": 1,
            # "seed":2022,
            "num_nodes": 1,
            "warmup_steps": 50,
            "result_path": "./result",
            "verbose": True,
            "num_workers": 4
    }

    train_collate = partial(train_collate_fn, processor=processor)
    eval_collate = partial(eval_collate_fn, processor=processor)

    
    model_module = LlavaModelPLModule.load_from_checkpoint(
        args.checkpoint_file,
        # pass in the same __init__ args you originally used:
        config        = config,
        processor     = processor,
        model         = model,
        train_ds      = None,           # not needed for inference
        val_ds        = None,
        train_collate = None,
        eval_collate  = None,
    )
    
    print("Loaded weights from checkpoint")
    
    model = model_module.model
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
    parser.add_argument("--checkpoint-file", type=str, default=None, required=True, help="Path to the checkpoint file to resume from")
    main(parser.parse_args())
