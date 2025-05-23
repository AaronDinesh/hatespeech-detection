import argparse
import os
from dotenv import load_dotenv
from PIL import Image
import pandas as pd
from transformers import BitsAndBytesConfig, LlavaNextForConditionalGeneration, AutoProcessor
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import torch
import json
import wandb
import numpy as np
from torch.utils.data import Dataset
from typing import Any, Dict
import random
from bitsandbytes.optim import Adam8bit
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score, accuracy_score, mean_absolute_error
import re
from functools import partial
from lightning.pytorch.callbacks import ModelCheckpoint

MAX_LENGTH = 3000


label_mapping = {
    "NotHate": 0,
    "Racist": 1,
    "Sexist": 1,
    "Homophobe": 1,
    "Religion": 1,
    "OtherHate": 1
}

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector', 'vision_model']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

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

def eval_collate_fn(inputs, processor):
    def create_chat_prompt(img_text, tweet_text): 
        return [
            {"role": "user", "content": f"<image>\nYou are a helpful vision assistant used to measure the hatefulness in memes. The user will give you an image and sometimes they might give you some text that was found in the image and the tweet. You should consider the image and the text given to determine a hatefulness score from 0 to 3. You can choose from the following integers for the score 0, 1, 2, 3. Do not give a score lower than 0 or higher than 3. A score of 3 is the maximum hate score possible, while 0 means no hate. Do not be overly sensitive in your judgement. You must be accurate in your judgement. Score the hatefulness in this meme.  Image Text: {img_text}. Tweet text: {tweet_text}"}
        ]
    
    images = []
    prompts = []
    answers = []

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

    batch = processor(text=prompts, images=images, return_tensors="pt", padding=True)

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values = batch["pixel_values"]
    image_sizes = batch["image_sizes"]


    return input_ids, attention_mask, pixel_values, image_sizes, answers


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

        self.log("train_loss", loss)

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
        
        scores = []

        for pred, ans in zip(predictions, answers):
            pred = pred.strip()
            ans = ans.strip()
            
            matches = re.search(r'\b[0-3]\b', pred)
            number = int(matches.group(0)) if matches else None
            if number is not None:
                ans = int(ans)
                acc = 1 if number == ans else 0
                mae = abs(number - ans)
                qwk = cohen_kappa_score([ans], [number], weights='quadratic', labels=[0, 1, 2, 3])
            else:
                acc = 0
                mae = 3
                qwk = 0
            scores.append((acc, mae, qwk))
            self.log(f"val_acc_{dataset_idx}", acc)
            self.log(f"val_mae_{dataset_idx}", mae)
            self.log(f"val_qwk_{dataset_idx}", qwk)
            
        return scores
        
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
    load_dotenv(args.env_file)
    dataset_json_path = args.dataset_json_path
    image_path = args.image_path
    image_text_path = args.image_text_path
    splits_path = args.splits_path
    model_path = args.model_path
    model_save_path = args.model_save_path
    checkpoint_save_path = args.checkpoint_save_path
    PROJECT = "mmhs-finetune"
    RUN_NAME = "llava-lora"


    if torch.cuda.is_available():
        print("PyTorch is connected to GPU.")
        print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
    else:
        print("PyTorch is not connected to GPU.")

    # Determine how many GPUs we have
    ngpus = torch.cuda.device_count()

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    processor = AutoProcessor.from_pretrained(model_path)
    processor.tokenizer.padding_side = "right" # during training, one always uses padding on the right
    #bnb_config = BitsAndBytesConfig(load_in_8bit=True, device_map="auto")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    model = LlavaNextForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, quantization_config=bnb_config, attn_implementation="flash_attention_2")
    
    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=find_all_linear_names(model),
        init_lora_weights="gaussian",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()  # reclaim activation memory
    model.config.use_cache = False         # drop the KV cache during training


    # after you’ve loaded your model (with 4-bit quant, LoRA, etc.)
    total_params = sum(p.numel() for p in model.parameters())
    bits_per_param = 4  # or 8 if you reverted to 8-bit quant, 16 for fp16, etc.
    bytes_for_weights = total_params * bits_per_param / 8
    print(f"≈{bytes_for_weights/1024**3:.2f} GB for raw weights")

    train_ds = MMHS150K(image_path, image_text_path, dataset_json_path, f"{splits_path}/train_ids.txt")
    val_ds = MMHS150K(image_path, image_text_path, dataset_json_path, f"{splits_path}/val_ids.txt")
    test_ds = MMHS150K(image_path, image_text_path, dataset_json_path, f"{splits_path}/test_ids.txt")

    WANDB_API_KEY = os.getenv("WANDB_KEY")
    wandb.login(key=WANDB_API_KEY)
    wandb_logger = WandbLogger(project=PROJECT, name=RUN_NAME)

    config = {"max_epochs": 4,
            "val_check_interval": 0.1, # how many times we want to validate during an epoch
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

    model_module = LlavaModelPLModule(config, processor, model, train_ds, val_ds, train_collate, eval_collate)


    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_save_path, # Use your checkpoint_save_path
        filename='{epoch}-{step}',
        save_top_k=-1,  # Save all checkpoints
        every_n_epochs=0, # Disable epoch-based saving if using val_check_interval for finer control
        every_n_train_steps=None # Set if you prefer step-based rather than val_check_interval
    )


    trainer = L.Trainer(
        accelerator="gpu",
        devices=ngpus,
        max_epochs=config.get("max_epochs"),
        accumulate_grad_batches=config.get("accumulate_grad_batches"),
        check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
        gradient_clip_val=config.get("gradient_clip_val"),
        val_check_interval=config.get("val_check_interval"),
        callbacks=[checkpoint_callback],
        precision="16-mixed",
        limit_val_batches=5,
        num_sanity_val_steps=0,
        logger=wandb_logger,
        strategy=DDPStrategy(find_unused_parameters=True),
    )

    trainer.fit(model_module)
    trainer.save_checkpoint(f"{checkpoint_save_path}/llava-lora.ckpt")

    # Save PEFT model (adapters) and processor only on global rank 0
    if trainer.global_rank == 0: # or trainer.is_global_zero
        model_module.model.save_pretrained(model_save_path)
        processor.save_pretrained(model_save_path)
        print(f"Adapter and processor saved in {model_save_path} by rank {trainer.global_rank}")

    
    # Optional: Barrier to ensure all processes wait for rank 0 to finish saving
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-json-path", type=str, required=True, help="Path to the MMHS150K_GT.json file")
    parser.add_argument("--image-path", type=str, required=True, help="Path to the MMHS150K images")
    parser.add_argument("--image-text-path", type=str, required=True, help="Path to the MMHS150K image text")
    parser.add_argument("--splits-path", type=str, required=True, help="Path to the train-test-split.csv file")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model files")
    parser.add_argument("--env-file", type=str, required=True, help="Path to the .env file")
    parser.add_argument("--model-save-path", type=str, required=True, help="Path to save the tuned model files")
    parser.add_argument("--checkpoint-save-path", type=str, required=True, help="Path to save the checkpoint files")
    args = parser.parse_args()
    main(args)
