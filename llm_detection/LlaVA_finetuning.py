import os
from dotenv import load_dotenv
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments
)
from bitsandbytes import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import wandb
import matplotlib.pyplot as plt
import ast
import json
import pandas as pd

#-------------- Config parameters --------------#
MODEL_NAME = "llava-hf/llava-v1.6-mistral-7b-hf"
PROJECT    = "mmhs-finetune"
RUN_NAME   = "llava-lora"
ALLOWED    = ["NotHate","Racist","Sexist","Homophobe","Religion","OtherHate"]
NUM_LABELS = len(ALLOWED)
#----------------------------------------------#


class MMHSDataset(Dataset):
    def __init__(self, image_path: str, image_text_path: str, dataset_json_path: str, data_split_ids_path: str, processor: AutoProcessor):
        self.image_path = image_path
        self.image_text_path = image_text_path
        self.dataset_json_path = dataset_json_path
        self.labels_mapping = {
            "NotHate": 0,
            "Racist": 1,
            "Sexist": 2,
            "Homophobe": 3,
            "Religion": 4,
            "OtherHate": 5
        }
        self.data_split_ids_path = data_split_ids_path
        self.dataset_df = pd.read_json(self.dataset_json_path, lines=False, orient='index', convert_dates=False, convert_axes=False, dtype=str)
        
        with open(self.data_split_ids_path, "r") as f:
            self.ids_to_select = [int(line.strip()) for line in f]
        
        self.data_length = len(self.ids_to_select)

        self.dataset_df = self.dataset_df.iloc[self.ids_to_select]
        self.dataset_df.reset_index(inplace=True)
        self.proc = processor

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        row = self.dataset_df.iloc[idx]
        image_id = row["index"]
        tweet_text = row["tweet_text"]

        prompt = (
            "[Image]\n"
            "Task: Classify using any of: NotHate,Racist,Sexist,Homophobe,Religion,OtherHate.\n"
            "Give 3 labels, list them separated by commas.\n"
        )

        if len(tweet_text) > 0 :
            prompt += f"Tweet text: {tweet_text}\n"

        image = Image.open(f"{self.image_path}/{image_id}.jpg").convert("RGB")
        
        if os.path.exists(f"{self.image_text_path}/{image_id}.json"):
            with open(f"{self.image_text_path}/{image_id}.json", "r") as f:
                image_text = json.load(f)["img_text"]
                prompt += f"OCR text: {image_text}\n"
        else:
            image_text = None
        prompt += "###\nAssistant:"
        inputs = self.proc(text=prompt, images=image, return_tensors="pt", padding=True, truncation=True)
        for k in inputs: 
            inputs[k] = inputs[k].squeeze(0) 
        label_str = ",".join(ast.literal_eval(row["labels_str"]))
        inputs["labels"] = self.proc.tokenizer(label_str, return_tensors="pt").input_ids.squeeze(0)
        return inputs
    

def main():
    load_dotenv()
    image_path = "../data/MMHS150K/img_resized"
    image_text_path= "../data/MMHS150K/img_txt"
    dataset_json_path = "../data/MMHS150K/MMHS150K_GT.json"
    WANDB_API_KEY = os.getenv("WANDB_API_KEY")
    wandb.login(key=WANDB_API_KEY)
    wandb.init(project=PROJECT, name=RUN_NAME)

    # 3) Model + LoRA + 8-bit
    proc = AutoProcessor.from_pretrained(MODEL_NAME)
    bnb  = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    lora_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","v_proj"],
                          bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)

    training_dataset = MMHSDataset(image_path, image_text_path, dataset_json_path, "../data/MMHS150K/splits/train_ids.txt", proc)
    validation_dataset = MMHSDataset(image_path, image_text_path, dataset_json_path, "../data/MMHS150K/splits/val_ids.txt", proc)

    args = TrainingArguments(
        output_dir="./out_llava",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=3,
        num_train_epochs=4,
        logging_steps=100,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        fp16=True,
        report_to="wandb",
        run_name=RUN_NAME,
        remove_unused_columns=False,
    )
    
    # Set up the trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=training_dataset,
        eval_dataset=validation_dataset,
        tokenizer=proc,
        data_collator=lambda bs: {k:torch.stack([b[k] for b in bs]) for k in bs[0]},
    )

    # Start the training
    trainer.train()

    logs = trainer.state.log_history
    steps = [x["step"] for x in logs if "loss" in x]
    train_loss = [x["loss"] for x in logs if "loss" in x]
    eval_loss  = [x["eval_loss"] for x in logs if "eval_loss" in x]
    # NOTE: causal LM may not produce eval_accuracy by default
    plt.figure(figsize=(6,4))
    plt.plot(steps, train_loss, label="train_loss")
    plt.plot(steps, eval_loss,  label="eval_loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("LLaVA-Next Loss")
    plt.savefig("llava_loss.png")

if __name__ == "__main__":
    main()