import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    BlipForConditionalGeneration,
    Trainer,
    TrainingArguments,
    default_data_collator
)
from bitsandbytes import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import wandb
import matplotlib.pyplot as plt
import pandas as pd
import ast
import json
from dotenv import load_dotenv
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import argparse


#-------------- Config parameters --------------#
MODEL_NAME = "Salesforce/instructblip-vicuna-7b"
IMAGE_DIR  = "./images"
DATA_JSON  = "./annotations.json"
PROJECT    = "mmhs-finetune"
RUN_NAME   = "instructblip-lora"
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

        if len(tweet_text) == 0 :
            tweet_text = None

        image = Image.open(f"{self.image_path}/{image_id}.jpg").convert("RGB")
        
        if os.path.exists(f"{self.image_text_path}/{image_id}.json"):
            with open(f"{self.image_text_path}/{image_id}.json", "r") as f:
                image_text = json.load(f)["img_text"]
        else:
            image_text = None

        text = " ".join(filter(None, [tweet_text, image_text]))
        inputs = self.proc(text=text, images=image, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}


        labels = torch.zeros(NUM_LABELS, dtype=torch.float32)
        for label in ast.literal_eval(row["labels_str"]):
            labels[self.labels_mapping[label]] = 1

        inputs["labels"] = labels
        return inputs
    

class InstructBLIPClassifier(nn.Module):
    def __init__(self, base_model, hidden_size, num_labels):
        super().__init__()
        self.blip = base_model
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids=None, pixel_values=None, attention_mask=None, labels=None):
        outputs = self.blip.vision_language_encoder(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            return_dict=True,
        )
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS token
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}
    

def predict_top3_labels(logits):
    probs = torch.sigmoid(torch.tensor(logits))
    top3 = torch.topk(probs, 3).indices
    multi_hot = torch.zeros_like(probs, dtype=torch.int)
    multi_hot[top3] = 1
    return multi_hot, [ALLOWED[i] for i in top3]

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    labels = torch.tensor(labels).int()
    
    preds = []
    for logit in logits:
        ohe_vector, _ = predict_top3_labels(logit)
        preds.append(ohe_vector)
    preds = torch.stack(preds)

    acc = (ohe_vector == labels).all(dim=1).float().mean().item()
    f1 = f1_score(labels, preds, average="micro")
    return {"eval_accuracy": acc, "eval_f1": f1}

    
def main(args):
    image_path = args.image_path
    image_text_path = args.image_text_path
    dataset_json_path = args.dataset_json_path
    env_path = args.env_file
    model_path = args.model_path
    load_dotenv(env_path)
    WANDB_API_KEY = os.getenv("WANDB_API_KEY")
    wandb.login(key=WANDB_API_KEY)
    wandb.init(project=PROJECT, name=RUN_NAME)

    processor = AutoProcessor.from_pretrained(model_path)
    base_model = BlipForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    )

    model = InstructBLIPClassifier(base_model, base_model.config.text_config.hidden_size, NUM_LABELS)
    lora_cfg = LoraConfig(r=8, 
                          lora_alpha=16, 
                          target_modules=["q_proj", "v_proj"], 
                          bias="none",
                          task_type="SEQ_CLS")
    
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()



    training_dataset = MMHSDataset(image_path, image_text_path, dataset_json_path, f"{args.splits_path}/train_ids.txt", processor)
    validation_dataset = MMHSDataset(image_path, image_text_path, dataset_json_path, f"{args.splits_path}/val_ids.txt", processor)
    testing_dataset = MMHSDataset(image_path, image_text_path, dataset_json_path, f"{args.splits_path}/test_ids.txt", processor)

    args = TrainingArguments(
        output_dir="/workspace/output/out_ib",
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

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=training_dataset,
        eval_dataset=validation_dataset,
        tokenizer=processor,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train(resume_from_checkpoint=True)
    trainer.save_model("/workspace/output/saved_instructblip_classifier_finetuned")

    logs = trainer.state.log_history
    train_steps = [x["step"] for x in logs if "loss" in x and "eval_loss" not in x]
    train_loss = [x["loss"] for x in logs if "loss" in x and "eval_loss" not in x]
    eval_steps = [x["step"] for x in logs if "eval_loss" in x]
    eval_loss = [x["eval_loss"] for x in logs if "eval_loss" in x]
    eval_acc  = [x["eval_accuracy"] for x in logs if "eval_accuracy" in x]

    plt.figure(figsize=(6, 4))
    plt.plot(train_steps, train_loss, label="Train Loss")
    plt.plot(eval_steps, eval_loss, label="Val Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.savefig("instructblip_loss.png")

    plt.figure(figsize=(6, 4))
    plt.plot(eval_steps, eval_acc, label="Val Accuracy")  # no train acc
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.savefig("instructblip_acc.png")

    outputs = trainer.predict(testing_dataset)
    logits = outputs.predictions
    true_labels = torch.tensor(outputs.label_ids).int()
    preds_multi_hot = []
    top3_label_strs = []
    for logit in logits:
        bin_vec, top3_strs = predict_top3_labels(logit)
        preds_multi_hot.append(bin_vec)
        top3_label_strs.append(top3_strs)

    preds_multi_hot = torch.stack(preds_multi_hot)

    # Compute metrics
    micro_f1 = f1_score(true_labels, preds_multi_hot, average='micro')
    macro_f1 = f1_score(true_labels, preds_multi_hot, average='macro')
    precision = precision_score(true_labels, preds_multi_hot, average='micro')
    recall = recall_score(true_labels, preds_multi_hot, average='micro')
    accuracy = (preds_multi_hot == true_labels).float().mean().item()

    print("\nValidation Metrics (Top-3 Prediction)")
    print(f"  Accuracy     : {accuracy:.4f}")
    print(f"  Micro F1     : {micro_f1:.4f}")
    print(f"  Macro F1     : {macro_f1:.4f}")
    print(f"  Precision    : {precision:.4f}")
    print(f"  Recall       : {recall:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_json_path", type=str, required=True, help="Path to the MMHS150K_GT.json file")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the MMHS150K images")
    parser.add_argument("--image_text_path", type=str, required=True, help="Path to the MMHS150K image text")
    parser.add_argument("--splits_path", type=str, required=True, help="Path to the train-test-split.csv file")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model files")
    parser.add_argument("--env-file", type=str, required=True, help="Path to the .env file")
    args = parser.parse_args()
    main(args)