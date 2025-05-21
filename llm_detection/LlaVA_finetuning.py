import os
from dotenv import load_dotenv
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    BitsAndBytesConfig
)
#from bitsandbytes import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import wandb
import json
import ast
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from functools import partial
import numpy as np

#-------------- Config parameters --------------#
MODEL_NAME       = "llava-hf/llava-v1.6-mistral-7b-hf"
PROJECT          = "mmhs-finetune"
RUN_NAME         = "llava-lora"
ALLOWED_LABELS   = ["NotHate","Racist","Sexist","Homophobe","Religion","OtherHate"]
LABEL2ID         = {lab:i for i,lab in enumerate(ALLOWED_LABELS)}
NUM_LABELS       = len(ALLOWED_LABELS)
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
        self.processor = processor

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        row = self.dataset_df.iloc[idx]
        image_id   = row['index']
        tweet_text = row['tweet_text'] or ""
        labels_list = ast.literal_eval(row['labels_str'])

        # build prompt string
        prompt = (
            "Task: Classify the following meme using exactly three labels from: " +
            ",".join(ALLOWED_LABELS) + ".\n"
        )
        if tweet_text:
            prompt += f"Tweet text: {tweet_text}\n"
        # optional OCR text
        img_txt_file = os.path.join(self.image_text_path, f"{image_id}.json")
        if os.path.exists(img_txt_file):
            with open(img_txt_file) as j:
                ocr = json.load(j).get('img_text', "")
            if ocr:
                prompt += f"OCR text: {ocr}\n"
        prompt += "###\nAssistant:"

        # load & prepare image + text
        image = Image.open(os.path.join(self.image_path, f"{image_id}.jpg")).convert('RGB')
        # apply chat template
        # construct single-turn conversation
        conv = [{"role":"user","content":[{"type":"image"},{"type":"text","text":prompt}]}]
        text_prompt = self.processor.apply_chat_template(conv, add_assistant_prompt=True)
        inputs = self.processor(text=text_prompt, images=image,
                                return_tensors="pt", padding=True, truncation=True)

        # labels -> token ids
        label_str = ",".join(labels_list)
        label_ids = self.processor.tokenizer(label_str,
                                             return_tensors="pt").input_ids.squeeze(0)

        item = {
            'input_ids':      inputs.input_ids.squeeze(0),
            'attention_mask': inputs.attention_mask.squeeze(0),
            'pixel_values':   inputs.pixel_values.squeeze(0),
            'labels':         label_ids
        }
        return item

def collate_fn(batch, proc):
    input_ids      = pad_sequence([b['input_ids'] for b in batch], batch_first=True,
                                   padding_value=proc.tokenizer.pad_token_id)
    attention_mask = pad_sequence([b['attention_mask'] for b in batch], batch_first=True,
                                   padding_value=0)
    pixel_values   = torch.stack([b['pixel_values'] for b in batch])
    labels = pad_sequence([b['labels'] for b in batch], batch_first=True,
                           padding_value=proc.tokenizer.pad_token_id)
    # mask padding tokens
    labels[labels == proc.tokenizer.pad_token_id] = -100

    return {
        'input_ids':      input_ids,
        'attention_mask': attention_mask,
        'pixel_values':   pixel_values,
        'labels':         labels
    }
    

def main(args):
    image_path = args.image_path
    image_text_path = args.image_text_path
    dataset_json_path = args.dataset_json_path
    env_path = args.env_file
    model_path = os.path.expanduser(args.model_path)
    splits_path = args.splits_path
    model_save_path = args.model_save_path
    load_dotenv(env_path)
    WANDB_API_KEY = os.getenv("WANDB_KEY")
    wandb.login(key=WANDB_API_KEY)
    wandb.init(project=PROJECT, name=RUN_NAME)


    proc = AutoProcessor.from_pretrained(model_path)
    #proc.patch_size                     = model.config.vision_config.patch_size
    #proc.num_additional_image_tokens    = proc.model_kwargs.get('num_image_tokens', 1)
    #proc.vision_feature_select_strategy = proc.model_kwargs.get('vision_feature_select_strategy', 'center')
    proc.tokenizer.padding_side   = 'left'
    proc.tokenizer.pad_token      = proc.tokenizer.eos_token
    proc.tokenizer.pad_token_id   = proc.tokenizer.eos_token_id    

    


    bnb = BitsAndBytesConfig(load_in_8bit=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        quantization_config=bnb,
        torch_dtype=torch.float16,
        device_map='auto'
    )
    model.config.use_cache = False
    lora_cfg = LoraConfig(r=8, lora_alpha=16,
                          target_modules=["q_proj","v_proj"],
                          bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)
    
    print("Vision patch size:", model.config.vision_config.patch_size)
    print("Image size:", model.config.vision_config.image_size)
    print("Tokenizer pad token ID:", proc.tokenizer.pad_token_id)


    train_ds = MMHSDataset(image_path, image_text_path, dataset_json_path, f"{splits_path}/train_ids.txt", proc)
    val_ds = MMHSDataset(image_path, image_text_path, dataset_json_path, f"{splits_path}/val_ids.txt", proc)
    test_ds = MMHSDataset(image_path, image_text_path, dataset_json_path, f"{splits_path}/test_ids.txt", proc)
    
    print("Created Model and Training Sets")
    
    def compute_metrics(eval_pred: EvalPrediction):
        # logits: [batch, seq_len, vocab]; labels: [batch, seq_len]
        logits, labels = eval_pred.predictions, eval_pred.label_ids
        pred_ids = np.argmax(logits, axis=-1)

        # decode all tokens (teacher-forcing outputs)
        decoded_preds  = proc.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
        decoded_labels = proc.tokenizer.batch_decode(
            np.where(labels != -100, labels, proc.tokenizer.pad_token_id),
            skip_special_tokens=True
        )

        # split on commas, strip whitespace, build sets
        y_true = [set(lbl.strip().split(','))      for lbl  in decoded_labels]
        y_pred = [set(pred.strip().split(',')[:3]) for pred in decoded_preds]

        # exact-match on sets (order-independent)
        correct = [1 if t == p else 0 for t, p in zip(y_true, y_pred)]
        accuracy = sum(correct) / len(correct) if correct else 0.0
        return {'accuracy': accuracy}




    train_args = TrainingArguments(
        output_dir="/workspace/output/out_llava",
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
        remove_unused_columns=False
    )

    collate_func = partial(collate_fn, proc=proc)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_func,
        tokenizer=proc,
        compute_metrics=compute_metrics
    )


    # train
    trainer.train()
    model.save_pretrained('/workspace/output/tuned_llava')

    # extract metrics
    logs = trainer.state.log_history
    steps = [x['step'] for x in logs if 'loss' in x]
    train_loss = [x['loss'] for x in logs if 'loss' in x]
    val_loss   = [x['eval_loss'] for x in logs if 'eval_loss' in x]
    val_acc    = [x['eval_accuracy'] for x in logs if 'eval_accuracy' in x]

    # plot losses
    plt.figure()
    plt.plot(steps, train_loss, label='train_loss')
    plt.plot(steps, val_loss, label='val_loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.savefig('llava_loss.png')

    # plot validation accuracy
    if val_acc:
        acc_steps = [x['step'] for x in logs if 'eval_accuracy' in x]
        plt.figure()
        plt.plot(acc_steps, val_acc, label='val_accuracy')
        plt.xlabel('Step')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Validation Accuracy')
        plt.savefig('llava_accuracy.png')

    # test set evaluation
    test_metrics = trainer.evaluate(test_dataset=test_ds)
    print('Test metrics:', test_metrics)
    with open('llava_test_metrics.json','w') as f:
        json.dump(test_metrics, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_json_path", type=str, required=True, help="Path to the MMHS150K_GT.json file")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the MMHS150K images")
    parser.add_argument("--image_text_path", type=str, required=True, help="Path to the MMHS150K image text")
    parser.add_argument("--splits-path", type=str, required=True, help="Path to the train-test-split.csv file")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model files")
    parser.add_argument("--env-file", type=str, required=True, help="Path to the .env file")
    parser.add_argument("--model-save-path", type=str, required=True, help="Path to save the tuned model files")
    args = parser.parse_args()
    main(args)
