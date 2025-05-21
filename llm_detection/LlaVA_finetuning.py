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


# In MMHSDataset class:
class MMHSDataset(Dataset):
    def __init__(self, image_path: str, image_text_path: str, dataset_json_path: str, data_split_ids_path: str): # Processor not strictly needed here
        self.image_path = image_path
        self.image_text_path = image_text_path
        self.allowed_labels = ALLOWED_LABELS # Make ALLOWED_LABELS accessible

        try:
            self.dataset_full_df = pd.read_json(dataset_json_path, lines=False, orient='index', convert_dates=False, convert_axes=False, dtype=str)
        except Exception as e:
            print(f"ERROR: Failed to load or parse dataset JSON: {dataset_json_path}. Error: {e}")
            raise

        try:
            with open(data_split_ids_path, "r") as f:
                split_image_ids_int = [int(line.strip()) for line in f if line.strip()]
        except FileNotFoundError:
            print(f"ERROR: ID file not found: {data_split_ids_path}")
            raise
        except ValueError as e:
            print(f"ERROR: Could not parse IDs in {data_split_ids_path}. Ensure they are integers. Error: {e}")
            raise
        
        split_image_ids_str = [str(id_val) for id_val in split_image_ids_int]
        self.dataset_df = self.dataset_full_df[self.dataset_full_df.index.isin(split_image_ids_str)]
        self.data_length = len(self.dataset_df) 

        if self.data_length == 0:
            print(f"WARNING: Dataset for split {data_split_ids_path} is empty after filtering.")

        self.dataset_df.reset_index(inplace=True) 
        
        if 'index' in self.dataset_df.columns:
            self.id_column_name = 'index'
        elif not self.dataset_df.empty:
            self.id_column_name = self.dataset_df.columns[0]
            print(f"Warning: 'index' column not found. Assuming ID column is '{self.id_column_name}'.")
        else:
             self.id_column_name = 'index'
        
        # self.processor = processor # Not needed if collator does all processing

    def __len__(self):
        return self.data_length
    
    def __getitem__(self, idx):
        if idx >= self.data_length:
            raise IndexError(f"Index {idx} out of bounds for dataset of length {self.data_length}")
        
        row = self.dataset_df.iloc[idx]
        image_id = str(row[self.id_column_name])
        tweet_text = str(row['tweet_text']) if pd.notna(row['tweet_text']) and row['tweet_text'] else ""

        try:
            labels_list_str = str(row['labels_str'])
            if not labels_list_str.strip() or pd.isna(row['labels_str']):
                labels_list = []
            else:
                labels_list = ast.literal_eval(labels_list_str)
            if not isinstance(labels_list, list) or not all(isinstance(lbl, str) for lbl in labels_list):
                labels_list = []
        except (ValueError, SyntaxError, TypeError):
            labels_list = []
            # print(f"Warning: Could not parse labels_str for {image_id}, using empty list.") # Less verbose
        
        # --- Construct User Prompt Content (Text Part) ---
        user_query_text = ( # This is the text that accompanies the image in the user's turn
            "Task: Classify the following meme using exactly three labels from: " +
            ",".join(self.allowed_labels) + ".\n"
        )
        if tweet_text:
            user_query_text += f"Tweet text: {tweet_text}\n"
        
        img_txt_file = os.path.join(self.image_text_path, f"{image_id}.json")
        if os.path.exists(img_txt_file):
            try:
                with open(img_txt_file, 'r') as j:
                    ocr_data = json.load(j)
                    ocr = str(ocr_data.get('img_text', ""))
                if ocr:
                    user_query_text += f"OCR text: {ocr}\n"
            except Exception: # Simplified error logging for brevity
                # print(f"Warning: Error loading OCR text for {image_id}: {e}")
                pass
        
        user_query_text = user_query_text.rstrip() 

        # --- Load Image ---
        image_file_path = os.path.join(self.image_path, f"{image_id}.jpg")
        try:
            pil_image = Image.open(image_file_path).convert('RGB')
        except Exception as e:
            print(f"ERROR: Could not load image {image_file_path} for ID {image_id}. Error: {e}. Returning None.")
            return None

        assistant_response_text = ",".join(labels_list)
        
        # Return the raw components needed by the collator
        return {
            "pil_image": pil_image,
            "user_query": user_query_text, # Text part of user's turn
            "assistant_response": assistant_response_text # Text of assistant's turn
        }
   
class AdaptedLLavaDataCollator:
    def __init__(self, processor):
        self.processor = processor
        # Determine the image token string from the tokenizer, default to <image>
        self.image_token = getattr(processor.tokenizer, "image_token", "<image>")
        # Determine the full "ASSISTANT: " prefix string for prompt length calculation
        # This requires applying chat template to an empty assistant message or knowing the template
        try:
            _dummy_user_turn = [{"role": "user", "content": "test"}]
            self.assistant_prefix = processor.tokenizer.apply_chat_template(
                _dummy_user_turn, tokenize=False, add_generation_prompt=True # Gets "USER: test\nASSISTANT:"
            )
            # Remove the user part to get just the assistant prefix
            _dummy_user_rendered = processor.tokenizer.apply_chat_template(
                _dummy_user_turn, tokenize=False, add_generation_prompt=False
            )
            self.assistant_prefix = self.assistant_prefix.replace(_dummy_user_rendered, "").strip()
            if not self.assistant_prefix: # Fallback if stripping fails or template is unusual
                self.assistant_prefix = "ASSISTANT:" # Common default
        except Exception:
            self.assistant_prefix = "ASSISTANT:" # Fallback

        print(f"DEBUG: Collator using assistant prefix: '{self.assistant_prefix}'")


    def __call__(self, examples):
        examples = [ex for ex in examples if ex is not None]
        if not examples:
            empty_float_dtype = torch.float16
            if hasattr(self.processor, 'image_processor') and hasattr(self.processor.image_processor, 'config') and hasattr(self.processor.image_processor.config, 'torch_dtype'):
                 empty_float_dtype = torch.float16 if self.processor.image_processor.config.torch_dtype == "float16" else torch.float32
            return {"input_ids": torch.empty(0, dtype=torch.long), "attention_mask": torch.empty(0, dtype=torch.long),
                    "pixel_values": torch.empty(0, dtype=empty_float_dtype), "labels": torch.empty(0, dtype=torch.long)}

        raw_texts_for_model_input = []
        images_for_processing = []
        prompt_part_lengths = []

        for example in examples:
            pil_image = example["pil_image"]
            user_query = example["user_query"]
            assistant_response = example["assistant_response"]
            images_for_processing.append(pil_image)
            
            user_turn_text_with_image_placeholder = f"{self.image_token}\n{user_query}"
            messages_for_template = [
                {"role": "user", "content": user_turn_text_with_image_placeholder},
                {"role": "assistant", "content": assistant_response}
            ]

            try:
                full_conversation_text = self.processor.tokenizer.apply_chat_template(
                    messages_for_template, tokenize=False, add_generation_prompt=False
                )
                raw_texts_for_model_input.append(full_conversation_text)

                prompt_messages_for_len_calc = [{"role": "user", "content": user_turn_text_with_image_placeholder}]
                prompt_prefix_text_for_len = self.processor.tokenizer.apply_chat_template(
                    prompt_messages_for_len_calc, tokenize=False, add_generation_prompt=True
                )
                tokenized_prompt_prefix = self.processor.tokenizer(
                    prompt_prefix_text_for_len, add_special_tokens=True, truncation=False
                ).input_ids
                prompt_part_lengths.append(len(tokenized_prompt_prefix))
            except Exception as e:
                print(f"Error in collator applying chat template. User query: '{user_query[:50]}...', Error: {e}")
                raw_texts_for_model_input.append(f"USER: {self.image_token}\n{user_query}\n###\n{self.assistant_prefix} {assistant_response}")
                prompt_part_lengths.append(len(self.processor.tokenizer(f"USER: {self.image_token}\n{user_query}\n###\n{self.assistant_prefix}", add_special_tokens=True).input_ids))

        batch = self.processor(
            text=raw_texts_for_model_input,
            images=images_for_processing,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=1024
        )

        labels = batch["input_ids"].clone()
        actual_batch_size = labels.shape[0]

        for i in range(actual_batch_size):
            if i < len(prompt_part_lengths):
                current_prompt_len = prompt_part_lengths[i]
                current_prompt_len = min(current_prompt_len, labels.shape[1])
                labels[i, :current_prompt_len] = -100
            else:
                print(f"Warning: prompt_part_length missing for item {i}. Labels for this item may be incorrect.")

        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        batch["labels"] = labels

        # --- FIX: Remove image_sizes if present ---
        if "image_sizes" in batch:
            # print("DEBUG: Removing 'image_sizes' from batch as it's not expected by CLIPVisionModel via this code path.")
            batch.pop("image_sizes")
        # --- END FIX ---

        return batch 

def compute_metrics(eval_pred: EvalPrediction, processor: AutoProcessor): # Pass processor for tokenizer
    # logits: [batch, seq_len, vocab]; labels: [batch, seq_len]
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    
    # Get most likely predicted token IDs
    pred_ids = np.argmax(logits, axis=-1)

    # Decode all predicted tokens (model's actual output string)
    # skip_special_tokens=True removes padding, EOS, etc.
    decoded_preds_text = processor.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    
    # Decode ground truth labels
    # Replace -100 (ignore index) with pad_token_id for decoding
    labels_for_decoding = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
    decoded_labels_text = processor.tokenizer.batch_decode(
        labels_for_decoding,
        skip_special_tokens=True
    )

    y_true_sets = []
    for label_str in decoded_labels_text:
        # True labels: split, strip, filter empty, then make a set.
        # We expect these to be the 3 labels from the dataset.
        true_tags = {tag.strip() for tag in label_str.split(',') if tag.strip()}
        y_true_sets.append(true_tags)

    y_pred_sets = []
    for pred_str in decoded_preds_text:
        # Predicted labels: split, strip, filter empty.
        predicted_tags_list = [tag.strip() for tag in pred_str.split(',') if tag.strip()]
        # Take the first 3 valid predicted tags and convert to a set.
        # This aligns with the instruction to the model to output "exactly three labels"
        # and our interest in its top 3 outputs.
        y_pred_sets.append(set(predicted_tags_list[:3])) 

    if len(y_true_sets) != len(y_pred_sets):
        # This check is more for sanity, lengths should match based on batch size
        print(f"Warning: Mismatch in number of true ({len(y_true_sets)}) and predicted ({len(y_pred_sets)}) samples.")
        # Depending on how critical this is, you might return 0 or raise an error.
        # For now, we'll proceed if possible but accuracy will be affected if lists are misaligned.
        # However, the zip below will only iterate up to the shorter list length.

    correct_matches = 0
    for true_set, pred_set in zip(y_true_sets, y_pred_sets):
        # Exact match of the sets (order-independent, counts matter)
        if true_set == pred_set:
            correct_matches += 1
            
    accuracy = correct_matches / len(y_true_sets) if len(y_true_sets) > 0 else 0.0

    # Optional: Log some examples to WandB for debugging
    if wandb.run and len(decoded_preds_text) > 0: # Check if wandb is active
        try: # Add try-except for wandb logging
            wandb.log({
                "eval/example_pred_raw_text": decoded_preds_text[0],
                "eval/example_label_raw_text": decoded_labels_text[0],
                "eval/example_pred_set_processed": str(y_pred_sets[0] if y_pred_sets else "N/A"),
                "eval/example_true_set_processed": str(y_true_sets[0] if y_true_sets else "N/A"),
            })
        except Exception as e:
            print(f"Wandb logging error in compute_metrics: {e}")


    return {'accuracy': accuracy}




def main(args):
    image_path = args.image_path
    image_text_path = args.image_text_path
    dataset_json_path = args.dataset_json_path
    env_path = args.env_file
    model_path = os.path.expanduser(args.model_path)
    checkpoint_save_path = args.checkpoint_save_path
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


    train_ds = MMHSDataset(image_path, image_text_path, dataset_json_path, f"{splits_path}/train_ids.txt")
    val_ds = MMHSDataset(image_path, image_text_path, dataset_json_path, f"{splits_path}/val_ids.txt")
    test_ds = MMHSDataset(image_path, image_text_path, dataset_json_path, f"{splits_path}/test_ids.txt")
    
    print("Created Model and Training Sets")
    


    train_args = TrainingArguments(
        output_dir=f"{checkpoint_save_path}/out_llava",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        eval_strategy="steps",
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

    data_collator_instance = AdaptedLLavaDataCollator(processor=proc) # Pass the processor
    compute_metrics_with_proc = partial(compute_metrics, proc=proc)
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator_instance,
        tokenizer=proc.tokenizer,
        compute_metrics=compute_metrics_with_proc
    )


    # train
    trainer.train()
    model.save_pretrained(f'{model_save_path}/tuned_llava')
    proc.save_pretrained(f'{model_save_path}/tuned_llava') # Save processor for easy loading later
    print(f"Fine-tuned LoRA adapters and processor saved to {args.model_save_path}")

    # extract metrics
    logs = trainer.state.log_history
    steps = [x['step'] for x in logs if 'loss' in x]
    train_loss = [x['loss'] for x in logs if 'loss' in x]
    val_loss   = [x['eval_loss'] for x in logs if 'eval_loss' in x]
    val_acc    = [x['eval_accuracy'] for x in logs if 'eval_accuracy' in x]
    eval_log_steps = [x['step'] for x in logs if 'eval_loss' in x] # Evaluation steps

    # plot losses
    plt.figure()
    plt.plot(steps, train_loss, label='train_loss')
    plt.plot(eval_log_steps, val_loss, label='val_loss')
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
        plt.savefig(f'{model_save_path}/llava_accuracy.png')

    # test set evaluation
    test_metrics = trainer.evaluate(test_dataset=test_ds)
    print('Test metrics:', test_metrics)
    with open(f'{model_save_path}/llava_test_metrics.json','w') as f:
        json.dump(test_metrics, f)

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
