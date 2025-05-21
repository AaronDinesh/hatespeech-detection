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
        self.dataset_full_df = pd.read_json(self.dataset_json_path, lines=False, orient='index', convert_dates=False, convert_axes=False, dtype=str)


        with open(self.data_split_ids_path, "r") as f:
            split_image_ids_int = [int(line.strip()) for line in f]
        
        split_image_ids_str = [str(id_val) for id_val in split_image_ids_int]

        

        self.dataset_df = self.dataset_full_df[self.dataset_full_df.index.isin(split_image_ids_str)]
        self.dataset_df.reset_index(inplace=True)
        self.data_length = len(self.dataset_df)
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
    # Filter out None items
    batch = [b for b in batch if b is not None]
    if not batch:
        # Return an empty batch dictionary or raise an error
        return {
            'input_ids': torch.empty(0, dtype=torch.long),
            'attention_mask': torch.empty(0, dtype=torch.long),
            'pixel_values': torch.empty(0, dtype=torch.float),
            'labels': torch.empty(0, dtype=torch.long)
            # Potentially 'pixel_attention_mask': torch.empty(0, dtype=torch.long)
        }

    # Pad input_ids and create attention_mask for text
    input_ids = pad_sequence(
        [b['input_ids'] for b in batch],
        batch_first=True,
        padding_value=proc.tokenizer.pad_token_id
    )
    attention_mask = pad_sequence(
        [b['attention_mask'] for b in batch],
        batch_first=True,
        padding_value=0  # Padding for attention_mask is 0
    )

    # Pad labels
    labels = pad_sequence(
        [b['labels'] for b in batch],
        batch_first=True,
        padding_value=proc.tokenizer.pad_token_id
    )
    labels[labels == proc.tokenizer.pad_token_id] = -100 # Mask padding tokens in labels

    # --- Handle pixel_values padding ---
    pixel_values_list = [b['pixel_values'] for b in batch]

    if not pixel_values_list:
        pixel_values = torch.empty(0, dtype=torch.float) # Or appropriate dtype
        pixel_attention_mask = torch.empty(0, dtype=torch.long)
    else:
        # Determine the maximum number of patches/tokens in this batch
        # pixel_values_list contains tensors of shape [N_i, C, H, W] where N_i can vary
        max_num_patches = 0
        for pv_tensor in pixel_values_list:
            if pv_tensor.ndim < 3: # Expecting at least [C,H,W] or [N,C,H,W]
                print(f"Warning: Unexpected pixel_values shape {pv_tensor.shape} in batch. Skipping or erroring.")
                # Decide how to handle this, e.g. skip this item or raise error
                continue # For now, let's assume they are mostly correct
            max_num_patches = max(max_num_patches, pv_tensor.shape[0] if pv_tensor.ndim == 4 else 1)

        if max_num_patches == 0 and batch : # If all items were skipped or had bad shapes
             raise ValueError("Could not determine max_num_patches from pixel_values in the batch.")


        # Assuming C, H, W are the same for all. Get them from the first valid tensor.
        # Find first valid tensor to get C, H, W and dtype, device
        first_valid_pv = None
        for pv in pixel_values_list:
            if pv.ndim == 4 and pv.shape[0] > 0: # [N, C, H, W]
                first_valid_pv = pv
                break
            elif pv.ndim == 3 and max_num_patches == 1 : # [C, H, W] implies N=1
                first_valid_pv = pv.unsqueeze(0) # Make it [1, C, H, W] for consistency
                break
        
        if first_valid_pv is None and batch:
            raise ValueError("No valid pixel_values tensors found in batch to determine C, H, W.")

        # If first_valid_pv is still None here, it means pixel_values_list was empty or all bad
        # (already handled by the `if not pixel_values_list` check, but being thorough)

        channels = first_valid_pv.shape[1]
        height = first_valid_pv.shape[2]
        width = first_valid_pv.shape[3]
        pv_dtype = first_valid_pv.dtype
        pv_device = first_valid_pv.device
        
        padded_pixel_values_tensors = []
        pixel_attention_mask_tensors = []
        
        # Define padding value for pixel values (usually 0 for images/features)
        # Check LLaVA's image processor config if a specific value is used (e.g., processor.image_processor.image_mean if normalizing and then padding)
        # For raw pixel values or features, 0.0 is common.
        pixel_padding_value = 0.0

        for pv_tensor in pixel_values_list:
            current_pv_processed = pv_tensor
            if pv_tensor.ndim == 3 and max_num_patches == 1 : # Was [C,H,W], treat as [1,C,H,W]
                current_pv_processed = pv_tensor.unsqueeze(0)
            elif pv_tensor.ndim != 4:
                print(f"Warning: Skipping item with pixel_values shape {pv_tensor.shape} as it's not 3D or 4D.")
                # This item will effectively be dropped from the batch for pixel values.
                # This is not ideal; __getitem__ should ensure consistent ndim or this collate fn
                # needs to be more robust to truly problematic shapes.
                # For now, let's assume this tensor should have been [max_num_patches, C,H,W]
                # and create a fully padded one. This is a guess.
                # A better solution is to fix __getitem__ or ensure all inputs are valid.
                # current_pv_processed = torch.full((0, channels, height, width), pixel_padding_value, dtype=pv_dtype, device=pv_device)
                # This tensor would then be fully padded below. This is just one strategy.
                # A simpler one for now: if shape is bad, this sample might be problematic for pixel values.
                # How to handle depends on how many such bad samples.
                # For now, this loop assumes pv_tensor is [N_i, C, H, W] or [C,H,W]
                pass # This tensor will not be added to padded_pixel_values_tensors if not processed

            num_current_patches = current_pv_processed.shape[0]
            padding_needed = max_num_patches - num_current_patches
            
            if padding_needed >= 0 : # Only proceed if padding_needed is not negative (sanity check)
                if padding_needed > 0:
                    padding_shape = (padding_needed, channels, height, width)
                    padding = torch.full(padding_shape, pixel_padding_value, dtype=pv_dtype, device=pv_device)
                    final_pv = torch.cat((current_pv_processed, padding), dim=0)
                else:
                    final_pv = current_pv_processed
                
                padded_pixel_values_tensors.append(final_pv)
                
                # Create attention mask for these patches (1 for real, 0 for padded)
                current_mask = torch.ones(num_current_patches, dtype=torch.long, device=pv_device)
                padding_mask_for_patches = torch.zeros(padding_needed, dtype=torch.long, device=pv_device)
                pixel_attention_mask_tensors.append(torch.cat((current_mask, padding_mask_for_patches), dim=0))
            else:
                 print(f"Warning: Negative padding_needed for a pixel_value tensor. Original shape {pv_tensor.shape}, max_patches {max_num_patches}")


        if padded_pixel_values_tensors:
            pixel_values = torch.stack(padded_pixel_values_tensors)
            pixel_attention_mask = torch.stack(pixel_attention_mask_tensors)
        else: # If all items had problematic pixel_values
            # Create empty tensors with expected dimensions if possible
            # Note: C, H, W might not be known if all pixel_values were bad.
            # This case should ideally be prevented by robust __getitem__ or earlier checks.
            num_channels_fallback = 3 # Common default
            height_fallback = proc.image_processor.size['height'] if hasattr(proc, 'image_processor') else 336
            width_fallback = proc.image_processor.size['width'] if hasattr(proc, 'image_processor') else 336

            pixel_values = torch.empty((0, max_num_patches if max_num_patches > 0 else 1, num_channels_fallback, height_fallback, width_fallback), dtype=torch.float)
            pixel_attention_mask = torch.empty((0, max_num_patches if max_num_patches > 0 else 1), dtype=torch.long)


    # Prepare the batch dictionary
    batch_dict = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'pixel_values': pixel_values,
        'labels': labels
    }

    # Add pixel_attention_mask if it was created and LLaVA model uses it
    # The LLaVA model's forward pass needs to accept an argument like `pixel_attention_mask` or `image_attention_mask`.
    if 'pixel_attention_mask' in locals() and pixel_attention_mask.nelement() > 0 : # Check if it was created and is not empty
        batch_dict['pixel_attention_mask'] = pixel_attention_mask

    return batch_dict

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


    train_ds = MMHSDataset(image_path, image_text_path, dataset_json_path, f"{splits_path}/train_ids.txt", proc)
    val_ds = MMHSDataset(image_path, image_text_path, dataset_json_path, f"{splits_path}/val_ids.txt", proc)
    test_ds = MMHSDataset(image_path, image_text_path, dataset_json_path, f"{splits_path}/test_ids.txt", proc)
    
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

    collate_func = partial(collate_fn, proc=proc)
    compute_metrics_with_proc = partial(compute_metrics, proc=proc)
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_func,
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
