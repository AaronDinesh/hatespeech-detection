import json
import gzip
import argparse
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm

def json_generator(filepath: str):
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                yield json.loads(line)
            except Exception as e:
                print(f"JSON decode error: {e}")
                raise StopIteration
            

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


            
def main(args):
    llm_output_file = args.llm_output
    ground_truth_file = args.ground_truth
    # hateful_score = args.hateful_score # Unused
    ground_truth_label_filter = args.ground_truth_label
    # llm_output_label_filter = args.llm_output_label # Unused in the matching logic below
    limit = args.limit
    image_text_folder_path = args.img_text_path
    image_folder_path = args.img_path

    total_length = sum(1 for _ in tqdm(json_generator(llm_output_file), desc="Enumerating prompts", unit=" prompts"))


    # Load ground truth
    ground_truth_df = pd.read_json(ground_truth_file, lines=False, orient='index', convert_dates=False)
    ground_truth_df['label'] = ground_truth_df['labels'].apply(label_agg) # Assumes 'labels' column contains lists
    ground_truth_df['id'] = ground_truth_df['tweet_url'].str.extract(r'/status/(\d+)')
    
    # Simple apply for img_text and img paths, no extensive error checking
    ground_truth_df['img_text_content'] = ground_truth_df['id'].apply(
        lambda row_id: load_img_text(row_id, image_text_folder_path) if pd.notnull(row_id) else None
    )
    ground_truth_df['img_path'] = ground_truth_df['id'].apply(
        lambda row_id: os.path.join(image_folder_path, f"{row_id}.jpg") if pd.notnull(row_id) else None
    )
    
    ground_truth_df_filtered = ground_truth_df[ground_truth_df['label'] == str(ground_truth_label_filter)]


    final_ids = []
    final_imgs_paths = []
    final_img_texts_content = []
    final_tweet_texts_content = []
    final_gt_labels = []
    
    items_found_count = 0

    for _, gt_row in tqdm(ground_truth_df_filtered.iterrows(), desc="Processing ground truth", total=limit):
        tweet_id_from_gt = gt_row['id']
        if pd.isnull(tweet_id_from_gt):
            continue


        for llm_line_data in tqdm(json_generator(llm_output_file), desc="Processing LLM output", total=total_length, leave=False):
            # Assuming keys exist, no error checking
            if llm_line_data['id'] == tweet_id_from_gt and \
               int(llm_line_data['response']['input_labels']) == ground_truth_label_filter:
                
                final_ids.append(gt_row['id'])
                final_imgs_paths.append(gt_row['img_path'])
                final_img_texts_content.append(gt_row['img_text_content']) # Using .get for slight safety
                final_tweet_texts_content.append(gt_row['tweet_text'])
                final_gt_labels.append(gt_row['labels_str'])
                
                items_found_count += 1
                if limit != -1 and items_found_count >= limit:
                    break 
        
        if limit != -1 and items_found_count >= limit:
            break

    print(f"Found {len(final_ids)} items to display.")

    # --- Simple Matplotlib Visualization ---
    if not final_imgs_paths:
        print("No images found to display.")
        return

    for i in range(len(final_imgs_paths)):
        img_display_path = final_imgs_paths[i]
        current_id_display = final_ids[i]
        img_text_to_display = final_img_texts_content[i]
        tweet_text_to_display = final_tweet_texts_content[i]
        gt_labels = final_gt_labels[i]

        if not img_display_path: # Minimal check for path existence
            print(f"Skipping ID {current_id_display} due to missing image path.")
            continue
        
        # NO ERROR CHECKING for imread, as requested
        img_data = mpimg.imread(img_display_path)
        
        # Use the object-oriented approach for more control
        fig, ax = plt.subplots() # Get figure and axes objects
        
        ax.imshow(img_data)
        ax.set_title(f"ID: {current_id_display}") # Use ax.set_title
        ax.axis('off') # Turn off axis numbers and ticks for the image

        # Prepare the caption text
        caption = f"Image Text: {img_text_to_display}\nTweet Text: {tweet_text_to_display}\nGround Truth Labels: {gt_labels}"

        # Add text below the image using ax.text
        ax.text(0.5, -0.05, caption, 
                ha='center', va='top',
                fontsize=16,
                transform=ax.transAxes,
                wrap=True)
        
        # Adjust layout to make room for the text.
        plt.subplots_adjust(bottom=0.2) # Increase bottom margin
                                       # Adjust this value (e.g., 0.1 to 0.3) as needed

        plt.show() # Shows one plot at a time. Close window to see the next. 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-output", type=str, required=True, help="Path to the .jsonl.gz file to validate")
    parser.add_argument("--ground-truth", type=str, required=True, help="Path to the MMHS150K_GT.json file")
    parser.add_argument("--hateful-score", action='store_true', help="Whether the output uses hateful score or not")
    parser.add_argument("--ground-truth-label", type=int, required=True, help="The ground truth label to use")
    parser.add_argument("--llm-output-label", type=int, required=True, help="The LLM output label to use")
    parser.add_argument("--limit", type=int, default=-1, help="Limit the number of comparisons to process. Use -1 for no limit")
    parser.add_argument("--img-text-path", type=str, required=True, help="Path to the MMHS150K image text")
    parser.add_argument("--img-path", type=str, required=True, help="Path to the MMHS150K images")
    
    parsed_args = parser.parse_args()
    main(parsed_args)
