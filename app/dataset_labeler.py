import tkinter as tk
from PIL import Image, ImageTk
import os
import pandas as pd
import argparse
import json
import gzip
import time


def json_generator(filepath: str):
    with open(filepath, 'rt', encoding='utf-8') as f:
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
    return res

def load_img_text(row_id, json_folder):
    file_path = os.path.join(json_folder, f"{row_id}.json")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("img_text", None)
    except Exception as e:
        return None
    

class AnnotationApp:
    def __init__(self, master, dataframe, output_path=None):
        self.master = master
        self.df = dataframe.reset_index(drop=True)
        self.index = 0
        self.annotations = {}
        self.output_path = output_path

        self.master.title("Hatefulness Annotation Tool")
        self.master.bind("<Key>", self.key_pressed)

        self.remaining_label = tk.Label(
            master,
            text="",
            font=("Arial", 24, "bold"),
            fg="blue"
        )
        self.remaining_label.pack(pady=10)


        self.image_label = tk.Label(master)
        self.image_label.pack()

        self.text1 = tk.Label(master, text="", wraplength=600, font=("Arial", 12))
        self.text1.pack(pady=5)

        self.text2 = tk.Label(master, text="", wraplength=600, font=("Arial", 12, "italic"))
        self.text2.pack(pady=5)

        self.status = tk.Label(master, text="", fg="red")
        self.status.pack(pady=5)
        self.num_images = len(self.df)

        self.display_current()

    def display_current(self):
        remaining = self.num_images - self.index
        self.remaining_label.config(text=f"Images Remaining: {remaining}")

        row = self.df.iloc[self.index]
        try:
            img = Image.open(row['img_path']).resize((400, 400))
            self.tk_image = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.tk_image)
        except Exception as e:
            self.status.config(text=f"Error loading image: {e}")
            return

        self.text1.config(text=f"Image Text: {row['img_text_content'], ''}")
        self.text2.config(text=f"Tweet: {row['tweet_text'], 'No tweet available'}")
        self.status.config(text="Press 0, 1, 2, or 3 to annotate.")

    def key_pressed(self, event):
        if event.char in ['0', '1', '2', '3']:
            label = int(event.char)
            row = self.df.iloc[self.index]
            record = {
                "id": row['id'],
                "img_path": row['img_path'],
                "img_text_content": row.get('img_text_content', ''),
                "text": row.get('text', ''),
                "label": label
            }

            # Write to JSONL file (append mode)
            with open(self.output_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

            self.status.config(text=f"Saved label {label}. Moving to next.")
            self.index += 1

            if self.index < len(self.df):
                self.display_current()
            else:
                self.remaining_label.config(text="Images Remaining: 0")
                self.status.config(text="All images annotated. Press any key to exit.")
                self.image_label.config(image='')
                self.text1.config(text='')
                self.text2.config(text='')
                self.master.unbind("<Key>")  # Prevent further input from being processed
                self.master.bind("<Key>", lambda e: self.master.destroy())  # Bind any key to close
        else:
            self.status.config(text="Invalid key. Please press 0, 1, 2, or 3.")


def main(args):
    dataset_path = args.dataset_path
    img_path = args.img_path
    img_text_path = args.img_text_path
    output_path = args.output_path
    limit = args.limit
    SEED = 42

    completed_annotations = set()
    
    if os.path.exists(output_path):
        for line in json_generator(output_path):
            completed_annotations.add(line['id'])

    if limit - len(completed_annotations) <= 0:
        print(f"You have annotated more images than the limit specified. Please increase the limit to above {len(completed_annotations)} to annotate more images.") 
        return

    if len(completed_annotations) > 0:
        print(f"You have already annotated {len(completed_annotations)} images. You have {limit - len(completed_annotations)} images to annotate.")
    else:
        print(f"You have not annotated any images yet. You have {limit} images to annotate.")

    ground_truth_df = pd.read_json(dataset_path, lines=False, orient='index', convert_dates=False)
    ground_truth_df['label'] = ground_truth_df['labels'].apply(label_agg) # Assumes 'labels' column contains lists
    ground_truth_df['id'] = ground_truth_df['tweet_url'].str.extract(r'/status/(\d+)')
    
    # Simple apply for img_text and img paths, no extensive error checking
    ground_truth_df['img_text_content'] = ground_truth_df['id'].apply(
        lambda row_id: load_img_text(row_id, img_text_path) if pd.notnull(row_id) else None
    )
    ground_truth_df['img_path'] = ground_truth_df['id'].apply(
        lambda row_id: os.path.join(img_path, f"{row_id}.jpg") if pd.notnull(row_id) else None
    )

    subset_df = ground_truth_df.sample(n=limit, random_state=SEED)
    subset_df = subset_df[subset_df['id'].isin(completed_annotations) == False]

    root = tk.Tk()
    app = AnnotationApp(root, subset_df, output_path=output_path)
    root.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, help="Path to the MMHS150K_GT.json file")
    parser.add_argument("--img-path", type=str, help="Path to the images")
    parser.add_argument("--img-text-path", type=str, help="Path to the image text file")
    parser.add_argument("--output-path", type=str, help="Path to the output JSONL file")
    parser.add_argument("--limit", type=int, default=100, help="Number of images to annotate")
    args = parser.parse_args()
    main(args)

