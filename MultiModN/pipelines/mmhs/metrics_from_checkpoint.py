import torch
import argparse
import pickle as pkl
from os import path as o
import os
import sys
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "../..")))
from pathlib import Path
import pandas as pd

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from datasets.mmhs import preprocessing, data_splitting, MMHSDataset
from multimodn.multimodn import MultiModN
from multimodn.encoders import resnet_encoder, LSTMTextEncoder
from multimodn.decoders import ClassDecoder
from multimodn.history import MultiModNHistory
"""
Sample use:
python metrics_from_checkpoint.py --ckpt ./checkpoint/step00000025.pt --save-csv
"""


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from {args.ckpt}")
    payload = torch.load(args.ckpt, map_location=device)

    # Setup model architecture
    state_size = 1024
    image_encoder = resnet_encoder.ResNet(state_size=state_size, freeze=True)
    text_encoder = LSTMTextEncoder(state_size)
    tweet_encoder = LSTMTextEncoder(state_size)
    encoders = [image_encoder, text_encoder, tweet_encoder]

    n_labels = 4
    batch_size = 2048
    decoders = [ClassDecoder(state_size, n_labels, activation=torch.nn.Softmax(dim=1))]
    model = MultiModN(state_size, encoders, decoders, 0.8, 0.2, device=device)

    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()

    print("✓ Model loaded successfully")

    # Load data
    path_to_mmhs = '../../../MMHS150K/'
    pickle_path = os.path.join(path_to_mmhs, "df.pkl")

    if os.path.exists(pickle_path):
        df = pd.read_pickle(pickle_path)
    else:
        df = preprocessing(path_to_mmhs) 
        df.to_pickle(pickle_path)

    _,test_df, _ = data_splitting(df, path_to_mmhs)
    test_dataset = MMHSDataset(test_df, root_dir=path_to_mmhs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    print(f"Loaded test set with {len(test_dataset)} samples")

    # Compute metrics
    criterion = CrossEntropyLoss()
    history = MultiModNHistory(targets=["label"])
    print("📊 Evaluating model on test set...")
    # metrics = model.test(val_loader, criterion, history, tag="val", log_results=True)
    metrics, preds = model.test_mmhs(test_loader, criterion, history, tag='test')
    # f1, acc, mae =metrics
    print("✅ Evaluation complete. Final metrics:")
    print(metrics)
    # history.print_results()

    if args.save_csv:
        results_dir = "results"
        Path(results_dir).mkdir(exist_ok=True)
        # save per-sample predictions
        df = test_df.reset_index(drop=True).copy()
        df["pred_label"] = preds
        preds_path = f"{results_dir}/val_predictions_from_{Path(args.ckpt).stem}.csv"
        df.to_csv(preds_path, index=False)
        print(f"Saved per-sample predictions to {preds_path}")

        # summary_txt =  f"{results_dir}/val_summary_{Path(args.ckpt).stem}.txt"
        # with open(summary_txt, "w") as out:
        #     out.write(metrics)
        # print(f"Saved scalar summary to {summary_txt}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--save-csv", action="store_true", help="Save results to CSV")

    args = parser.parse_args()
    main(args)
