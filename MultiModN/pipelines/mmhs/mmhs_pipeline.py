import sys
import os
from os import path as o
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "../..")))

from tqdm.auto import trange
import torch
from torch.optim.lr_scheduler import LinearLR  
from torch.optim.lr_scheduler import SequentialLR
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from multimodn.multimodn import MultiModN
# from multimodn.encoders import LSTMEncoder
from multimodn.encoders import resnet_encoder
from multimodn.encoders import LSTMTextEncoder
from multimodn.decoders import LogisticDecoder
from multimodn.history import MultiModNHistory
from datasets.mmhs import preprocessing, data_splitting, MMHSDataset
from multimodn.decoders import ClassDecoder      # or MLPDecoder
from pipelines import utils
import torch.nn.functional as F
import pickle as pkl
import glob
import re
from pathlib import Path
import wandb
# from dotenv import load_dotenv
import pandas as pd
    
def main():
    PIPELINE_NAME = utils.extract_pipeline_name(sys.argv[0])
    print('Running ̣{}...'.format(utils.get_display_name(PIPELINE_NAME)))
    args = utils.parse_args()
    # load_dotenv(args.env_path)
    # 1) Point to your file (for example, ~/.config/wandb_api_key.txt)
    key_file = "./api_keys/api_key.txt"

    # 2) Read and strip any whitespace/newlines
    api_key = key_file.read_text().strip()
    os.environ["WANDB_API_KEY"] = api_key
    wandb_logger = wandb.init(project="multimodn", name="multimodn-run")

    torch.manual_seed(args.seed)

    device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
    print('using gpu:', torch.cuda.is_available())

    features = ['image', 'img_text', 'tweet_text']
    
    targets = ['label']

    # Batch size: set 0 for full batch
    batch_size = 1024

    # Representation state size
    state_size = 512

    learning_rate = 0.1
    epochs = 5 if not args.epoch else args.epoch

    ckpt_dir          = "./checkpoint2"
    ckpt_every_iter   = 5                # iterations, not epochs
    os.makedirs(ckpt_dir, exist_ok=True)

    global_step = 0                       # will count mini-batches

    ##############################################################################
    ###### Create dataset and data loaders
    ##############################################################################
    path_to_mmhs = '../../../MMHS150K/'
    pickle_path = os.path.join(path_to_mmhs, "rebalanced_df.pkl")

    if os.path.exists(pickle_path):
        dataset = pd.read_pickle(pickle_path)
    else:
        dataset = preprocessing(path_to_mmhs) 
        dataset.to_pickle(pickle_path)

    print('Loaded data:')
    train_data, test_data, val_data = data_splitting(dataset, path_to_mmhs)
    print('Split data')
    train_dataset = MMHSDataset(train_data, root_dir=path_to_mmhs)
    val_dataset = MMHSDataset(val_data, root_dir=path_to_mmhs)
    # test_dataset = MMHSDataset(test_data, root_dir=path_to_mmhs)
    # breakpoint()
    #datasplit = (0.8, 0.2, 0)
    #target_idx_to_balance = 0 # Balance 'Survived' during split
    #train_data, val_data, test_data = dataset.random_split(datasplit, args.seed, target_idx_to_balance)

    if batch_size == 0:
        batch_size_train = len(train_data)
        batch_size_val = len(val_data)
        # batch_size_test = len(test_data)
    else:
        batch_size_train = batch_size
        batch_size_val = batch_size
        # batch_size_test = batch_size

    train_loader = DataLoader(train_dataset, batch_size_train, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size_val, num_workers=8)
    #test_loader = DataLoader(test_dataset, batch_size=batch_size_test)
    print('Loaded DataLoaders')

    ##############################################################################
    ###### Set encoder and decoders
    ##############################################################################
    image_encoder = resnet_encoder.ResNet(state_size=state_size, freeze=True)
    text_encoder   = LSTMTextEncoder(state_size)
    tweet_encoder = LSTMTextEncoder(state_size)
    
    encoders = [image_encoder, text_encoder, tweet_encoder]
    n_labels = 4            # 0,1,2,3
    decoders = [ClassDecoder(state_size, n_labels, activation=lambda x: x)]

    model = MultiModN(state_size, encoders, decoders, 0.7, 0.3, device = device)
    print('loaded Encoders and Decoders')
    optimizer = torch.optim.Adam(list(model.parameters()), learning_rate)

    warmup_iters = 5
    scheduler = LinearLR(
        optimizer,
        start_factor=1,          # begin at 10 % of base LR
        end_factor=0.1,
        total_iters=epochs - warmup_iters
    )
    warmup = LinearLR(
        optimizer,
        start_factor=0.1,          # ramps 0.1 → 1.0 over warm-up
        end_factor=1.0,
        total_iters=warmup_iters
    )
    # Use SequentialLR to chain them
    scheduler = SequentialLR(optimizer, schedulers=[warmup, scheduler],
                            milestones=[warmup_iters])

    criterion = CrossEntropyLoss()

    history = MultiModNHistory(targets)


    ckpt_dir = "./checkpoint"
    Path(ckpt_dir).mkdir(exist_ok=True)

    ckpts = sorted(
        glob.glob(f"{ckpt_dir}/step*.pt"),
        key=lambda f: int(re.findall(r"step(\d+).pt", f)[0])
    )

    start_epoch = 0          # default: begin fresh
    if ckpts:
        last_ckpt = ckpts[-1]
        payload   = torch.load(last_ckpt, map_location=device)

        # restore weights / optimiser / scheduler
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optim"])
        if "sched" in payload and scheduler is not None:
            scheduler.load_state_dict(payload["sched"])

        # restore the running iteration counter used by train_epoch()
        model._global_step = payload.get("step", 0)

        # optionally restore epoch if you stored it
        start_epoch = payload.get("epoch", 0) + 1

        print(f"Resumed from {last_ckpt} (step {model._global_step})")
    else:
        model._global_step = 0
        print("➤  No checkpoint found – starting fresh")

    ##############################################################################
    ###### Train and Test model
    ##############################################################################
    for _ in trange(epochs):
        model.train_epoch_mmhs(
            train_loader,
            optimizer,
            criterion,
            history,
            checkpoint_dir="./checkpoint",
            checkpoint_every=5,
            log_interval=1
        )
        model.test(val_loader, criterion, history, tag='val')
        scheduler.step()

    ##############################################################################
    ###### Store model and history
    ##############################################################################
    directory = o.join(o.dirname(os.path.realpath(__file__)), 'models')

    if args.save_model:
        if not o.exists(directory):
            os.makedirs(directory)
        model_path = o.join(directory, PIPELINE_NAME + '_model.pkl')
        pkl.dump(model, open(model_path, 'wb'))

    if args.save_history:
        if not o.exists(directory):
            os.makedirs(directory)
        history_path = o.join(directory, PIPELINE_NAME + '_history.pkl')
        pkl.dump(history, open(history_path, 'wb'))

    ##############################################################################
    ###### Save learning curves
    ##############################################################################
    if args.save_plot:
        directory = o.join(o.dirname(os.path.realpath(__file__)), 'plots')
        if not o.exists(directory):
            os.makedirs(directory)
        plot_path = o.join(directory, PIPELINE_NAME + '.png')

        targets_to_display = targets

        history.plot(plot_path, targets_to_display, show_state_change=False)

    ##############################################################################
    ###### Display results and save them
    ##############################################################################
    if args.save_results:
        directory = o.join(o.dirname(os.path.realpath(__file__)), 'results')
        if not o.exists(directory):
            os.makedirs(directory)
        results_path = o.join(directory, PIPELINE_NAME + '.csv')

        history.print_results()
        history.save_results(results_path)

if __name__ == "__main__":
    main()
