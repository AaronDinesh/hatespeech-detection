import random
import torch
from torch.optim import Optimizer
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
import os 

from multimodn.encoders.multimod_encoder import MultiModEncoder
from multimodn.decoders.multimod_decoder import MultiModDecoder
from multimodn.state import InitState, TrainableInitState
from multimodn.history import MultiModNHistory
from typing import List, Optional, Iterable, Tuple, Callable, Union
import torch.nn as nn
import numpy as np
from torchsummary import summary

from torchmetrics import ConfusionMatrix, F1Score, ROC, PrecisionRecallCurve, Accuracy, AUROC, MeanAbsoluteError

performance_metrics = ['f1', 'auc',  'accuracy', 'sensitivity', 'specificity', 'fpr', 'tpr', 'precision', 'recall', \
    'tn', 'fp', 'fn', 'tp', 'thr_roc', 'thr_pr']

# Calculation of various performance metrics for the binary classification task (default)
# def get_performance_metrics(y_true, y_pred, y_prob):
#     f1_score = F1Score(task="binary", average = 'macro')
#     roc = ROC(task="binary")
#     pr_curve = PrecisionRecallCurve(task="binary")
#     accuracy_score = Accuracy(task="binary")
#     roc_auc_score = AUROC(task="binary", average = 'macro')
#     confmat = ConfusionMatrix(task="binary")
    
#     cm = confmat(y_pred, y_true)
#     tp = cm[1][1] 
#     fp = cm[0][1]
#     fn = cm[1][0]
#     tn = cm[0][0]
    
#     if (tp + fn) != 0:
#         sensitivity = tp / (tp + fn)
#     else:
#         sensitivity = 0
#     if (tn + fp) != 0:
#         specificity = tn / (tn + fp)   
#     else:
#         specificity = 0
        
#     fpr, tpr, thr_roc = roc(y_prob, y_true)   
#     precision, recall, thr_pr = pr_curve(y_prob, y_true)   
    
#     return f1_score(y_prob, y_true), roc_auc_score(y_prob, y_true), accuracy_score(y_pred, y_true), sensitivity, specificity, fpr, tpr, precision, recall, \
#       tn, fp, fn, tp, thr_roc, thr_pr

def get_performance_metrics(
    y_true: Tensor, 
    y_pred: Tensor
) -> Tuple[float, float]:
    """
    Computes macro-F1 and accuracy for a 4-class classification task.

    Args:
        y_true:   LongTensor of shape [N] with values in {0,1,2,3}
        y_pred:   LongTensor of shape [N] with predicted classes

    Returns:
        (f1_macro, accuracy) as Python floats
    """
    # initialize metrics
    f1 = F1Score(task="multiclass", num_classes=4, average="macro").to(y_true.device)
    acc = Accuracy(task="multiclass", num_classes=4).to(y_true.device)
    mae_m = MeanAbsoluteError().to(y_true.device)

    # compute
    f1_score = f1(y_pred, y_true)
    accuracy = acc(y_pred, y_true)
    mae = mae_m(y_pred, y_true)

    return f1_score.item(), accuracy.item(), mae.item()

def compute_metrics(tp, tn, fp, fn, cm, enc_idx, dec_idx):
    if cm is not None:
        # In torch cm fp are in rows, fn are in columns
        # According to https://torchmetrics.readthedocs.io/en/stable/classification/confusion_matrix.html
        tp[enc_idx][dec_idx] += cm[1][1]
        tn[enc_idx][dec_idx] += cm[0][0]
        fp[enc_idx][dec_idx] += cm[0][1]
        fn[enc_idx][dec_idx] += cm[1][0]
    else:
        tp[enc_idx][dec_idx] = float('nan')
        tn[enc_idx][dec_idx] = float('nan')
        fp[enc_idx][dec_idx] = float('nan')
        fn[enc_idx][dec_idx] = float('nan')

class MultiModN(nn.Module):
    def __init__(
            self,
            state_size: int,
            encoders: List[MultiModEncoder],
            decoders: List[MultiModDecoder],
            err_penalty: float,
            state_change_penalty: float,
            shuffle_mode: Optional[bool] = False,
            init_state: Optional[InitState] = None,
            device: Optional[torch.device] = None,
    ):
        super(MultiModN, self).__init__()
        self.shuffle_mode = shuffle_mode
        self.device = device if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.init_state = TrainableInitState(
            state_size, self.device) if not init_state else init_state
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)
        self.err_penalty = err_penalty
        self.state_change_penalty = 0.01 * state_change_penalty
        self.to(self.device)  # Move to device

    def train_epoch(
            self,
            train_loader: DataLoader,
            optimizer: Optimizer,
            criterion: Union[nn.Module, Callable],
            history: Optional[MultiModNHistory] = None,
            log_interval: Optional[int] = None,
            logger: Optional[Callable] = None,
            last_epoch: Optional[bool] = False,
            checkpoint_dir: Optional[str] = None,
            checkpoint_every: int = 5,
    ) -> None:
        # If log interval is given and logger is not, use print as default logger
        if log_interval and not logger:
            logger = print
        self.train()

        n_batches = len(train_loader)
        n_samples_epoch = np.ones((len(self.encoders) + 1, 1))

        err_loss_epoch = np.zeros((len(self.encoders) + 1, len(self.decoders)))
        state_change_epoch = np.zeros(len(self.encoders))
        n_correct_epoch = np.zeros((len(self.encoders) + 1, len(self.decoders)))

        # For computation of sensitivity, specificity and balanced accuracy
        tp_epoch = torch.zeros((len(self.encoders) + 1, len(self.decoders))).to(self.device)
        tn_epoch = torch.zeros((len(self.encoders) + 1, len(self.decoders))).to(self.device)
        fp_epoch = torch.zeros((len(self.encoders) + 1, len(self.decoders))).to(self.device)
        fn_epoch = torch.zeros((len(self.encoders) + 1, len(self.decoders))).to(self.device)
        # ------------------------------------------------------------------
        # initialise (or retrieve) running iteration counter
        if not hasattr(self, "_global_step"):
            self._global_step = 0
        # ------------------------------------------------------------------
        for batch_idx, batch in tqdm(enumerate(train_loader),leave = False):
            # Note: for multiclass target should be = [0, n_classes -1] for the correctness of CrossEntropyLoss
            data, target, encoder_sequence = (list(batch) + [None])[:3]
            batch_size = target.shape[0]
            n_samples_epoch[0] += batch_size

            err_loss = torch.zeros((len(self.encoders) + 1, len(self.decoders)))
            state_change = torch.zeros(len(self.encoders))

            # Collect metrics from each step
            tp = torch.zeros((len(self.encoders)+1, len(self.decoders))).to(self.device)
            tn = torch.zeros((len(self.encoders)+1, len(self.decoders))).to(self.device)
            fp = torch.zeros((len(self.encoders)+1, len(self.decoders))).to(self.device)
            fn = torch.zeros((len(self.encoders)+1, len(self.decoders))).to(self.device)

            data_encoders = [data_encoder.to(self.device) for data_encoder in data]
            
            target = target.type(torch.LongTensor)
            target = target.to(self.device)

            optimizer.zero_grad()

            state: Tensor = self.init_state(batch_size)

            for dec_idx, decoder in enumerate(self.decoders):
                target_decoder = target[:, dec_idx]
                output_decoder = decoder(state)
                _, prediction = torch.max(output_decoder, dim=1)

                err_loss[0][dec_idx] = criterion(output_decoder, target_decoder)
                n_correct_epoch[0][dec_idx] += sum(prediction == target_decoder).float()

                # Each decoder can possibly solve different task, so we redefine confusion matrices for each decoder,
                # and store them in list for the next encoding steps
                # Now, for simplicity we only calculate confusion matrix for deocders with binary tasks
                cm = None
                if decoder.n_classes == 2:
                    confmat = ConfusionMatrix(task="binary", num_classes=2).to(self.device)
                    cm = confmat(prediction, target_decoder)

                compute_metrics(tp, tn, fp, fn, cm, 0, dec_idx)

            for data_idx, enc_idx in self.get_encoder_iterable(encoder_sequence,
                                                               shuffle_mode=self.shuffle_mode,
                                                               train=True):
                encoder = self.encoders[enc_idx]
                data_encoder = data_encoders[data_idx]

                old_state = state.clone()

                # Skip encoder if data contains nan value
                if any(data_encoder.isnan().flatten()):
                    continue

                n_samples_epoch[enc_idx + 1] += batch_size

                state = encoder(state, data_encoder)
                state_change[enc_idx] = torch.mean((state - old_state) ** 2)

                for dec_idx, decoder in enumerate(self.decoders):
                    target_decoder = target[:, dec_idx]
                    output_decoder = decoder(state)
                    _, prediction = torch.max(output_decoder, dim=1)

                    err_loss[enc_idx + 1][dec_idx] = criterion(output_decoder,
                                                               target_decoder)
                    n_correct_epoch[enc_idx + 1][dec_idx] += sum(
                        prediction == target_decoder)

                    cm = None
                    if decoder.n_classes == 2:
                        confmat = ConfusionMatrix(task="binary", num_classes=2).to(self.device)
                        cm = confmat(prediction, target_decoder)

                    compute_metrics(tp, tn, fp, fn, cm, enc_idx+1, dec_idx)

            # Global losses (combining all encoders and decoders) at batch level
            global_err_loss = torch.sum(err_loss) / (
                    len(self.decoders) * (len(self.encoders) + 1))
            global_state_change = torch.sum(state_change) / (len(self.encoders))
            # Loss = global_err_loss * err_penalty +
            #        0.01 * global_state_change * state_change_penalty
            loss = (
                global_err_loss * self.err_penalty +
                global_state_change * self.state_change_penalty
            )
            loss.backward()
            optimizer.step()

            self._global_step += 1

            # ───────────────── checkpoint every N iterations ──────────────────
            if checkpoint_dir and (self._global_step % checkpoint_every == 0):
                os.makedirs(checkpoint_dir, exist_ok=True)
                ckpt_path = os.path.join(
                    checkpoint_dir, f"step{self._global_step:08d}.pt"
                )
                torch.save(
                    {
                        "step":  self._global_step,
                        "model": self.state_dict(),
                        "optim": optimizer.state_dict(),
                    },
                    ckpt_path,
                )
                if logger:
                    logger(f"💾  saved checkpoint → {ckpt_path}")
            # ------------------------------------------------------------------

            err_loss_epoch += err_loss.detach().numpy()
            state_change_epoch += state_change.detach().numpy()

            tp_epoch += tp
            fp_epoch += fp
            fn_epoch += fn
            tn_epoch += tn

            if log_interval and batch_idx % log_interval == log_interval - 1:
                logger(
                    f"Batch {batch_idx + 1}/{n_batches}\n"
                    f"\tLoss: {loss.item():.4f}\n"
                    f"\tErr loss: {global_err_loss.item():.4f}\n"
                    f"\tState change: {global_state_change.item():.4f}"
                )

        err_loss_epoch /= n_batches
        state_change_epoch /= n_batches
        accuracy_epoch = n_correct_epoch / n_samples_epoch

        # Compute metrics for the current epoch
        # Use np.where to avoid NaNs, set the whole metric to zero
        # in case of the equality of denominator to zero
        # At the end move all metrics to cpu and convert to numpy for history

        #Note, that here we compute metrics for all encoders and decoders, \
        # and at the history file select the last encoder for the final metric

        sensitivity_denominator = tp_epoch + fn_epoch
        sensitivity_epoch = torch.where(sensitivity_denominator == 0, 0,
                                     tp_epoch / sensitivity_denominator).detach().cpu().numpy()

        specificity_denominator = tn_epoch + fp_epoch
        specificity_epoch = torch.where(specificity_denominator == 0, 0,
                                     tn_epoch / specificity_denominator).detach().cpu().numpy()

        balanced_accuracy_epoch = (sensitivity_epoch + specificity_epoch) / 2

        if history is not None:
            history.state_change_loss.append(state_change_epoch)
            history.loss['train'].append(err_loss_epoch)
            history.accuracy['train'].append(accuracy_epoch)
            history.sensitivity['train'].append(sensitivity_epoch)
            history.specificity['train'].append(specificity_epoch)
            history.balanced_accuracy['train'].append(balanced_accuracy_epoch)
        if last_epoch: 
            return self.test(train_loader, criterion, history = None)       


    def train_epoch_mmhs( 
        self,
        train_loader: DataLoader,
        optimizer: Optimizer,
        criterion: Union[nn.Module, Callable],
        history: Optional[MultiModNHistory] = None,
        log_interval: Optional[int] = None,
        logger: Optional[Callable] = None,
        last_epoch: Optional[bool] = False,
        checkpoint_dir: Optional[str] = None,
        checkpoint_every: int = 5,
        wandb_logger = None
    ):
        # If log interval is given and logger is not, use print as default logger
        if log_interval and not logger:
            logger = print
        self.train()

        # Prepare AMP scaler
        scaler = torch.cuda.amp.GradScaler()

        # Pre-epoch accumulators (on GPU)
        n_stages = len(self.encoders) + 1
        n_classes = self.decoders[0].n_classes
        n_batches = len(train_loader)

        n_samples_epoch     = torch.ones((n_stages,1), device=self.device)
        err_loss_epoch      = torch.zeros((n_stages,1), device=self.device)
        n_correct_epoch     = torch.zeros((n_stages,1), device=self.device)
        state_change_epoch  = torch.zeros(n_stages - 1, device=self.device)

        if not hasattr(self, "_global_step"):
            self._global_step = 0

        for batch_idx, batch in enumerate(train_loader):
            data, target, encoder_sequence = (list(batch) + [None])[:3]
            data = [d.to(self.device, non_blocking=True) for d in data]
            target = target[:,0].to(self.device, non_blocking=True)  # single-decoder
            batch_size = target.shape[0]
            n_samples_epoch[0] += batch_size
            optimizer.zero_grad()

            # Zero out per-batch metrics
            err_loss       = torch.zeros((n_stages,1), device=self.device)
            state_change   = torch.zeros(n_stages - 1, device=self.device)

            with torch.cuda.amp.autocast():
                states = [self.init_state(batch_size)]

                output_decoder = self.decoders[0](states[-1])
                _, prediction = torch.max(output_decoder, dim=1)
                err_loss[0][0] = criterion(output_decoder, target)
                n_correct_epoch[0][0] += sum(prediction == target).float()
                                
                
                for enc_idx, data_idx in self.get_encoder_iterable(encoder_sequence, shuffle_mode=self.shuffle_mode, train=True):
                    n_samples_epoch[enc_idx + 1] += batch_size
                    new_state = self.encoders[enc_idx](states[-1], data[data_idx])
                    state_change[enc_idx] = torch.mean(torch.abs(new_state - states[-1]))
                    output_decoder = self.decoders[0](new_state)
                    _, prediction = torch.max(output_decoder, dim=1)
                    err_loss[enc_idx + 1][0] = criterion(output_decoder, target)
                    n_correct_epoch[enc_idx + 1][0] += sum(prediction == target).float()
                    states.append(new_state)
                
            global_err_loss = torch.sum(err_loss) / (len(self.decoders) * (len(self.encoders) + 1))
            global_state_change = torch.sum(state_change) / len(self.encoders)
            loss = (global_err_loss * self.err_penalty + global_state_change * self.state_change_penalty) 
            
        
            # 4) Backward with AMP
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            self._global_step += 1


            # ───────────────── checkpoint every N iterations ──────────────────
            if checkpoint_dir and (self._global_step % checkpoint_every == 0):
                os.makedirs(checkpoint_dir, exist_ok=True)
                ckpt_path = os.path.join(
                    checkpoint_dir, f"step{self._global_step:08d}.pt"
                )
                torch.save(
                    {
                        "step":  self._global_step,
                        "model": self.state_dict(),
                        "optim": optimizer.state_dict(),
                    },
                    ckpt_path,
                )
                if logger:
                    logger(f"💾  saved checkpoint → {ckpt_path}")
            # ------------------------------------------------------------------
            
            err_loss_epoch     += err_loss.detach()
            state_change_epoch += state_change.detach()
            accuracy_epoch = n_correct_epoch / n_samples_epoch  # this is a tensor/array
            overall_acc   = accuracy_epoch.mean().item()  
            last_acc = accuracy_epoch[-1].item()
            if log_interval and (batch_idx % log_interval == 0):
                logger(
                    f"Batch {batch_idx + 1}/{n_batches}\n"
                    f"\tLoss: {loss.item():.4f}\n"
                    f"\tErr loss: {global_err_loss.item():.4f}\n"
                    f"\tState change: {global_state_change.item():.4f}\n"
                    f"\tmean accuracy: {overall_acc:.4f}\n"
                    f"\tlast accuracy: {last_acc:.4f}"
                )

                wandb_logger.log({
                    "loss": loss.item(),
                    "err_loss": global_err_loss.item(),
                    "state_change": global_state_change.item(),
                    "mean_accuracy": overall_acc,
                    "last_accuracy": last_acc
                })
            
            err_loss_epoch /= n_batches
            state_change_epoch /= n_batches
            accuracy_epoch = n_correct_epoch / n_samples_epoch


            if history:
                history.state_change_loss.append(state_change_epoch.cpu().numpy())
                history.loss['train'].append(err_loss_epoch.cpu().numpy())
                history.accuracy['train'].append(accuracy_epoch)
            
            if last_epoch:
                return self.text(train_loader, criterion, history=None)

    @torch.no_grad()
    def test_mmhs(
        self,
        data_loader: DataLoader,
        criterion: nn.Module,
        history: Optional[MultiModNHistory] = None,
        log_interval: Optional[int] = None,
        logger: Optional[Callable] = print,
        tag: str = "test",
        wandb_logger=None,
    ):
        """
        A GPU‐accelerated evaluation loop mirroring train_epoch_mmhs.
        """
        self.eval()
        n_stages  = len(self.encoders) + 1
        n_batches = len(data_loader)

        # on‐device accumulators
        n_samples_epoch    = torch.ones((n_stages,1), device=self.device)
        err_loss_epoch     = torch.zeros((n_stages,1), device=self.device)
        n_correct_epoch    = torch.zeros((n_stages,1), device=self.device)
        state_change_epoch = torch.zeros((n_stages-1), device=self.device)
        # will hold all the batch‐wise final predictions
        all_preds = []

        for batch_idx, batch in tqdm(enumerate(data_loader)):
            # unpack and send to GPU
            (data, target, encoder_seq) = (list(batch) + [None])[:3]
            data   = [d.to(self.device, non_blocking=True) for d in data]
            target = target[:,0].to(self.device, non_blocking=True)  # single‐decoder
            bsz    = target.size(0)

            # per‐batch accumulators
            err_loss     = torch.zeros((n_stages,1), device=self.device)
            state_change = torch.zeros((n_stages-1), device=self.device)

            # count samples for stage 0
            n_samples_epoch[0] += bsz

            # 1) initial state & decoder
            state = self.init_state(bsz)
            out   = self.decoders[0](state)
            _, pred = out.max(dim=1)
            err_loss[0][0] = criterion(out, target)
            n_correct_epoch[0][0] += (pred == target).sum().float()

            # 2) subsequent encoders + decoder
            for enc_idx, data_idx in self.get_encoder_iterable(
                    encoder_seq, shuffle_mode=self.shuffle_mode, train=False
                ):
                # encode, measure state change
                new_state = self.encoders[enc_idx](state, data[data_idx])
                state_change[enc_idx] = (new_state - state).pow(2).mean()
                state = new_state

                # decode
                out2 = self.decoders[0](state)
                _, pred2 = out2.max(dim=1)
                err_loss[enc_idx+1][0] = criterion(out2, target)
                n_correct_epoch[enc_idx+1][0] += (pred2 == target).sum().float()
                n_samples_epoch[enc_idx+1] += bsz
            all_preds.append(pred2.cpu().numpy())

            # accumulate
            err_loss_epoch     += err_loss
            state_change_epoch += state_change

            if log_interval and batch_idx % log_interval == 0:
                mean_err = (err_loss / bsz).mean().item()
                mean_acc = (n_correct_epoch / n_samples_epoch).mean().item()
                logger(
                    f"{tag.capitalize()} Batch {batch_idx+1}/{n_batches} "
                    f"ErrLoss={mean_err:.4f} Acc={mean_acc:.4f}"
                )
                if wandb_logger:
                    wandb_logger.log({
                        f"{tag}/batch_err_loss": mean_err,
                        f"{tag}/batch_accuracy": mean_acc,
                    })

        # finalize averages
        err_loss_epoch     /= n_batches
        state_change_epoch /= n_batches
        accuracy_epoch      = n_correct_epoch / n_samples_epoch

        # record in history
        if history is not None:
            history.loss.setdefault(tag, []).append(err_loss_epoch.cpu().numpy())
            history.accuracy.setdefault(tag, []).append(accuracy_epoch.cpu().numpy())
            history.state_change_loss.append(state_change_epoch.cpu().numpy())
        
        preds_array = np.concatenate(all_preds, axis=0)

        # return the summary metrics if you want
        return {
            "val_err_loss":     err_loss_epoch.mean().item(),
            "val_accuracy":     accuracy_epoch.mean().item(),
            "val_state_change": state_change_epoch.mean().item(),
        }, preds_array

    def test(
            self,
            test_loader: DataLoader,
            criterion: Union[nn.Module, Callable],
            history: Optional[MultiModNHistory] = None,
            tag: str = 'test',
            log_results: bool = False,
            logger: Optional[Callable] = None,
    ):
        # If log interval is given and logger is not, use print as default logger
        if log_results and not logger:
            logger = print
        self.eval()

        n_batches = len(test_loader)
        n_samples_prediction = np.ones((len(self.encoders) + 1, 1))

        err_loss_prediction = np.zeros((len(self.encoders) + 1, len(self.decoders)))
        n_correct_prediction = np.zeros((len(self.encoders) + 1, len(self.decoders)))

        output_decoder_epoch = [[]] * len(self.decoders)

        # tp_prediction = torch.zeros((len(self.encoders) + 1, len(self.decoders))).to(self.device)
        # tn_prediction = torch.zeros((len(self.encoders) + 1, len(self.decoders))).to(self.device)
        # fp_prediction = torch.zeros((len(self.encoders) + 1, len(self.decoders))).to(self.device)
        # fn_prediction = torch.zeros((len(self.encoders) + 1, len(self.decoders))).to(self.device)

        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(test_loader)):
                data, target, encoder_sequence = (list(batch) + [None])[:3]

                batch_size = target.shape[0]
                n_samples_prediction[0] += batch_size

                err_loss = torch.zeros((len(self.encoders) + 1, len(self.decoders)))
                # Matrices for each batch
                # tp = torch.zeros((len(self.encoders) + 1, len(self.decoders))).to(self.device)
                # tn = torch.zeros((len(self.encoders) + 1, len(self.decoders))).to(self.device)
                # fp = torch.zeros((len(self.encoders) + 1, len(self.decoders))).to(self.device)
                # fn = torch.zeros((len(self.encoders) + 1, len(self.decoders))).to(self.device)

                data_encoders = [data_encoder.to(self.device) for data_encoder in data]
                
                target = target.type(torch.LongTensor)
                target = target.to(self.device)

                state: Tensor = self.init_state(batch_size)

                if batch_idx == 0:  
                    target_decoder_epoch = target.cpu().detach()
                else:
                    target_decoder_epoch = torch.cat((target_decoder_epoch, target.cpu().detach()), dim = 0)   

                for dec_idx, decoder in enumerate(self.decoders):
                    target_decoder = target[:, dec_idx]
                    output_decoder = decoder(state)
                    _, prediction = torch.max(output_decoder, dim=1)
                        
                    err_loss[0][dec_idx] = criterion(output_decoder, target_decoder)
                    n_correct_prediction[0][dec_idx] += sum(
                        prediction == target_decoder).float()

                    # cm = None
                    # if decoder.n_classes == 2:
                    #     confmat = ConfusionMatrix(task="binary", num_classes=2).to(self.device)
                    #     cm = confmat(prediction, target_decoder)

                    # compute_metrics(tp, tn, fp, fn, cm, 0, dec_idx)

                for data_idx, enc_idx in self.get_encoder_iterable(encoder_sequence,
                                                                   shuffle_mode=self.shuffle_mode,
                                                                   train=False):
                    encoder = self.encoders[enc_idx]
                    data_encoder = data_encoders[data_idx]

                    # skip encoder if data contains nan value
                    if any(data_encoder.isnan().flatten()):
                        continue

                    n_samples_prediction[enc_idx + 1] += batch_size

                    state = encoder(state, data_encoder)

                    for dec_idx, decoder in enumerate(self.decoders):
                        target_decoder = target[:, dec_idx]
                        output_decoder = decoder(state)
                        _, prediction = torch.max(output_decoder, dim=1)
                        err_loss[enc_idx + 1][dec_idx] = criterion(output_decoder,
                                                                   target_decoder)
                        n_correct_prediction[enc_idx + 1][dec_idx] += sum(
                            prediction == target_decoder)

                        # cm = confmat(prediction, target_decoder)
                        # tp[enc_idx + 1][dec_idx] += cm[1][1]
                        # fp[enc_idx + 1][dec_idx] += cm[0][1]
                        # fn[enc_idx + 1][dec_idx] += cm[1][0]
                        # tn[enc_idx + 1][dec_idx] += cm[0][0]
                        
                        # To calculate the performance metrics for the decoder taking as input the state after the last encoder
                        if enc_idx == len(self.encoders)-1 and batch_idx== 0:
                            output_decoder_epoch[dec_idx] = output_decoder.cpu().detach()
                        elif enc_idx == len(self.encoders)-1:
                            output_decoder_epoch[dec_idx] = torch.cat((output_decoder_epoch[dec_idx], output_decoder.cpu().detach()), dim = 0)            


                err_loss_prediction += err_loss.detach().numpy()

                # tp_prediction += tp
                # fp_prediction += fp
                # fn_prediction += fn
                # tn_prediction += tn

        err_loss_prediction /= n_batches
        accuracy_prediction = n_correct_prediction / n_samples_prediction

        # sensitivity_denominator = tp_prediction + fn_prediction
        # sensitivity_prediction = torch.where(sensitivity_denominator == 0, 0,
        #                                   tp_prediction / sensitivity_denominator).detach().cpu().numpy()

        # specificity_denominator = tn_prediction + fp_prediction
        # specificity_prediction = torch.where(specificity_denominator == 0, 0,
        #                                   tn_prediction / specificity_denominator).detach().cpu().numpy()

        # balanced_accuracy_prediction = (sensitivity_prediction + specificity_prediction) / 2

        if log_results:
            logger(
                f"{tag.capitalize()} results\n"
                f"\tAverage loss: {np.mean(err_loss_prediction):.4f}\n"
                f"\tAccuracy: {np.mean(accuracy_prediction):.4f}\n"
                # f"\tSensitivity: {sensitivity_prediction:.4f}\n"
                # f"\tSpecificity: {specificity_prediction:.4f}\n"
                # f"\tBalanced accuracy: {balanced_accuracy_prediction:.4f}"
            )

        if history is not None:
            if tag not in history.loss:
                history.loss[tag] = []
            history.loss[tag].append(err_loss_prediction)

            if tag not in history.accuracy:
                history.accuracy[tag] = []
            history.accuracy[tag].append(accuracy_prediction)

            # if tag not in history.sensitivity:
            #     history.sensitivity[tag] = []
            # history.sensitivity[tag].append(sensitivity_prediction)

            # if tag not in history.specificity:
            #     history.specificity[tag] = []
            # history.specificity[tag].append(specificity_prediction)

            # if tag not in history.balanced_accuracy:
            #     history.balanced_accuracy[tag] = []
            # history.balanced_accuracy[tag].append(balanced_accuracy_prediction)
        # Output the results for each decoder with the state vector after the last encoder as input   
        results = [[]] * len(self.decoders)          
        for dec_idx in range(len(output_decoder_epoch)):
            output_decoder_epoch_dec_idx = output_decoder_epoch[dec_idx] 
            # Normalize so that the class probabilities sum to 1
            output_decoder_epoch_dec_idx = torch.div(output_decoder_epoch_dec_idx, torch.sum(output_decoder_epoch_dec_idx, dim =1).reshape(-1,1))
            _ , prediction_epoch_dec_idx = torch.max(output_decoder_epoch_dec_idx, dim=1)       
            target_decoder_epoch_dec_idx = target_decoder_epoch[:, dec_idx]     
            # results[dec_idx] = get_performance_metrics(target_decoder_epoch_dec_idx, prediction_epoch_dec_idx, output_decoder_epoch_dec_i dx[:,1])
            results[dec_idx] = get_performance_metrics(target_decoder_epoch_dec_idx, prediction_epoch_dec_idx)
        # also return the raw predictions from the last encoder & last decoder
        # shape: [n_samples]
        final_probs = output_decoder_epoch[-1]            # Tensor [n_samples, n_classes]
        _, final_preds = torch.max(final_probs, dim=1)   # Tensor [n_samples]
        return results, final_preds.cpu().numpy()   


    def predict(
            self,
            x: List[Tensor],
            encoder_sequence: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        self.eval()
        n_samples = x[0].shape[0]
        full_predictions = np.zeros(
            (len(self.encoders) + 1, len(self.decoders), n_samples))

        with torch.no_grad():
            x_encoders = [x_encoder.to(self.device) for x_encoder in x]
            state: Tensor = self.init_state(n_samples)

            for dec_idx, decoder in enumerate(self.decoders):
                output_decoder = decoder(state)
                _, prediction = torch.max(output_decoder, dim=1)

                full_predictions[0][dec_idx] = prediction.detach().numpy()

                # To predict probabilities instead of final class
                #full_predictions[0][dec_idx] = output_decoder[..., -1].item()

            for data_idx, enc_idx in self.get_encoder_iterable(encoder_sequence,
                                                               shuffle_mode=self.shuffle_mode,
                                                               train=False):
                encoder = self.encoders[enc_idx]
                state = encoder(state, x_encoders[data_idx])

                for dec_idx, decoder in enumerate(self.decoders):
                    output_decoder = decoder(state)
                    _, prediction = torch.max(output_decoder, dim=1)

                    full_predictions[enc_idx + 1][dec_idx] = prediction.detach().numpy()
                    # full_predictions[enc_idx + 1][dec_idx] = output_decoder[..., -1].item()

        return full_predictions

    def get_states(
            self,
            data_loader: DataLoader,
    ) -> List[Tensor]:
        self.eval()

        batch_states = []

        with torch.no_grad():
            for batch in data_loader:
                data, _, encoder_sequence = (list(batch) + [None])[:3]

                batch_size = data[0].shape[0]

                data_encoders = [data_encoder.to(self.device) for data_encoder in data]

                state: Tensor = self.init_state(batch_size)

                for data_idx, enc_idx in self.get_encoder_iterable(encoder_sequence,
                                                                   shuffle_mode=self.shuffle_mode,
                                                                   train=False):
                    encoder = self.encoders[enc_idx]
                    data_encoder = data_encoders[data_idx]

                    # skip encoder if data contains nan value
                    if any(data_encoder.isnan().flatten()):
                        continue

                    state = encoder(state, data_encoder)

                batch_states.append(state)

        return list(torch.cat(batch_states, dim=0))

    def display_arch(self, input: np.ndarray):
        for i, enc in enumerate(self.encoders):
            print('Encoder {}:'.format(i))
            state_shape = torch.Size([self.init_state.state_size])

            summary(enc, [state_shape, input[i].shape])
            print()

        for i, dec in enumerate(self.decoders):
            print('Decoder {}:'.format(i))
            state_shape = torch.Size([self.init_state.state_size])

            summary(dec, state_shape)
            print()

    def get_encoder_iterable(
            self,
            encoder_sequence: List[int],
            shuffle_mode: bool,
            train: bool,
    ) -> Iterable[Tuple[int, int]]:
        if encoder_sequence is None:
            encoder_iterable = enumerate(range(len(self.encoders)))
        else:
            encoder_iterable_batch = encoder_sequence.numpy().copy()
            encoder_iterable = encoder_iterable_batch[0]
            if not (encoder_iterable_batch == encoder_iterable).all():
                raise ValueError(
                    "Encoder sequence has different values across the batch. Hint: set batch size to 1 to avoid this error."
                )

            encoder_iterable = enumerate(encoder_iterable)

        if shuffle_mode and train:
            encoder_iterable = list(encoder_iterable)
            random.shuffle(encoder_iterable)

        return encoder_iterable
