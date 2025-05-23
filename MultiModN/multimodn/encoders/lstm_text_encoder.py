# multimodn/encoders/lstm_text_encoder.py
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Tuple, Optional
from transformers import AutoTokenizer
from multimodn.encoders import MultiModEncoder


class LSTMTextEncoder(MultiModEncoder):
    """
    LSTM encoder for token sequences.
      • input_ids shape : [batch, seq_len]   (LongTensor)
      • internal flow   : input_ids → Embedding → LSTM(s) → concat with state
    """

    def __init__(
        self,
        state_size: int,
        embed_dim: int = 300,
        hidden_layers: Tuple[int, ...] = (256,),
        padding_idx: Optional[int] = None,
        dropout: float = 0.1,
        activation: Callable = F.relu,
        freeze_embed: bool = False,
    ):
        """
        Args
        ----
        state_size   : size of the running state vector (comes from MultiModN)
        embed_dim    : dimension of each token embedding
        hidden_layers: one or more hidden sizes for stacked LSTMs
        padding_idx  : index that should be treated as <PAD>
        dropout      : dropout applied *after* embedding
        activation   : non-linearity between stacked LSTMs (default ReLU)
        freeze_embed : if True, embeddings are kept fixed
        """
        super().__init__(state_size)

        # ---- vocabulary from BERTweet tokenizer ----
        tok = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)
        vocab_size = tok.vocab_size
        padding_idx = padding_idx if padding_idx is not None else tok.pad_token_id

        # ---- layers ----
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        if freeze_embed:
            self.embed.weight.requires_grad_(False)

        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        # first LSTM layer takes embed_dim
        dim_layers = [embed_dim] + list(hidden_layers)
        self.lstm_layers = nn.ModuleList(
            nn.LSTM(in_dim, out_dim, batch_first=True)
            for in_dim, out_dim in zip(dim_layers, dim_layers[1:])
        )

        # final projection back to state_size
        # last_hidden + current_state  →  new_state
        final_hidden = hidden_layers[-1]
        self.fc = nn.Linear(final_hidden + state_size, state_size)

    # ---------------------------------------------------------------------- #
    def forward(self, state: Tensor, input_ids: Tensor) -> Tensor:
        """
        input_ids : [B, T]  (LongTensor)
        state     : [B, state_size]
        returns   : [B, state_size]
        """
        x = self.embed(input_ids)                  # [B, T, E]
        x = self.dropout(x)

        # stacked LSTM(s)
        for lstm in self.lstm_layers[:-1]:
            x, _ = lstm(x)
            x = self.activation(x)

        # last LSTM without activation so gradients propagate cleanly
        x, (h_n, _) = self.lstm_layers[-1](x)      # h_n shape [1, B, H]
        last_hidden = h_n.squeeze(0)               # [B, H]

        # concat with incoming state and project
        out = self.fc(torch.cat([last_hidden, state], dim=1))
        return out
