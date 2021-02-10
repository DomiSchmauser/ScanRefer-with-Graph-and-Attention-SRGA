import os
import sys
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models.self_attention import Attention

class LangModule(nn.Module):
    def __init__(self, num_text_classes, use_lang_classifier=True, use_bidir=False,
        emb_size=300, hidden_size=256, num_hops=30, num_attention_hidden=256):
        super().__init__() 

        self.num_text_classes = num_text_classes
        self.use_lang_classifier = use_lang_classifier
        self.use_bidir = use_bidir
        self.hidden_size = hidden_size
        self.num_hops = num_hops

        self.gru = nn.GRU(
            input_size=emb_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=self.use_bidir
        )
        lang_size = hidden_size * 2 if self.use_bidir else hidden_size

        self.attention = Attention(lang_size, num_attention_hidden, num_hops)

        self.lang_size = lang_size

        self.final_linear = nn.Linear(self.num_hops, 1)

        # language classifier
        if use_lang_classifier:
            self.lang_cls = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.lang_size, num_text_classes),
                nn.Dropout()
            )

    def forward(self, data_dict):
        """
        encode the input descriptions
        """
        
        word_embs = data_dict["lang_feat"] #batch * max len tokens * featdim

        # Encoder GRU
        lengths = data_dict["lang_len"]
        sorted_lengths, sorted_idx = torch.sort(lengths, descending=True) #longest first
        word_embs = word_embs[sorted_idx] #pack longest first
        lang_feat = pack_padded_sequence(word_embs, sorted_lengths, batch_first=True, enforce_sorted=False) #num_tokens * featdim + batchsize packed

        # Encode description
        output, _ = self.gru(lang_feat) #tensor containing output features h_t of the last layer of the GRU for each t # batch, seq, feature 1,B,feat
        unpacked_gru_out, _ = pad_packed_sequence(output, batch_first=True) #batchsize x num_hiddenstates x featuresize
        _, reversed_idx = torch.sort(sorted_idx)
        unpacked_gru_out = unpacked_gru_out[reversed_idx] # sort shortest first

        # Self Attention
        attention_weights = self.attention(unpacked_gru_out, lengths) #B x  num_hops x num_hiddenstates
        lang_last = torch.bmm(attention_weights, unpacked_gru_out) #B x num_hops x featuresize(lang_size)
        lang_last = lang_last.permute(0, 2, 1).contiguous()
        lang_last = self.final_linear(lang_last) #B x hiddensize x 1
        lang_last = torch.squeeze(lang_last)

        # Store the encoded language features, attention weights and num_hops
        data_dict["lang_emb"] = lang_last #B, hidden_size
        data_dict["attention_weights"] = attention_weights
        data_dict["num_hops"] =  self.num_hops
        
        # classify
        if self.use_lang_classifier:
            data_dict["lang_scores"] = self.lang_cls(data_dict["lang_emb"])

        return data_dict

