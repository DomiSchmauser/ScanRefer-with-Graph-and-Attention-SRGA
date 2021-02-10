import torch
import torch.nn as nn

class Attention(nn.Module):
    ''' Computation of Attention weights'''

    def __init__(self, num_encoder_hidden, num_attention_hidden, num_hops):

        super().__init__()

        self.num_attention_hidden = num_attention_hidden
        self.num_hops = num_hops

        self.score = nn.Sequential(
            nn.Linear(num_encoder_hidden, num_attention_hidden, bias=False),
            nn.Tanh(),
            nn.Linear(num_attention_hidden, num_hops, bias=False)
        )

    def forward(self, encoder_hidden, lengths):

        scores = self.score(encoder_hidden) # B x num_hiddenstates x num_hops

        # mask attention scores
        B, S, _ = scores.size() # S = len longest hiddenstate
        idx = lengths.new_tensor(torch.arange(0, S).unsqueeze(0).repeat(B, 1)).long() #clone lengths tensor
        lengths = lengths.unsqueeze(1).repeat(1, S)

        mask = (idx >= lengths)
        mask = mask.unsqueeze(2).repeat(1, 1, self.num_hops)
        scores.masked_fill_(mask, float('-1e30')) #attn mask

        # softmax
        weights = nn.functional.softmax(scores, dim=1) # B x num_hiddenstates x  num_hops
        weights = weights.transpose(1, 2)  # B x  num_hops x num_hiddenstates

        return weights
