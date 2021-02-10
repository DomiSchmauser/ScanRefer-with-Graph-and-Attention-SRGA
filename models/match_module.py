import torch
import torch.nn as nn

from models.dgcnn import DGCNN

class MatchModule(nn.Module):
    def __init__(self, num_proposals=256, lang_size=256, hidden_size=128, fuse_twice=False, skip_connection=False):
        super().__init__()

        self.num_proposals = num_proposals
        self.lang_size = lang_size
        self.hidden_size = hidden_size
        self.fuse_twice = fuse_twice
        self.skip_connection = skip_connection

        self.fuse = nn.Sequential(
            nn.Conv1d(self.lang_size + 128, hidden_size, 1),
        )

        self.reduce = nn.Sequential(
            nn.Conv1d(self.lang_size + 128, hidden_size, 1),
        )

        self.match = nn.Sequential(
            nn.Conv1d( hidden_size, hidden_size, 1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, 1, 1)
        )

        self.graph = DGCNN(initial_dim=128 + self.lang_size,  # if fuse before
                           out_dim=128,
                           k_neighbors=7,
                           intermediate_feat_dim=[64, 64, 128],
                           subtract_from_self=True)

    def forward(self, data_dict):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4)
        """

        # unpack outputs from detection branch
        features = data_dict['aggregated_vote_features']  # batch_size, num_proposal, 128
        objectness_masks = data_dict['objectness_scores'].max(2)[1].float().unsqueeze(2)  # batch_size, num_proposals, 1 c

        # unpack outputs from language branch
        lang_feat = data_dict["lang_emb"]  # batch_size, lang_size
        lang_feat = lang_feat.unsqueeze(1).repeat(1, self.num_proposals, 1)  # batch_size, num_proposals, lang_size

        # unpack bbox coordinates
        center = data_dict['center']  # (batch_size, num_proposal, 3)
        center = center.permute(0, 2, 1).contiguous()  # (batch_size, 3, num_proposal)

        # concat lang and object features
        features = torch.cat([features, lang_feat], dim=-1)
        features = features.permute(0, 2, 1).contiguous()  # batch_size, 128+lang_size, num_proposals

        # mask out invalid proposals
        objectness_masks = objectness_masks.permute(0, 2, 1).contiguous()  # batch_size, 1, num_proposals
        features = features * objectness_masks  # batch_size, hidden_size, num_proposals

        # Graph module with skip connection
        if self.skip_connection:

            skip_features = self.reduce(features)  # reduce dim from 384 to 128
            graph_out_features = self.graph(features, center) + skip_features  # skip connection  # batch_size, hidden_size, num_proposals

        else:
            graph_out_features = self.graph(features, center) # batch_size, hidden_size, num_proposals

        # fuse lang features twice
        if self.fuse_twice:

            print('fusing language features before and after graph...')
            graph_out_features = graph_out_features.permute(0, 2, 1).contiguous()
            features = torch.cat([graph_out_features, lang_feat], dim=-1)  # batch_size, num_proposals, 128 + lang_size
            features = features.permute(0, 2, 1).contiguous()  # batch_size, 128 + lang_size, num_proposals
            # fuse features
            features = self.fuse(features)  # batch_size, hidden_size, num_proposals
            confidences = self.match(features).squeeze(1)

        else:

            confidences = self.match(graph_out_features).squeeze(1)  # batch_size, num_proposals

        data_dict["cluster_ref"] = confidences

        return data_dict