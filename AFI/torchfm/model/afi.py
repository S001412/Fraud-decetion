import torch
import torch.nn.functional as F

from torchfm.layer import FeaturesEmbedding, FeaturesLinear, MultiLayerPerceptron


class AutomaticFeatureInteractionModel(torch.nn.Module):
    """
    A pytorch implementation of AutoInt.

    Reference:
        W Song, et al. AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks, 2018.
    """

    def __init__(self, field_dims, embed_dim, num_heads, num_layers, mlp_dims, dropouts):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_fields = len(field_dims)
        self.linear = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        # self.res = torch.nn.Linear(self.embed_output_dim,self.embed_output_dim)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim+399, mlp_dims, dropouts[1])
        self.self_attns = torch.nn.ModuleList([
            torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=dropouts[0]) for _ in range(num_layers)
        ])
        self.attn_fc = torch.nn.Linear(self.embed_output_dim, 1)

    def forward(self, x,x2):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        x2 = torch.sigmoid(x2)
        cross_term = embed_x.transpose(0, 1)
        for self_attn in self.self_attns:
            cross_term, _ = self_attn(cross_term, cross_term, cross_term)
        cross_term = cross_term.transpose(0, 1)
        # Res = self.res(embed_x.view(-1,self.embed_output_dim)).view(-1,int(self.embed_output_dim/self.embed_dim),
        #                                                             self.embed_dim)
        # cross_term = cross_term + Res
        cross_term = F.relu(cross_term).contiguous().view(-1, self.embed_output_dim)
        mlp_x = torch.cat((embed_x.view(-1, self.embed_output_dim),x2),dim=1)
        x = self.linear(x) + self.attn_fc(cross_term) + self.mlp(mlp_x)

        return torch.sigmoid(x.squeeze(1))
