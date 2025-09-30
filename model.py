import torch
from torch import nn


class RNNModel(nn.Module):
    def __init__(
        self,
        cat_sizes: list[int],
        num_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_cats = len(cat_sizes)
        self.num_dimensions = 2 if bidirectional else 1

        embed_dims = [min(15, (size + 1) // 2) for size in cat_sizes]

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(size, embed_dim)
                for size, embed_dim in zip(cat_sizes, embed_dims)
            ]
        )
        input_size = sum(embed_dims) + num_size

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.linear = nn.Linear(self.hidden_size * self.num_dimensions, output_size)

    def forward(self, cat_x: torch.Tensor, num_x: torch.Tensor):
        embedded = [self.embeddings[i](cat_x[:, :, i]) for i in range(self.num_cats)]
        embedded_cat = torch.cat(embedded, dim=-1)
        x = torch.cat([embedded_cat, num_x], dim=-1)

        device = x.device
        h0 = torch.zeros(
            self.num_layers * self.num_dimensions,
            x.size(0),
            self.hidden_size,
            device=device,
        )
        c0 = torch.zeros(
            self.num_layers * self.num_dimensions,
            x.size(0),
            self.hidden_size,
            device=device,
        )

        out, _ = self.rnn(x, (h0, c0))
        out = self.linear(out[:, -1, :]).squeeze(1)

        return out
