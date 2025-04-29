import torch
import torch.nn as nn
from models.mlp import MLP

class MapTransformer(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=8, num_encoder_layers=6, num_decoder_layers=6, num_queries=50,
                 num_classes=10, max_trips=10, max_lines=50, points_per_line=10):
        super(MapTransformer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.max_trips = max_trips
        self.max_lines = max_lines
        self.points_per_line = points_per_line

        self.input_proj = MLP(2 * points_per_line, hidden_dim, hidden_dim, 3)
        self.class_embed = nn.Embedding(num_classes, hidden_dim)
        self.conditional_layer = nn.Linear(hidden_dim, hidden_dim)

        self.transformer = nn.Transformer(
            d_model=hidden_dim, nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            batch_first=True
        )

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.class_head = nn.Linear(hidden_dim, num_classes + 1)
        self.polyline_head = MLP(hidden_dim, hidden_dim, points_per_line * 2, 3)

        self.trip_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.line_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, input_tensor, mask):
        B, M, L_max, N, _ = input_tensor.shape
        mask = mask.view(B, M * L_max)

        coord_embed = self.input_proj(input_tensor[..., :2].reshape(B, M * L_max, N * 2))
        class_embed = self.class_embed(input_tensor[..., 2].view(B, M * L_max, N)[..., 0].long())
        conditional_embed = self.conditional_layer(class_embed)
        combined_embed = coord_embed + conditional_embed

        pos = torch.cat([
            self.trip_embed[:M].unsqueeze(0).repeat(L_max, 1, 1),
            self.line_embed[:L_max].unsqueeze(1).repeat(1, M, 1)
        ], dim=-1).flatten(0, 1).unsqueeze(0).repeat(B, 1, 1)

        src = combined_embed
        tgt = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        hs = self.transformer(pos + src, tgt, src_key_padding_mask=mask)

        outputs_class = self.class_head(hs)
        outputs_polylines = self.polyline_head(hs).sigmoid()

        return outputs_class, outputs_polylines
