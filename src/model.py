"""
音乐流派分类模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

class GenreClassifier(nn.Module):

    def __init__(self, config: DictConfig):
        super().__init__()

        self.config = config
        self.num_layers = len(config.model.selected_layers)
        self.layer_dim = config.model.layer_dim
        self.num_classes = config.model.num_classes

        # 可学习的层权重（如果启用）
        if config.model.classifier.use_layer_weighting:
            self.layer_weights = nn.Parameter(torch.ones(self.num_layers))
        else:
            self.register_buffer('layer_weights', torch.ones(self.num_layers) / self.num_layers)

        # 构建编码器
        hidden_dims = config.model.classifier.hidden_dims
        dropout = config.model.classifier.dropout

        layers = []
        input_dim = self.layer_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)
        # 分类头
        self.classifier_head = nn.Linear(hidden_dims[-1], self.num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        # 重塑为 [batch, num_layers, layer_dim]
        x = x.view(batch_size, self.num_layers, self.layer_dim)
        # 应用层权重
        weights = F.softmax(self.layer_weights, dim=0)
        x = (x * weights.view(1, self.num_layers, 1)).sum(dim=1)  # [batch, layer_dim]
        # 通过编码器
        features = self.encoder(x)  # [batch, hidden_dims[-1]]
        logits = self.classifier_head(features)  # [batch, num_classes]
        return logits

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = x.view(batch_size, self.num_layers, self.layer_dim)

        weights = F.softmax(self.layer_weights, dim=0)
        x = (x * weights.view(1, self.num_layers, 1)).sum(dim=1)

        embedding = self.encoder(x)
        return embedding