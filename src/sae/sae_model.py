"""
Sparse Autoencoder (SAE) Model Definition

标准 SAE 架构，用于学习稀疏的可解释特征
"""

import torch
from typing import Dict, Tuple


class SparseAutoencoder(torch.nn.Module):
    """
    标准 SAE 架构：
    - Encoder: x -> ReLU(W_enc @ (x - b_dec) + b_enc)
    - Decoder: f -> W_dec @ f + b_dec
    """
    def __init__(
        self,
        d_in: int,
        n_features: int,
        l1_coefficient: float = 1e-3,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.d_in = d_in
        self.n_features = n_features
        self.l1_coefficient = l1_coefficient
        
        # 参数初始化
        self.W_enc = torch.nn.Parameter(torch.randn(n_features, d_in, dtype=dtype) * 0.01)
        self.W_dec = torch.nn.Parameter(torch.randn(d_in, n_features, dtype=dtype) * 0.01)
        self.b_enc = torch.nn.Parameter(torch.zeros(n_features, dtype=dtype))
        self.b_dec = torch.nn.Parameter(torch.zeros(d_in, dtype=dtype))
        
        # 归一化 decoder weights (单位向量)
        self.normalize_decoder()
    
    def normalize_decoder(self):
        """Normalize decoder weights to unit norm"""
        with torch.no_grad():
            self.W_dec.data = torch.nn.functional.normalize(self.W_dec.data, dim=0)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, d_in]
        return: [batch, n_features] (sparse activations)
        """
        x_centered = x - self.b_dec
        pre_relu = x_centered @ self.W_enc.T + self.b_enc
        return torch.nn.functional.relu(pre_relu)
    
    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """
        f: [batch, n_features]
        return: [batch, d_in] (reconstructed)
        """
        return f @ self.W_dec.T + self.b_dec
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        return: (reconstructed, features)
        """
        features = self.encode(x)
        reconstructed = self.decode(features)
        return reconstructed, features
    
    def loss(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算 SAE 损失：reconstruction + L1 sparsity
        """
        x_reconstructed, features = self.forward(x)
        
        # MSE reconstruction loss
        mse_loss = torch.nn.functional.mse_loss(x_reconstructed, x)
        
        # L1 sparsity loss
        l1_loss = self.l1_coefficient * features.abs().sum(dim=-1).mean()
        
        # Total loss
        total_loss = mse_loss + l1_loss
        
        # 统计信息
        with torch.no_grad():
            # L0: 平均每个样本激活了多少个 features
            l0 = (features > 0).float().sum(dim=-1).mean()
            # 重构相关系数
            cos_sim = torch.nn.functional.cosine_similarity(x, x_reconstructed, dim=-1).mean()
        
        return {
            'total': total_loss,
            'mse': mse_loss,
            'l1': l1_loss,
            'l0': l0,
            'cos_sim': cos_sim,
        }


