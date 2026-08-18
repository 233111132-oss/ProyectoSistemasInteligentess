"""
Vision Transformer (ViT) para Clasificación de Enfermedades en Hojas Foliares.
Implementación de arquitectura Transformer aplicada a visión artificial:
- Patch Embedding lineal
- Class Token ([CLS]) aprendible
- Positional Embeddings 1D
- Bloques Encoder con Multi-Head Self-Attention (MHSA)
- Normalización por capas (LayerNorm) y Conexiones Residuales
- Feed-Forward MLP con activación GELU
- Cabezal de Clasificación (MLP Head) y Extracción de Mapas de Atención (Attention Rollout)
"""

import sys, os
site_pkg = os.path.abspath('.venv/Lib/site-packages')
if os.path.exists(site_pkg) and site_pkg not in sys.path:
    sys.path.insert(0, site_pkg)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    """
    Divide la imagen de entrada en parches (patches) y los proyecta a un vector de dimensión 'embed_dim'.
    Entrada: (B, C, H, W)
    Salida: (B, N, embed_dim) donde N = (H // patch_size) * (W // patch_size)
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Convolución 2D equivalente a una proyección lineal de parches no superpuestos
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W) -> proj: (B, embed_dim, H/P, W/P)
        x = self.proj(x)
        # Flatten espacial: (B, embed_dim, N) -> Transpose: (B, N, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Mecanismo de Autoatención Multi-Cabezal (Multi-Head Self-Attention).
    Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
    """
    def __init__(self, embed_dim=192, num_heads=6, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attn=False):
        B, N, C = x.shape
        # qkv: (B, N, 3 * C) -> (B, N, 3, num_heads, head_dim) -> permute: (3, B, num_heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Producto punto escalado: (B, num_heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn_weights = attn
        attn = self.attn_drop(attn)

        # Contexto ponderado: (B, num_heads, N, head_dim) -> (B, N, C)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        if return_attn:
            return out, attn_weights
        return out


class MLP(nn.Module):
    """
    Red Perceptrón Multicapa (Feed-Forward Network) con activación GELU.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """
    Bloque de Encoder Transformer con conexiones residuales y LayerNorm.
    x_1 = x + MHSA(LN(x))
    x_2 = x_1 + MLP(LN(x_1))
    """
    def __init__(self, embed_dim=192, num_heads=6, mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(in_features=embed_dim, hidden_features=hidden_dim, drop=drop)

    def forward(self, x, return_attn=False):
        if return_attn:
            attn_out, attn_weights = self.attn(self.norm1(x), return_attn=True)
            x = x + attn_out
            x = x + self.mlp(self.norm2(x))
            return x, attn_weights
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x


class VisionTransformer(nn.Module):
    """
    Arquitectura Vision Transformer (ViT) completa para Diagnóstico Fitosanitario.
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=4,
        embed_dim=192,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.1,
        attn_drop_rate=0.1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size

        # 1. Patch Embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.n_patches

        # 2. [CLS] Token aprendible y Positional Encodings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 3. Bloques Transformer Encoder
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate
            )
            for _ in range(depth)
        ])

        # 4. Capa de Normalización Final
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # 5. Cabezal de Clasificación
        self.head = nn.Linear(embed_dim, num_classes)

        # Inicialización de pesos
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_module_weights)

    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward_features(self, x, return_attn=False):
        B = x.shape[0]
        # Parches lineales: (B, N, embed_dim)
        x = self.patch_embed(x)

        # Expandir CLS token para todo el batch: (B, 1, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, N+1, embed_dim)

        # Sumar codificación posicional
        x = x + self.pos_embed
        x = self.pos_drop(x)

        attn_list = []
        for block in self.blocks:
            if return_attn:
                x, attn_w = block(x, return_attn=True)
                attn_list.append(attn_w)
            else:
                x = block(x)

        x = self.norm(x)
        if return_attn:
            return x, attn_list
        return x

    def forward(self, x):
        # Extraer representación del CLS token
        feat = self.forward_features(x)
        cls_feat = feat[:, 0]
        logits = self.head(cls_feat)
        return logits

    def get_attention_map(self, x):
        """
        Calcula el mapa de atención acumulado (Attention Rollout) para interpretabilidad.
        """
        self.eval()
        with torch.no_grad():
            _, attn_list = self.forward_features(x, return_attn=True)
            # Promediar cabezales en cada capa
            # attn_list: lista de (B, num_heads, N+1, N+1)
            rollout = torch.eye(attn_list[0].size(-1)).to(x.device).unsqueeze(0).repeat(x.size(0), 1, 1)
            for attn in attn_list:
                # Promedio sobre los cabezales
                attn_mean = attn.mean(dim=1)
                # Agregar identidad para flujo residual
                identity = torch.eye(attn_mean.size(-1)).to(x.device).unsqueeze(0)
                attn_mean = 0.5 * attn_mean + 0.5 * identity
                # Normalizar filas
                attn_mean = attn_mean / attn_mean.sum(dim=-1, keepdim=True)
                rollout = torch.bmm(attn_mean, rollout)

            # Extraer atención desde el CLS token a todos los parches espaciales (excluyendo el CLS en sí)
            cls_attn = rollout[:, 0, 1:]  # (B, N_patches)
            grid_size = int(math.sqrt(cls_attn.size(-1)))
            cls_attn = cls_attn.reshape(-1, 1, grid_size, grid_size)
            # Interpolar al tamaño original de la imagen
            cls_attn = F.interpolate(cls_attn, size=(self.img_size, self.img_size), mode='bicubic', align_corners=False)
            cls_attn = cls_attn.squeeze(1)
            return cls_attn
