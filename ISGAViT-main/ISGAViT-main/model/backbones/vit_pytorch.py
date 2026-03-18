import math
from functools import partial
from itertools import repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import abc


# --------------------------------------------------------
# Utility Functions
# --------------------------------------------------------
def _ntuple(n):
    def parse(x):
        if isinstance(x, abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# --------------------------------------------------------
# Standard ViT Components (Mlp, Attention, Block, PatchEmbed)
# --------------------------------------------------------
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size_tuple = to_2tuple(stride_size)
        self.num_x = (img_size[1] - patch_size[1]) // stride_size_tuple[1] + 1
        self.num_y = (img_size[0] - patch_size[0]) // stride_size_tuple[0] + 1
        print('Using stride: {}, and patch number is num_y {} * num_x {}'.format(stride_size, self.num_y, self.num_x))
        self.num_patches = self.num_x * self.num_y
        self.img_size = img_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        return x


# --------------------------------------------------------
# 1. SFC: Semantic Feature Calibration Module
# --------------------------------------------------------
class SFC_Module(nn.Module):
    """
    Semantic Feature Calibration (SFC) Module as described in Sec 3.2.
    It acts as a soft gate to suppress identical uniforms and background noise.
    """

    def __init__(self, channel, b=1, gamma=2):
        super(SFC_Module, self).__init__()
        # Adaptively compute 1D Conv kernel size k based on channel dimension C
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input x shape: [B, N, C] (Patch Embeddings)
        y = x.transpose(1, 2)  # [B, C, N]
        y = self.avg_pool(y)  # Global Average Pooling -> [B, C, 1]

        y = y.transpose(1, 2)  # [B, 1, C]
        y = self.conv(y)  # 1D Conv -> [B, 1, C]
        y = self.sigmoid(y)  # Sigmoid weight recalibration -> [B, 1, C]

        # Element-wise calibration
        return x * y  # [B, N, C] *[B, 1, C] -> [B, N, C]


# --------------------------------------------------------
# 2. LCP: Latent Context Prompting Module
# --------------------------------------------------------
class LCP_Module(nn.Module):
    """
    Latent Context Prompting (LCP) Module as described in Sec 3.3.
    Implicitly generates instance-specific prompts from features without manual labels.
    """

    def __init__(self, embed_dim, num_prompts=5, reduction_ratio=4):
        super(LCP_Module, self).__init__()
        self.num_prompts = num_prompts
        self.embed_dim = embed_dim
        hidden_dim = embed_dim // reduction_ratio

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_prompts * embed_dim)
        )

    def forward(self, x):
        # Input x shape: [B, N, C] (Calibrated Patch Embeddings)
        v_ctx = x.mean(dim=1)  # Extract global context v_ctx -> [B, C]
        P_latent = self.mlp(v_ctx)  # Generate dynamic prompts -> [B, K * C]
        P_latent = P_latent.view(-1, self.num_prompts, self.embed_dim)  # [B, K, C]
        return P_latent


# --------------------------------------------------------
# Main Model: ISGA-ViT Framework
# --------------------------------------------------------
class ISGA_ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, num_classes=1000, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 camera=0, view=0, drop_path_rate=0., norm_layer=nn.LayerNorm, num_prompts=5):
        super().__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.cam_num = camera
        self.view_num = view
        self.num_prompts = num_prompts

        # Image Patch Embedding
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, stride_size=stride_size,
                                      in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # Class Token and Position Embeddings (1 cls + K prompts + N patches)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_prompts + num_patches, embed_dim))

        # -----------------------------------------------------
        # Our Core Modules
        # -----------------------------------------------------
        # 1. SFC: Semantic Feature Calibration
        self.sfc = SFC_Module(channel=embed_dim)

        # 2. LCP: Latent Context Prompting
        self.lcp = LCP_Module(embed_dim=embed_dim, num_prompts=num_prompts)

        # 3. GMA: Geometric Manifold Alignment (Replacing SIE)
        if camera > 0 and view > 0:
            self.gma_matrix = nn.Parameter(torch.zeros(camera * view, 1, embed_dim))
        elif camera > 0:
            self.gma_matrix = nn.Parameter(torch.zeros(camera, 1, embed_dim))
        elif view > 0:
            self.gma_matrix = nn.Parameter(torch.zeros(view, 1, embed_dim))

        # Lambda scaling factor for GMA, initialized to 0.01 as per Sec 4.2.2
        self.gma_scale = nn.Parameter(torch.ones(1, 1, embed_dim) * 0.01)

        # Transformer Blocks
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.fc = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Initializations
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        if hasattr(self, 'gma_matrix'):
            trunc_normal_(self.gma_matrix, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, camera_id=None, view_id=None):
        B = x.shape[0]

        # 1. Patch Embedding -> [B, N, C]
        E_patch = self.patch_embed(x)

        # 2. SFC: Feature Denoiser ->[B, N, C]
        E_patch_calibrated = self.sfc(E_patch)

        # 3. LCP: Generate Implicit Prompts -> [B, K, C]
        P_latent = self.lcp(E_patch_calibrated)

        # 4. Class Token -> [B, 1, C]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        # Sequence Assembly ->[B, 1 + K + N, C]
        Z_input = torch.cat((cls_tokens, P_latent, E_patch_calibrated), dim=1)

        # 5. GMA: Inject Geometric Bias
        b_geo = 0
        if self.cam_num > 0 and self.view_num > 0 and camera_id is not None and view_id is not None:
            b_geo = self.gma_matrix[camera_id * self.view_num + view_id]
        elif self.cam_num > 0 and camera_id is not None:
            b_geo = self.gma_matrix[camera_id]
        elif self.view_num > 0 and view_id is not None:
            b_geo = self.gma_matrix[view_id]

        E_pos = self.pos_embed
        if isinstance(b_geo, torch.Tensor):
            # E'_pos = E_pos + lambda * b_geo
            E_pos = E_pos + (self.gma_scale * b_geo).squeeze(1)

        Z_input = Z_input + E_pos
        Z_input = self.pos_drop(Z_input)

        # 6. Transformer Forward Pass
        for blk in self.blocks:
            Z_input = blk(Z_input)
        Z_input = self.norm(Z_input)

        return Z_input

    def forward(self, x, camera_id=None, view_id=None):
        x = self.forward_features(x, camera_id, view_id)
        # Final identity descriptor is the class token (index 0)
        cls_feat = x[:, 0]
        return self.fc(cls_feat)

    def load_param(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')
        if 'model' in param_dict: param_dict = param_dict['model']
        if 'state_dict' in param_dict: param_dict = param_dict['state_dict']

        for k, v in param_dict.items():
            if 'head' in k or 'dist' in k or 'fc' in k:
                continue
            if k == 'pos_embed':
                # Handle dimension mismatch when loading pretrained ImageNet weights
                if v.shape[1] != self.pos_embed.shape[1]:
                    # Standard ViT pos_embed is [1, 1+N_old, D]. We need to inject zero prompts in the middle.
                    if v.shape[1] < self.pos_embed.shape[1]:
                        prompt_pos = torch.zeros(1, self.num_prompts, self.embed_dim)
                        v = torch.cat([v[:, :1], prompt_pos, v[:, 1:]], dim=1)
                    v = resize_pos_embed(v, self.pos_embed, self.patch_embed.num_y, self.patch_embed.num_x,
                                         self.num_prompts)
            try:
                self.state_dict()[k].copy_(v)
            except Exception as e:
                print(f'Shape mismatch ignoring {k}: {v.shape} vs {self.state_dict()[k].shape}')


def resize_pos_embed(posemb, posemb_new, hight, width, num_prompts):
    # Split cls, prompts, and grid
    posemb_token = posemb[:, :1]
    posemb_prompts = posemb[:, 1:1 + num_prompts]
    posemb_grid = posemb[0, 1 + num_prompts:]

    ntok_new = posemb_new.shape[1] - 1 - num_prompts

    if posemb_grid.size()[0] == 196:
        posemb_grid = posemb_grid.reshape(1, 14, 14, -1).permute(0, 3, 1, 2)
    elif posemb_grid.size()[0] == 128:
        posemb_grid = posemb_grid.reshape(1, 16, 8, -1).permute(0, 3, 1, 2)
    elif posemb_grid.size()[0] == 192:
        posemb_grid = posemb_grid.reshape(1, 24, 8, -1).permute(0, 3, 1, 2)

    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)

    posemb = torch.cat([posemb_token, posemb_prompts, posemb_grid], dim=1)
    return posemb


# --------------------------------------------------------
# Model Builders
# --------------------------------------------------------
def vit_base_patch16_224_ISGA(img_size=(256, 128), stride_size=16, drop_rate=0.0, attn_drop_rate=0.0,
                              drop_path_rate=0.1, camera=0, view=0, **kwargs):
    model = ISGA_ViT(img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=768, depth=12, num_heads=12,
                     mlp_ratio=4, qkv_bias=True, camera=camera, view=view, drop_path_rate=drop_path_rate,
                     drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                     **kwargs)
    return model


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor