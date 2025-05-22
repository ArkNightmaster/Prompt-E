from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from functools import partial
import math
from backbone.vision_transformer_clip_prompt_l2p import resize_pos_embed_clip
from utils.toolkit import cosine_similarity
from backbone.prompt import Prompt

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x

class VisionTransformer_l2p(nn.Module):
    """
    VisionTransformer 类的增强版本，引入了提示池（Prompt Pool）。
    """

    def __init__(
            self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
            num_classes: int = None, prompt_length: int = 5, pool_size: int = 10, top_k: int = 5, prompt_init: str = 'uniform',
            prompt_pool: bool = True, embedding_key: str = 'cls', use_prompt_mask: bool = False,
            batchwise_prompt: bool = True, prompt_key: bool = True, prompt_key_init: str = 'uniform', head_type: str = 'token', mask_ratio: float = 0.75,
            use_lrg: bool = True, use_fc_norm: bool = True, **kwargs):
        super().__init__()
        norm_layer = LayerNorm if use_fc_norm else nn.Identity
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        num_patches = (input_resolution // patch_size) ** 2
        
        self.prompt_pool = prompt_pool
        if prompt_pool:
            total_prompt_length = prompt_length * top_k
        else:
            total_prompt_length = 0

        self.total_seq_length = num_patches + 1 + total_prompt_length

        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.total_seq_length, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        # 引入提示池
        if prompt_pool:
            self.prompt = Prompt(
                length=prompt_length,
                embed_dim=width,
                embedding_key=embedding_key,
                prompt_init=prompt_init,
                prompt_pool=prompt_pool,
                pool_size=pool_size,
                top_k=top_k,
                batchwise_prompt=batchwise_prompt,
                prompt_key_init=prompt_key_init,
                prompt_key=prompt_key
            )
            self.use_prompt_mask = use_prompt_mask
        else:
            self.prompt = None

        self.head_type = head_type
        self.use_lrg = use_lrg

        if self.use_lrg:
            self.downstream_layers = nn.Sequential(OrderedDict([
                ('align_layer', self.align_layers(mask_ratio, output_dim, width)),
                ('generative_linear', nn.Sequential(
                    nn.Conv1d(in_channels=width, out_channels=width, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Conv1d(in_channels=width, out_channels=width, kernel_size=1, stride=1, padding=0)))
            ]))
        
        # Classifier Head
        self.fc_norm = norm_layer(output_dim) if use_fc_norm else nn.Identity()
        self.head = nn.Linear(output_dim, num_classes) if num_classes > 0 else nn.Identity()

    class align_layers(nn.Module):
        """
        align feature and do mask ratio
        """
        def __init__(self, mask_ratio, in_channel, out_channel):
            super().__init__()
            self.lambda1 = mask_ratio
            self.align_layer = nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0)

        def forward(self, x):

            x = self.align_layer(x)
            mask_ratio = torch.rand(x.shape).to(x.device)
            mask_ratio = (mask_ratio >= self.lambda1).float()

            return x * mask_ratio

    def forward(self, x: torch.Tensor, task_id: int = -1, cls_features: torch.Tensor = None, train: bool = False):
        x = self.conv1(x)  # [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # [*, grid ** 2, width]

        if self.prompt is not None and self.prompt_pool:
            if self.use_prompt_mask:
                prompt_mask = None  
            else:
                prompt_mask = None
            res = self.prompt(x, prompt_mask=prompt_mask, cls_features=cls_features)
            self.total_prompt_len = res['total_prompt_len']
            x = res['prompted_embedding']
        else:
            res = {}
            x = x

        x = torch.cat([
            self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x
        ], dim=1)  # [*, grid ** 2 + 1 (+ prompt tokens), width]

        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)

        if self.proj is not None:
            x = x @ self.proj
        
        # x: Batch, cls_token + prompt_length, 512
        # cur_text: Batch, 512

        res['x'] = x

        if self.head_type == 'token':
            x = x[:, 0]
        elif self.head_type == 'gap':
            x = x.mean(dim=1)
        elif self.head_type == 'prompt':
            x = x[:, 1:(1 + self.total_prompt_len)] if hasattr(self, 'class_token') and self.class_token else x[:, 0:self.total_prompt_len]
            x = x.mean(dim=1)
        elif self.head_type == 'token+prompt':
            x = x[:, 0:self.total_prompt_len + 1]
            x = x.mean(dim=1)
        elif self.head_type == 'token_with_prompt':
            # Cosine Prompt Regularization (CPR)
            cls_tokens = x[:, 0]
            prompt_tokens = x[:, 1:self.total_prompt_len + 1]
            if self.use_lrg:
                x2 = cls_tokens # Batch, 512
                x2 = x2.unsqueeze(2)  # [batch_size, 512, 1]
                x2 = self.downstream_layers(x2)
                x2 = x2.squeeze(2)  # [batch_size, 768]
                res['reconstruct_pre_logits'] = x2  # [batch_size, 768]
            else:
                res['reconstruct_pre_logits'] = None

            anchor_feature = cls_tokens

            if hasattr(self, 'rt_weight') and self.rt_weight:
                x, prompt_weights = cosine_similarity(anchor_feature, prompt_tokens, self.rt_weight)
                res['prompt_weight'] = prompt_weights
            else:
                x = cosine_similarity(anchor_feature, prompt_tokens)
            x = x.mean(dim=1)
        else:
            raise ValueError(f'Invalid classifier={self.head_type}')

        res['pre_logits'] = x

        x = self.fc_norm(x)

        res['logits'] = self.head(x) # Batch, 512

        return res


        
class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 **kwargs
                 ):
        super().__init__()

        self.context_length = context_length

        self.num_classes = kwargs.get('num_classes', 10)
        self.prompt_length = kwargs.get('prompt_length', 5)
        self.pool_size = kwargs.get('pool_size', 10)
        self.top_k = kwargs.get('top_k', 5)
        self.prompt_init = kwargs.get('prompt_init', 'uniform')
        self.prompt_pool = kwargs.get('prompt_pool', True)
        self.embedding_key = kwargs.get('embedding_key', 'cls')
        self.use_prompt_mask = kwargs.get('use_prompt_mask', False)
        self.batchwise_prompt = kwargs.get('batchwise_prompt', True)
        self.prompt_key_init = kwargs.get('prompt_key_init', 'uniform')
        self.prompt_key = kwargs.get('prompt_key', True)
        self.head_type = kwargs.get('head_type', 'token_with_propmt')
        self.use_lrg = kwargs.get('use_lrg', True)
        self.mask_ratio = kwargs.get('mask_ratio', 0.75)

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            get_original_backbone = kwargs.get('original', False)
            if get_original_backbone:
                self.visual = VisionTransformer(
                    input_resolution=image_resolution,
                    patch_size=vision_patch_size,
                    width=vision_width,
                    layers=vision_layers,
                    heads=vision_heads,
                    output_dim=embed_dim
                )
                # cls_feature: Batch, 768
                if not kwargs.get('use_clip_proj', True):
                    setattr(self.visual, "proj", None)
            else:
                self.visual = VisionTransformer_l2p(
                    input_resolution=image_resolution,
                    patch_size=vision_patch_size,
                    width=vision_width,
                    layers=vision_layers,
                    heads=vision_heads,
                    output_dim=embed_dim,
                    num_classes=self.num_classes,
                    prompt_length=self.prompt_length,
                    pool_size=self.pool_size,
                    top_k=self.top_k,
                    prompt_init=self.prompt_init,
                    prompt_pool=self.prompt_pool,
                    embedding_key=self.embedding_key,
                    use_prompt_mask=self.use_prompt_mask,
                    batchwise_prompt=self.batchwise_prompt,
                    prompt_key=self.prompt_key,
                    prompt_key_init=self.prompt_key_init,
                    head_type=self.head_type,
                    use_lrg=self.use_lrg,
                    mask_ratio=self.mask_ratio
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        if isinstance(l, LayerNorm):
            if hasattr(l, 'weight'):
                l.weight.data = l.weight.data.half()
            if hasattr(l, 'bias'):
                l.bias.data = l.bias.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

        if hasattr(l, 'prompt'):
            if l.prompt is not None:
                if isinstance(l.prompt, Prompt):
                    l.prompt.prompt.data = l.prompt.prompt.data.half()
                    l.prompt.prompt_key.data = l.prompt.prompt_key.data.half()
                else:
                    l.prompt.data = l.prompt.data.half()
                    l.prompt_key.data = l.prompt_key.data.half()

        if hasattr(l, 'downstream_layers'):
            for name, module in l.downstream_layers.named_modules():
                if isinstance(module, (nn.Conv1d, nn.Linear)):
                    module.weight.data = module.weight.data.half()
                    if module.bias is not None:
                        module.bias.data = module.bias.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, **kwargs):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, 
        **kwargs
    )
    state_dict1 = {}
    state_dict2 = state_dict.copy()
    if 'positional_embedding' in state_dict:
        state_pos_embed = state_dict['visual.positional_embedding'].clone()
        model_pos_embed = model.visual.positional_embedding

        if state_pos_embed.shape != model_pos_embed.shape:
            state_pos_embed = resize_pos_embed_clip(
                posemb=state_pos_embed,
                posemb_new=model_pos_embed,
                num_prefix_tokens=1,  # cls token
                gs_new=(model.visual.input_resolution // vision_patch_size, model.visual.input_resolution // vision_patch_size)
            )
            state_dict1['visual.positional_embedding'] = state_pos_embed
            model.load_state_dict(state_dict1, strict=False)

    for key in ["input_resolution", "context_length", "vocab_size", "visual.positional_embedding"]:
        if key in state_dict2:
            del state_dict2[key]

    convert_weights(model)
    model.load_state_dict(state_dict2, strict=False)
    return model.eval()
