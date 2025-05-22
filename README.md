# Prompt-Enhanced: Leveraging Language Representation for Prompt Continual Learning

Official code repository for the paper 

"Prompt-Enhanced: Leveraging Language Representation for Prompt Continual Learning" 

This project includes the plug-and-play method Prompt-E implemented using PyTorch.

## Environment Requirements

- Python 3.9.18
- PyTorch 2.0.0
- torchvision 0.15.2
- timm 0.6.12
- For more dependencies, see `requirements.txt`

## Installation Guide

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/ArkNightmaster/Prompt-E.git
cd Prompt-E
pip install -r requirements.txt
```

## Key Implementations

Our method consists of two key components: Cosine Prompt Regularization (CPR) and Language Representation Guidance (LRG). Here are the key implementations that showcase our plug-and-play approach:

### 1. Cosine Prompt Regularization (CPR)

```python
# In vision_transformer_clip_prompt_l2p.py, forward_head method
def forward_head(self, res, pre_logits: bool = False):
    # ... existing code ...
    
    elif self.head_type == 'token_with_prompt' and self.prompt_pool and self.class_token:
        res['cls_token_logits'] = x[:,0]

        # Cosine Prompt Regularization (CPR)
        cls_tokens = x[:,0]
        prompt_tokens = x[:, 1:self.total_prompt_len + 1]
        if self.rt_weight:
            x, prompt_weights = cosine_similarity(cls_tokens, prompt_tokens, self.rt_weight)
            res['prompt_weight'] = prompt_weights
        else:
            x = cosine_similarity(cls_tokens, prompt_tokens)
        x = x.mean(dim=1)
    
    # ... existing code ...
```

### 2. Language Representation Guidance (LRG)

```python
# In models/clip_prompt_l2p.py
def _train(self, train_loader, val_loader, test_loader):
    # ... existing code ...
    
    for _, epoch in enumerate(prog_bar):
        # ... existing code ...
        
        for i, (_, inputs, targets, names) in enumerate(train_loader):
            inputs, targets = inputs.to(self._device), targets.long().to(self._device)
            text_feature = self.text_feature
            
            with torch.cuda.amp.autocast():
                output = self.network1(inputs, task_id=self._cur_task, train=True)
                logits = output["logits"][:, :self._total_classes]
                logits[:, :self._known_classes] = float('-inf')
                
                # Standard cross-entropy loss
                loss_origin = F.cross_entropy(logits, targets)
                
                # Language Representation Guidance loss
                loss_text_matching = info_nce_loss(output['reconstruct_pre_logits'], text_feature, targets, tau=self.tau)
                
                # Combined loss
                loss = alpha * loss_origin + beta * loss_text_matching
                
                # ... remaining code ...
```

The LRG module in the ViT backbone:

```python
# In vision_transformer_clip_prompt_l2p.py
# Generative Decoder for LRG
self.mask_ratio = mask_ratio
self.downstream_layers = nn.Sequential(OrderedDict([
    ('align_layer', self.align_layers(mask_ratio)),
    ('generative_linear', nn.Sequential(
        nn.Conv1d(in_channels=768, out_channels=768, kernel_size=1, stride=1, padding=0),
        nn.ReLU(),
        nn.Conv1d(in_channels=768, out_channels=768, kernel_size=1, stride=1, padding=0)))
]))

# ... existing code ...

class align_layers(nn.Module):
    """
    Align feature and apply mask ratio for LRG
    """
    def __init__(self, mask_ratio):
        super().__init__()
        self.lambda1 = mask_ratio
        self.align_layer = nn.Conv1d(in_channels=768, out_channels=768, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        mask_ratio = torch.rand(x.shape).to(x.device)
        mask_ratio = (mask_ratio >= self.lambda1).float()
        x = self.align_layer(x)
        return x * mask_ratio
```

## Usage Instructions

1. Datasets

   **CIFAR-100:** Download by code-self.

   **ImageNet-R:** Download and unzip by code-self.

   You can access it by https://github.com/hendrycks/imagenet-r

   **DomainNet:** Download by code-self but need to unzip manually.

   You can access it by https://ai.bu.edu/M3SDA/ 

2. You can start the training process with the following command, where the `--config` parameter can be replaced with any other configuration file:
    
    ```bash
    python main.py --config ./exps/clip4l2p.json
    ```
   
   or

    ```bash
    python main.py --config ./exps/clip4dual.json
    ```
    
    All available configuration files are in the `./exps/` directory.
3. hyper-parameters

    - **if_written**: whether print the output log. **True / False**.
    - **if_tsne**: whether visualize with tsne. **True / False**.
    - **alpha**: the scale of origin supervised loss. **Default: 0.5**.
    - **gamma**: the scale of contrast loss. **Default: 0.5**.
    - **mask_ratio**: mask ratio for cls token in **[0,1]**. **Default: 0.75**


## Acknowledgments

We thank the following repos providing helpful components/functions in our work.

- [PILOT](https://github.com/sun-hailong/LAMDA-PILOT)
- [l2p-pytorch](https://github.com/JH-LEE-KR/l2p-pytorch)
- [CODA-Prompt](https://github.com/GT-RIPL/CODA-Prompt)
