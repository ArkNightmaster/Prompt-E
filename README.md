# Prompt-Enhanced: Leveraging Language Representation for Prompt Continual Learning

Official code repository for the paper 

"Prompt-Enhanced: Leveraging Language Representation for Prompt Continual Learning" 

This project includes the plug-and-play method Prompt-E implemented using PyTorch.

## Key Features

- **Plug-and-Play Prompt-E Implementation**: Utilizes the PyTorch framework, making it easy to study and modify.
- **Optimized for Incremental Learning**: Enhances the performance of pretrained models in incremental learning scenarios by optimizing the update weights of prompts, suitable for handling large-scale datasets and multiple tasks.
- **Enhanced Quality of Feature Representations**: Integrates textual information into image feature representations using comparison and reconstruction techniques to refine the quality of features.
- **Focused Optimization of Prompt Efficiency**: Revisits the methods of prompt incremental learning, emphasizing the importance of optimizing prompt efficiency.

## Environment Requirements

- Python 3.9.18
- PyTorch 2.0.0
- torchvision 0.15.2
- timm 0.6.12
- For more dependencies, see `requirements.txt`

## Installation Guide

Clone the repository and install the required dependencies:

```bash
git clone https://your-repository-url.git
cd your-project-directory
pip install -r requirements.txt
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
