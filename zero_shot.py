import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import clip
from PIL import Image
import numpy as np
from tqdm import tqdm

def evaluate_zero_shot(dataset='cifar100', batch_size=128, device='cuda'):
    
    model, preprocess = clip.load('ViT-B/16', device=device)
    
    if dataset == 'cifar100':
        test_dataset = datasets.CIFAR100(root='./data/cifar100', train=False, transform=preprocess, download=True)
        classnames = test_dataset.classes
    elif dataset == 'domainnet':
        test_dataset = datasets.ImageFolder(root='./data/domainnet/test', transform=preprocess)
        classnames = test_dataset.classes
    else:
        raise ValueError(f'Unsupported dataset: {dataset}')
        
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Build text prompts
    text_prompts = [f"a photo of a {classname}" for classname in classnames]
    text_tokens = clip.tokenize(text_prompts).to(device)
    
    # Extract text features
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = F.normalize(text_features, dim=1)
    
    # Evaluate
    model.eval()
    total = 0
    correct = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Extract image features
            image_features = model.encode_image(images)
            image_features = F.normalize(image_features, dim=1)
            
            # Calculate similarity
            similarity = image_features @ text_features.T
            
            # Get prediction results
            pred = similarity.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    
    accuracy = 100 * correct / total
    print(f'{dataset} Zero-shot Accuracy: {accuracy:.2f}%')
    return accuracy

if __name__ == '__main__':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Evaluate CIFAR-100
    cifar_acc = evaluate_zero_shot('cifar100', device=device)
    
    # Evaluate DomainNet
    # domainnet_acc = evaluate_zero_shot('domainnet', device=device)
