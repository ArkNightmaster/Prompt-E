import os
import numpy as np
import torch
from datetime import datetime


def count_parameters(model, trainable=False, component=None):
    """
    Count model parameters
    
    Args:
        model: model
        trainable: whether to only count trainable parameters
        component: specify which component to count, options are 'prompt', 'fc', None
                  None means count all parameters
    
    Returns:
        parameter count (in millions)
    """
    if component == 'prompt':
        # Count prompt parameters
        if trainable:
            return sum(p.numel() for name, p in model.named_parameters() 
                      if ('prompt' in name or 'e_prompt' in name or 'g_prompt' in name) and p.requires_grad) / 1e6
        return sum(p.numel() for name, p in model.named_parameters() 
                  if ('prompt' in name or 'e_prompt' in name or 'g_prompt' in name)) / 1e6
    elif component == 'fc':
        # Count fc parameters
        if trainable:
            return sum(p.numel() for name, p in model.named_parameters() 
                      if ('fc' in name or 'classifier' in name or 'head' in name) and p.requires_grad) / 1e6
        return sum(p.numel() for name, p in model.named_parameters() 
                  if ('fc' in name or 'classifier' in name or 'head' in name)) / 1e6
    else:
        # Count all parameters
        if trainable:
            return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        return sum(p.numel() for p in model.parameters()) / 1e6


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()

def print_file(string, root, if_print=True, if_write=True):

    # Get current time and format as string (e.g., 'YYYY-MM-DD HH:MM:SS' format)
    time_format = "%Y-%m-%d %H:%M:%S"
    current_time = datetime.now().strftime(time_format)
    if if_write:
        # Open file for writing. If file doesn't exist, it will be created
        with open(root, 'a') as file:
            # Write string to file
            file.write(f"{current_time} => {string}\n")

    if if_print is True:
        print(f"{current_time} => {string}")
    else:
        pass

def order_num2label(list1: list, list2: list) -> dict:
    """
    Example:
    list1 = [0, 2, 1, 2, 3, 0]
    list2 = ['a', 'c', 'b', 'c', 'd', 'a']
    return map = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}
    """

    # Create an empty list to store results
    result = []

    # Create a dictionary to map numbers in list1 to letters in list2
    mapping = {}

    for num, letter in zip(list1, list2):
        # If the number doesn't have a corresponding key in the dictionary, add the letter to the result list
        if num not in mapping:
            mapping[num] = letter
            result.append(letter)

    return mapping


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot


@torch.no_grad()
def cosine_similarity(cls_token, prompt_tokens, rt_weight=False):
    """
    cls_token: Batch, 1, 768
    prompt_token: Batch, total_token_length, 768
    T: weight of cls_token
    """
    # Calculate cosine similarity between cls_token and each prompt_token
    # Need to adapt cosine similarity calculation to support batch operations
    # The method used here is to expand cls_token to match the shape of prompt_tokens
    prompt_length = prompt_tokens.shape[1]
    total_length = 1 + prompt_length

    cls_token = cls_token.unsqueeze(1)
    cls_token_expanded = cls_token.expand(-1, prompt_length, -1)  # Expand cls_token to match prompt_tokens shape

    similarities = torch.nn.functional.cosine_similarity(prompt_tokens, cls_token_expanded, dim=-1)

    # Use similarities directly as weights
    weights = similarities

    # Normalize weights so that the sum of weights for each sample is 1
    weights /= weights.sum(dim=1, keepdim=True)

    # Use these weights to compute weighted average of prompt_tokens
    # To perform weighted average, need to expand weights shape to match the last dimension of prompt_tokens
    weighted_prompt_tokens = prompt_tokens * weights.unsqueeze(-1)

    # To maintain consistent magnitude, need to scale cls_token
    weighted_cls_tokens = cls_token * prompt_length / total_length

    # Combine cls_token and weighted_prompt_tokens
    # In PyTorch, can use torch.cat for concatenation, but need to adjust dimensions to match
    combined_features = torch.cat([weighted_cls_tokens, weighted_prompt_tokens], dim=1)

    if rt_weight:
        return combined_features, weights
    else:
        return combined_features

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def accuracy(y_pred, y_true, nb_old, init_cls=10, increment=10):
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {}
    all_acc["total"] = np.around(
        (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
    )

    # Grouped accuracy, for initial classes
    idxes = np.where(
        np.logical_and(y_true >= 0, y_true < init_cls)
    )[0]
    label = "{}-{}".format(
        str(0).rjust(2, "0"), str(init_cls - 1).rjust(2, "0")
    )
    all_acc[label] = np.around(
        (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
    )
    # for incremental classes
    for class_id in range(init_cls, np.max(y_true), increment):
        idxes = np.where(
            np.logical_and(y_true >= class_id, y_true < class_id + increment)
        )[0]
        label = "{}-{}".format(
            str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
        )
        all_acc[label] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )

    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]

    all_acc["old"] = (
        0
        if len(idxes) == 0
        else np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
    )

    # New accuracy
    idxes = np.where(y_true >= nb_old)[0]
    all_acc["new"] = np.around(
        (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
    )

    return all_acc


def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)

def show_images(img_tensor):
    from matplotlib.pyplot import imshow,show
    tensor = img_tensor.numpy()
    tensor = np.transpose(tensor, (1, 2, 0))
    imshow(tensor)
    show()
