from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Optimizer
from statistics import mean
from sperm.methods.few_shot_classifier import FewShotClassifier
import numpy as np

def perf_measure(ground_truth, predicted):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    # Calculate TP, TN, FP, FN
    TP = torch.sum((predicted == 1) & (ground_truth == 1))
    TN = torch.sum((predicted == 0) & (ground_truth == 0))
    FP = torch.sum((predicted == 1) & (ground_truth == 0))
    FN = torch.sum((predicted == 0) & (ground_truth == 1))

    return (TP, TN, FP, FN)

def train(
      model: FewShotClassifier,
      data_loader: DataLoader,
      optimizer: Optimizer,
      LOSS_FUNCTION = None,
      DEVICE = None
    ):  
    all_loss = []
    model.train()

    with tqdm(enumerate(data_loader), total=len(data_loader), desc="Training") as tqdm_train:
        for episode_index, (
            support_images,
            support_labels,
            query_images,
            query_labels,
            _,
        ) in tqdm_train:
            optimizer.zero_grad()
            model.process_support_set(support_images.to(DEVICE), support_labels.to(DEVICE))
            classification_scores = model(query_images.to(DEVICE))
            #query_labels = two_columns_labels(query_labels).float() #######
            loss = LOSS_FUNCTION(classification_scores, query_labels.to(DEVICE))
            loss.backward()
            optimizer.step()

            all_loss.append(loss.item())
            tqdm_train.set_postfix(loss=mean(all_loss))

    return mean(all_loss)

def evaluate_on_one_task(
    model: FewShotClassifier,
    support_images: Tensor,
    support_labels: Tensor,
    query_images: Tensor,
    query_labels: Tensor,
) -> Tuple[int, int, int, int, int, int]:
    """
    Returns the number of correct predictions of query labels, and the total number of
    predictions.
    """
    model.process_support_set(support_images, support_labels)
    TP, TN, FP, FN = perf_measure(query_labels, torch.max(model(query_images).detach().data,1,)[1])


    return ((
        torch.max(
            model(query_images).detach().data,
            1,
        )[1]
        == query_labels
    ).sum().item(), len(query_labels), TP, TN, FP, FN)


def evaluate(
    model: FewShotClassifier,
    data_loader: DataLoader,
    device: str = "cuda",
    use_tqdm: bool = True,
    tqdm_prefix: Optional[str] = None,
) -> Tuple[float, int, int, int, int]:
    """
    Evaluate the model on few-shot classification tasks
    Args:
        model: a few-shot classifier
        data_loader: loads data in the shape of few-shot classification tasks*
        device: where to cast data tensors.
            Must be the same as the device hosting the model's parameters.
        use_tqdm: whether to display the evaluation's progress bar
        tqdm_prefix: prefix of the tqdm bar
    Returns:
        average classification accuracy
    """
    # We'll count everything and compute the ratio at the end
    total_predictions = 0
    correct_predictions = 0
    total_TP = 0
    total_TN = 0
    total_FP = 0
    total_FN = 0

    # eval mode affects the behaviour of some layers (such as batch normalization or dropout)
    # no_grad() tells torch not to keep in memory the whole computational graph
    model.eval()
    with torch.no_grad():
        with tqdm(enumerate(data_loader), total=len(data_loader), disable=not use_tqdm, desc=tqdm_prefix,) as tqdm_eval:
            for _, (
                support_images,
                support_labels,
                query_images,
                query_labels,
                _,
            ) in tqdm_eval:
                correct, total, TP, TN, FP, FN = evaluate_on_one_task(model,support_images.to(device),support_labels.to(device),query_images.to(device),query_labels.to(device),)
                total_predictions += total
                correct_predictions += correct
                total_TP += TP
                total_TN += TN
                total_FP += FP
                total_FN += FN

                # Log accuracy in real time
                tqdm_eval.set_postfix(accuracy=correct_predictions / total_predictions)

    return ((correct_predictions / total_predictions), total_TP.item(), total_TN.item(), total_FP.item(), total_FN.item())