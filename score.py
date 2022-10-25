from typing import List
import torch
import torch.nn as nn


def square(outputs, targets):
    '''
    差の二乗をする
    outputs: 予測結果
    targets: 正解

    -> Batch * C
    '''
    loss = torch.square(outputs - targets)
    return loss


def order_loss(outputs, targets, device, label_dim=12) -> torch.Tensor:
    '''
    outputs: 予測結果
    targets: 正解

    -> Batch * C
    '''
    outputs_soft = nn.LogSoftmax(dim=1)(outputs)
    target_order = nn.Softmax(dim=1)(torch.FloatTensor(
        [[-abs(k - ki) for ki in range(label_dim)] for k in targets]).to(device))
    loss = torch.mul(outputs_soft, target_order)
    return loss


def clc_accuracy(conf_m: torch.Tensor) -> float:
    ans = 0
    for i in range(len(conf_m)):
        ans += conf_m[i][i]
    acc = ans / torch.sum(conf_m.view(-1))
    return acc


def clc_precision(conf_m: torch.Tensor) -> float:
    ans = 0
    denominator = torch.sum(conf_m, dim=1)
    for i in range(len(conf_m)):
        numerator = conf_m[i][i]
        if denominator[i]:
            ans += numerator / denominator[i]
    precision = ans / len(conf_m)
    return precision.item()


def clc_recall(conf_m: torch.Tensor) -> float:
    ans = 0
    denominator = torch.sum(conf_m, dim=0)
    for i in range(len(conf_m)):
        numerator = conf_m[i][i]
        if denominator[i]:
            ans += numerator / denominator[i]
    recall = ans / len(conf_m)
    return recall.item()


def create_confusion_matrix(outputs, targets, label_dim: int = 12) -> List[List[int]]:
    """混同行列を手に入れる

    Args:
        outputs (Tensor): model出力（Batch * C）
        targets (Tensor): 正解ラベル（Batch）
        label_dim (int, optional): ラベルのクラス数. Defaults to 12.

    Returns:
        List[List[int]]: 混同行列 a[i][j]:= iと予測したときにjが正解の時の数
    """

    outputs_soft = nn.Softmax(dim=1)(outputs)
    predict = torch.argmax(outputs_soft, dim=1)
    confusion_matrix = [[0] * label_dim for _ in range(label_dim)]
    for out, ans in zip(predict, targets):
        confusion_matrix[int(out.item())][int(ans.item())] += 1
    return confusion_matrix
