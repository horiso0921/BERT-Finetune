import wandb
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import argparse
from model.bert_mse import BertMSE
import dataload
import datetime
import score

parser = argparse.ArgumentParser()

# sample
# python test.py --arc bertlabel --checkpoint "Bert_turn_ditect/bertlabel_qgc_turn_2022-10-19 14-29-06.313732/model-1000.model" --bs 32

def parse_args() -> argparse.Namespace:

    parser.add_argument("--arc", help="アーキテクチャ（詳しくはmodel内を参照）")
    parser.add_argument("--checkpoint", help="モデルのパス model/saved/ の後", type=str)
    parser.add_argument("--bs", help="バッチサイズ", type=int, default=32)
    parser.add_argument("--label_dim", help="ラべルの数", type=int, default=12)

    args = parser.parse_args()
    return args


def test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args = parse_args()
    arc = args.arc
    bs = args.bs
    checkpoint = args.checkpoint
    label_dim = args.label_dim
    
    # Dataload
    dataloader_test = dataload.load_data_pkl("qgc_turn", "test", bs)

    # アーキテクチャ
    if arc == "bertmse":
        net = BertMSE(feature_dim=768, label_dim=1, checkpoint=True)
        net.to(device)
        def loss_f(x, y): return torch.sum(score.square(x.view(-1), y)) / bs
    elif arc == "bertlabel":
        net = BertMSE(feature_dim=768, label_dim=label_dim, checkpoint=True)
        net.to(device)
        def loss_f(x, y): return torch.sum(F.cross_entropy(x, y))
    elif arc == "bertlabel_order":
        net = BertMSE(feature_dim=768, label_dim=label_dim, checkpoint=True)
        net.to(device)

        def loss_f(x, y): return torch.sum(
            score.order_loss(x, y, device, label_dim=label_dim).view(-1))
    else:
        raise ValueError("アーキテクチャが無いぞ")
    
    m = torch.load(f"model/saved/{checkpoint}")
    net.load_state_dict(m)

    # Wandbd
    name = f"{checkpoint}_{datetime.datetime.now()}".replace(":", "-")
    project_n = "Bert_turn_ditect_test"
    wandb.init(project=project_n, name=name)
    wandb.config.arc = arc
    wandb.config.checkpoint = checkpoint
    wandb.config.batch_size = bs


    with torch.no_grad():
        # Test
        conf_m = torch.zeros(label_dim, label_dim)
        losses = []
        for xs, ys in dataloader_test:
            xs1, xmsk = [], []
            for tid in xs:
                xs1.append(torch.LongTensor(tid))
                xmsk.append(torch.LongTensor([1] * len(tid)))

            xs1 = pad_sequence(
                xs1, batch_first=True).to(device)
            xmsk = pad_sequence(
                xmsk, batch_first=True).to(device)
            ys = torch.LongTensor(ys).to(device)

            outputs = net(xs1, attention_mask=xmsk)

            if arc == "bertmse":
                loss = loss_f(outputs, ys)
                losses.append(loss.item())

            else:
                conf_m_tmp = torch.Tensor(
                    score.create_confusion_matrix(outputs, ys, label_dim))
                conf_m = torch.add(conf_m, conf_m_tmp)

        if arc == "bertmse":
            mean = torch.mean(losses)
            wandb.log({
                "mean" : mean
            })
        else:            
            acc = score.clc_accuracy(conf_m)
            recall = score.clc_recall(conf_m)
            precision = score.clc_precision(conf_m)
            wandb.log({
                "acc": acc, 
                "recall": recall,
                "precision": precision,
                "confusion matrix": wandb.Table(
                    columns=list(range(label_dim)),
                    data=list(conf_m)   
                )
            })


if __name__ == "__main__":
    test()
