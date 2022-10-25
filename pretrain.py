import wandb
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import argparse
from model.bert_cls import BertCLS
import dataload
import datetime
import score

parser = argparse.ArgumentParser()


def parse_args() -> argparse.Namespace:

    parser.add_argument("--data", help="データの名前")
    parser.add_argument("--arc", help="アーキテクチャ（詳しくはmodel内を参照）")
    parser.add_argument("--lr", help="学習率", type=float)
    parser.add_argument("--bs", help="バッチサイズ", type=int)
    parser.add_argument("--label_dim", help="ラべルの数", type=int, default=12)
    parser.add_argument("--base_model", help="ベースモデル", type=str, default="cl-tohoku/bert-large-japanese")
    parser.add_argument("--debug", action='store_true')


    args = parser.parse_args()
    return args


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args = parse_args()
    arc = args.arc
    lr = args.lr
    data = args.data
    bs = args.bs
    label_dim = args.label_dim
    debug = args.debug
    base_model = args.base_model

    # Dataload
    dataloader_train = dataload.load_data_pkl(data, "train", bs)
    dataloader_valid = dataload.load_data_pkl(data, "valid", bs)
    dataloader_test = dataload.load_data_pkl(data, "test", bs)
    
    from transformers import BertForPreTraining, AutoConfig
    config = AutoConfig.from_pretrained(base_model)
    model = BertForPreTraining(config)
    
    from transformers import BertJapaneseTokenizer
    tknz = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')

    from transformers import LineByLineTextDataset

    dataset = LineByLineTextDataset(
        tokenizer=tknz,
        file_path="/data/group1/z44384r/raw_corpus/qgc/qgc/for_bert_pretrain/train.src",
        block_size=512, # tokenizerのmax_length
    )
    
    from transformers import DataCollatorForLanguageModeling

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tknz, 
        mlm=True,
        mlm_probability= 0.15
    )


    
    # Wandbd
    name = f"{arc}_{data}_{bs}_{base_model}__{datetime.datetime.now()}".replace(":", "-")
    project_n = f"Bert_turn_ditect_{data}_debug" if debug else f"Bert_turn_ditect_{data}"
    wandb.init(project=project_n, name=name)
    wandb.config.lr = lr
    wandb.config.opti = "Adam"
    wandb.config.arc = arc
    wandb.config.data = data
    wandb.config.batch_size = bs
    wandb.config.label_dim = label_dim
    wandb.config.base_model = base_model

    best_valid_loss = float("INF")
    no_progress = 0
    max_step = len(dataloader_train)
    step = 0

    while 1:

        for xs, ys in tqdm(dataloader_train):
            # Train
            net.train()

            step += 1
            xs1, xmsk = [], []
            for tid in xs:
                xs1.append(torch.LongTensor(tid))
                xmsk.append(torch.LongTensor([1] * len(tid)))

            xs1 = pad_sequence(xs1, batch_first=True).to(device)
            xmsk = pad_sequence(xmsk, batch_first=True).to(device)
            ys = torch.LongTensor(ys).to(device)
            outputs = net(xs1, attention_mask=xmsk)

            optimizer.zero_grad()
            loss = loss_f(outputs, ys)
            loss.backward()

            wandb.log({
                'step': step, 
                'epoch': step / max_step, 
                'loss': loss.item()
            })
            optimizer.step()


            if step % (max_step // 10) == 0:
                # Valid
                net.eval()
                optimizer.zero_grad()
                losses = []
                with torch.no_grad():
                    for xs, ys in dataloader_valid:
                        xs1, xmsk = [], []
                        for tid in xs:
                            xs1.append(torch.LongTensor(tid))
                            xmsk.append(torch.LongTensor([1] * len(tid)))

                        xs1 = pad_sequence(xs1, batch_first=True).to(device)
                        xmsk = pad_sequence(xmsk, batch_first=True).to(device)
                        ys = torch.LongTensor(ys).to(device)
                        outputs = net(xs1, attention_mask=xmsk)
                        
                        loss = loss_f(outputs, ys)
                        losses.append(loss.item())

                val_loss_mean = sum(losses) / len(losses)
                wandb.log({'val_loss': sum(losses) / len(losses)})

                if best_valid_loss < val_loss_mean:
                    no_progress += 1

                    if no_progress > 5:
                        break

                else:
                    no_progress = 0
                    best_valid_loss = val_loss_mean
                    if not os.path.exists(f"model/saved/{project_n}/{name}"):
                        os.makedirs(f"model/saved/{project_n}/{name}")
                    torch.save(
                        net.state_dict(), f"model/saved/{project_n}/{name}/model-{step}.model")

                    # Test
                    with torch.no_grad():
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
                        mean = torch.mean(torch.Tensor(losses))
                        wandb.log({
                            "test_mean" : mean.item()
                        })
                    else:            
                        acc = score.clc_accuracy(conf_m)
                        recall = score.clc_recall(conf_m)
                        precision = score.clc_precision(conf_m)
                        wandb.log({
                            "test_acc": acc, 
                            "test_recall": recall,
                            "test_precision": precision,
                            f"test_confusion matrix_": wandb.Table(
                                columns=list(range(label_dim)),
                                data=list(conf_m)   
                            )
                        })
        else:
            continue

        break


if __name__ == "__main__":
    train()
