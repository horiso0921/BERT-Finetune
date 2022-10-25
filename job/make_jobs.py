import os
from omegaconf import DictConfig, OmegaConf
import hydra

PREFIX = """#!/bin/bash -x
#PJM -L rscgrp=cx-small
#PJM -L node=1
#PJM -L elapse=12:00:00
#PJM -j
#PJM -o {}/{}.log

module load cuda/10.2.89_440.33.01 openmpi_cuda/4.0.4 nccl/2.7.3
module load gcc
eval "$(~/miniconda3/bin/conda shell.bash hook)"

conda activate pytorch-ad-3.8
pip install --upgrade wandb
cd /data/group1/z44384r/sentence-classification/BERT-doccls/src && """

@hydra.main(config_name="config.yaml", config_path="conf")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    arcs = cfg.train.arcs
    data = cfg.train.data
    label_dim = cfg.train.label_dim
    bs = cfg.train.bs
    lrs = cfg.train.lrs
    
    exec_files = []
    for arc in arcs:
        for lr in lrs:
            out_fname = f"{arc}_{data}_{label_dim}_{bs}_{lr}.sh"
            exec_files.append(out_fname)
            os.mkdir(out_fname[:-3])
            with open(out_fname, "w", encoding="utf-8") as t:
                t.write(PREFIX.format(out_fname[:-3], out_fname[:-3]))
                t.write(f"python train.py --data {data} --label_dim {label_dim} --arc {arc} --lr {lr} --bs {bs}")

    with open("jobs.sh", "w", encoding="utf-8") as t:
        for exec_file in exec_files:
            t.write(f"pjsub {exec_file} && sleep 3 \n")

if __name__ == "__main__":
    my_app()