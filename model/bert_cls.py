#!/usr/bin/python

from tabnanny import check
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

# Define model
class BertCLS(nn.Module):
    def __init__(self, feature_dim, label_dim, base, checkpoint=None):
        super(BertCLS, self).__init__()
        
        self.feature_dim = feature_dim
        
    
        self.config = AutoConfig.from_pretrained(base)
        self.config.output_hidden_states = True

        self.base_model = AutoModel.from_pretrained(base, config=self.config)
        self.linear = nn.Linear(feature_dim, label_dim)

    def forward(self, input_ids, attention_mask):
        # BERT出力
        outputs = self.base_model(input_ids, attention_mask=attention_mask)

        # BERT最終層出力だけとってくる
        outputs = outputs['last_hidden_state']
        # 最終層のCLS部分だけとってくる
        cls_outputs =  torch.reshape(outputs[:,0,:], (-1, self.feature_dim))

        # MLP部分
        outputs = self.linear(cls_outputs)
        return outputs
    
