# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F

from torch import nn
from typing import Dict, List

from utils.misc import NestedTensor, is_main_process
# from .position_encoding import build_position_encoding

# from pytorch_pretrained_bert.modeling import BertModel
from transformers.models.bert.configuration_bert import BertConfig
from models.language_model.med import BertModel


# lt
from urllib.parse import urlparse
from timm.models.hub import download_cached_file
import os
from models.adapter import MOE_Adapter,Bert_Adapter

class BERT(nn.Module):
    def __init__(self, name: str, train_bert: bool, hidden_dim: int, max_len: int, enc_num, med_config = 'configs/med_config.json',config = None):
        super().__init__()
        if name == 'bert-base-uncased':
            self.num_channels = 768
        else:
            self.num_channels = 1024
        self.enc_num = enc_num

        # lt
        self.config = config
        med_config = BertConfig.from_json_file(med_config)
      
     

        # lt
        self.bert = BertModel.from_pretrained('/share/home/liuting/transvg_data/bert-base-uncased',config=med_config, add_pooling_layer=False, adapter_config=config)

        # if not train_bert:
        #     for parameter in self.bert.parameters():
        #         parameter.requires_grad_(False)

    def forward(self,text_data_tensors, text_data_mask,text_fea_clip=None):

        if self.enc_num > 0:

            # lt
            # xs,mha_mlp_adapter,embedding_output = self.bert(text_data_tensors, token_type_ids=None, attention_mask=text_data_mask, return_dict = True, mode = 'text')

            # all_encoder_layers,_ = self.bert(tensor_list.tensors, token_type_ids=None, attention_mask=tensor_list.mask, return_dict = True, mode = 'text')
            # use the output of the X-th transformer encoder layers
            # xs = all_encoder_layers[self.enc_num - 1]
            xs = self.bert(text_data_tensors, token_type_ids=None, attention_mask=text_data_mask, return_dict = True,text_fea_clip=text_fea_clip)
        
        else:
            xs = self.bert.embeddings.word_embeddings(text_data_tensors)

        mask = text_data_mask.to(torch.bool)
        mask = ~mask
        out = NestedTensor(xs, mask)

        return out

def build_bert(args,config):
    # position_embedding = build_position_encoding(args)
    train_bert = args.lr_bert > 0

    # lt
    
    bert = BERT(args.bert_model, train_bert, args.hidden_dim, args.max_query_len, args.bert_enc_num,config = config)
    # model = Joiner(bert, position_embedding)
    # model.num_channels = bert.num_channels
    return bert



