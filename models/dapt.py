import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Global_Adapter(nn.Module):
    def __init__(self,
                 bert_dim=None,
                 clip_dim=None,
                 out_dim=None,
                 rank=None,
                 dropout=0.0,
                 adapter_layernorm_option="in",
                 use_square=False, ):
        super().__init__()
        self.bert_dim = bert_dim
        self.clip_dim = clip_dim
        self.rank = rank
        self.use_square = use_square
        
        # _before
        self.adapter_layernorm_option = adapter_layernorm_option
        self.bert_adapter_layer_norm_before = None
        self.clip_adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.bert_adapter_layer_norm_before = nn.LayerNorm(self.bert_dim)
            self.clip_adapter_layer_norm_before = nn.LayerNorm(self.clip_dim)

        # clip
        self.clip_down_proj = nn.Linear(self.clip_dim, self.rank)
        self.scale = nn.Linear(self.clip_dim, 1)

        # bert
        self.bert_down_proj = nn.Linear(self.bert_dim, self.rank)
        if out_dim is None:
            self.up_proj = nn.Linear(self.rank, self.bert_dim)
        else:
            self.up_proj = nn.Linear(self.rank, out_dim)
        
        # share
        self.non_linear_func = nn.ReLU()
        self.dropout = dropout

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.bert_down_proj.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.clip_down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.bert_down_proj.bias)
            nn.init.zeros_(self.clip_down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)
            nn.init.kaiming_uniform_(self.scale.weight, a=math.sqrt(5))
            nn.init.zeros_(self.scale.bias)

    def forward(self, bert_token, text_fea_clip, add_residual=False, residual=None):
        residual = bert_token if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            bert_token = self.bert_adapter_layer_norm_before(bert_token)
            text_fea_clip = self.clip_adapter_layer_norm_before(text_fea_clip.to(torch.float32))

        scale = F.relu(self.scale(text_fea_clip))

        down = self.bert_down_proj(bert_token)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)
        up = up * scale
        
        if self.adapter_layernorm_option == 'out':
            up = self.bert_adapter_layer_norm_before(up)
        if add_residual:
            output = up + residual
        else:
            output = up
        return output

# if __name__ == "__main__":
#     clip_token = torch.randn(8, 20, 512)
#     bert_token = torch.randn(8, 20, 768)
#     adapter = DAPTVG(bert_dim=768, clip_dim=512, out_dim=768, rank=64)
#     adapted_bert_token = adapter(bert_token, clip_token)
#     print(adapted_bert_token.shape)
