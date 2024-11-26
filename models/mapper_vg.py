import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_pretrained_bert.modeling import BertModel
from .visual_model.detr import build_detr
from .language_model.bert import build_bert
from .vl_transformer import build_vl_transformer
from utils.box_utils import xywh2xyxy

from .backbone.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
from .clip_vg import CLIP
from .adapter import Text_Adapter

class MaPPER(nn.Module):
    def __init__(self, args,config):
        super(MaPPER, self).__init__()
        hidden_dim = args.vl_hidden_dim
        divisor = 16 if args.dilation else 32
        self.num_visu_token = int((args.imsize / divisor) ** 2)
        self.num_text_token = args.max_query_len

        self.visumodel = vit_base(img_size=518,
                                            patch_size=14,
                                            init_values=1,
                                            block_chunks=0,
                                            output_dim=512)


        self.textmodel = build_bert(args,config)
        self.textmodel_clip = CLIP(args)
        self.text_adapter = Text_Adapter()

        # num_total = self.num_visu_token + self.num_text_token + 1
        # print("=============================35")
        # print(self.num_visu_token)
        # print(self.num_text_token)
      
        # num_total = 1371+20
        num_total = 1371 +self.num_text_token
        self.vl_pos_embed = nn.Embedding(num_total, hidden_dim)
        self.reg_token = nn.Embedding(1, hidden_dim)

        self.visu_proj = nn.Linear(self.visumodel.num_channels, hidden_dim)
        self.text_proj = nn.Linear(self.textmodel.num_channels, hidden_dim)

        self.vl_transformer = build_vl_transformer(args)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)


    def forward(self, img_data, text_data):
        bs = img_data.tensors.shape[0]

        # visual backbone
        visu_mask, visu_src = self.visumodel(img_data)
        visu_src = self.visu_proj(visu_src) # (N*B)xC

        # language bert
         # lt 
        text_data_ids = text_data.tensors
        text_data_mask = text_data.mask

        #  clip text
        text_fea_clip=self.textmodel_clip(text_data)


        text_fea = self.textmodel(text_data_ids,text_data_mask,text_fea_clip)
        # text_fea = self.textmodel(text_data_ids,text_data_mask)

     
        text_src, text_mask = text_fea.decompose()

        # lt
        text_src = torch.concat([text_src,text_fea_clip],dim=2)
        text_src = self.text_adapter(text_src)

        assert text_mask is not None
        text_src = self.text_proj(text_src)
        # permute BxLenxC to LenxBxC
        text_src = text_src.permute(1, 0, 2)
        text_mask = text_mask.flatten(1)

        # target regression token
        tgt_src = self.reg_token.weight.unsqueeze(1).repeat(1, bs, 1)
        tgt_mask = torch.zeros((bs, 1)).to(tgt_src.device).to(torch.bool)
        
        # lt
        # vl_src = torch.cat([tgt_src, text_src, visu_src], dim=0)
        vl_src = torch.cat([tgt_src, text_src, visu_src.permute(1, 0, 2)], dim=0)
        vl_mask = torch.cat([tgt_mask, text_mask, visu_mask], dim=1)
        vl_pos = self.vl_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)

        # 正确的shape,[1391,2,256]   [2,1391]  [1391,2,256]
        vg_hs = self.vl_transformer(vl_src, vl_mask, vl_pos) # (1+L+N)xBxC
        vg_hs = vg_hs[0]

        pred_box = self.bbox_embed(vg_hs).sigmoid()

        return pred_box


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
