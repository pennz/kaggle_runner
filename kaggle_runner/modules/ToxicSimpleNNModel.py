import torch
import torch.nn as nn
from transformers import XLMRobertaModel, DistilBertModel
from transformers import XLNetConfig, XLNetModel, XLNetPreTrainedModel

BACKBONE_PATH = 'xlm-roberta-large'


class ToxicSimpleNNModel(nn.Module):
    def __init__(self, use_aux=True):
        super(ToxicSimpleNNModel, self).__init__()
        self.backbone = XLMRobertaModel.from_pretrained(BACKBONE_PATH)
        self.dropout = nn.Dropout(0.3)
        aux_len = 0

        if use_aux:
            aux_len = 5
        self.linear = nn.Linear(
            in_features=self.backbone.pooler.dense.out_features*2,
            out_features=2+aux_len,
        )

    def forward(self, input_ids, attention_masks):
        bs, seq_length = input_ids.shape
        seq_x, _ = self.backbone(
            input_ids=input_ids, attention_mask=attention_masks)
        apool = torch.mean(seq_x, 1)
        mpool, _ = torch.max(seq_x, 1)
        x = torch.cat((apool, mpool), 1)
        x = self.dropout(x)

        return self.linear(x)
