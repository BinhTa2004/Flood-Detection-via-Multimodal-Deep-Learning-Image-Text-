import torch
import torch.nn as nn
from transformers import BertModel, ViTModel
from transformers.modeling_outputs import SequenceClassifierOutput

class CombinedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')

        bert_hidden = self.bert.config.hidden_size
        vit_hidden = self.vit.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(bert_hidden + vit_hidden, 512),
            # Tổng chiều đầu ra của torch.cat((bert_cls, vit_cls), dim=1) = bert_dim + vit_dim → Phải khớp với input size của nn.Linear
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_cls = bert_out.last_hidden_state[:, 0, :]  # CLS token

        vit_out = self.vit(pixel_values=pixel_values)
        vit_cls = vit_out.last_hidden_state[:, 0, :]

        combined = torch.cat((bert_cls, vit_cls), dim=1)
        logits = self.classifier(combined)
        # return logits.squeeze(1) # batch size = 1: return logits.view(-1)
        return SequenceClassifierOutput(logits=logits)
