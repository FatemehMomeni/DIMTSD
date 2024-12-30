import torch
import torch.nn as nn
from transformers import AutoModel, BertweetTokenizer


class StanceClassifier(nn.Module):
    def __init__(self, num_labels):
        super(StanceClassifier, self).__init__()
        self.dropout = nn.Dropout(0.)
        self.relu = nn.ReLU()
        self.bert = AutoModel.from_pretrained("vinai/bertweet-base")
        self.tokenizer = BertweetTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)
        self.bert.pooler = None
        self.linear = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.out = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        last_hidden = self.bert(input_ids, attention_mask, token_type_ids)
        cls = last_hidden[0][:, 0]
        query = self.dropout(cls)
        linear = self.relu(self.linear(query))
        out = self.out(linear)

        return out
