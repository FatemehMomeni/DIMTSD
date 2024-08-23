import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F


class StanceClassifier(nn.Module):
    def __init__(self):
        super(StanceClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased").to('cuda')
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        labels = ['against', 'none', 'favor']        
        encoded = self.tokenizer(labels, max_length=512, padding='max_length', truncation=True, add_special_tokens=True, return_tensors='pt').to('cuda')
        output = self.bert(encoded.input_ids, encoded.attention_mask)[0]
        self.label_embeds = output[:, 1] # tensor(3*768)        

    def forward(self, input_ids, attention_mask, mask_pos):
        last_hidden = self.bert(input_ids, attention_mask)[0]
        similarities = list()
        for i in range(len(input_ids)):
            h_mask = last_hidden[i, mask_pos[i]]
            temp = torch.zeros(1, 3).to('cuda')
            for j in range(3):
                temp[0, j] = F.cosine_similarity(h_mask, self.label_embeds[j], dim=0)
            similarities.append([temp[0, 0], temp[0, 1], temp[0, 2]])
        
        return torch.tensor(similarities).to('cuda')