import torch
from torch.utils.data import TensorDataset, DataLoader


def prompt_template(text: list, target: list, domain: list):
    '''
    create prompt
    :param text: tweets
    :param target: targets
    :param domain: domains
    :return: list of prompts
    '''    
    prompt = list()
    for i in range(len(text)):
        text[i] = ' '.join(text[i])
        target[i] = ' '.join(target[i])
        prompt.append(f"The stance of text '{text[i]}' towards target '{target[i]}' on domain {domain[i]} is [MASK] from the set of 'favor', 'against', 'none'.")

    return prompt


def tokenization(tokenizer, prompt: list, y: list, batch_size: int, shuffle: bool):
    '''
    tokenize data
    :param tokenizer: BERTweet tokenizer
    :param prompt: input prompt
    :param y: ground truth
    :param batch_size: size of mini-batch
    :param shuffle: shuffle the data or not
    :return: tokenized data and ground truth as tensor
    '''    
    input_ids, attention_mask, mask_pos = [], [], []
    mask_id = tokenizer.mask_token_id
    for i in range(len(prompt)):
        encoded = tokenizer(prompt[i], max_length=512, padding='max_length', return_attention_mask=True, truncation=True, add_special_tokens=True,)
        input_ids.append(encoded.input_ids)
        attention_mask.append(encoded.attention_mask)
        mask_pos.append(encoded.input_ids.index(mask_id))

    input_ids = torch.tensor(input_ids, dtype=torch.long).to('cuda')
    attention_mask = torch.tensor(attention_mask, dtype=torch.long).to('cuda')
    mask_pos = torch.tensor(mask_pos, dtype=torch.long).to('cuda')    
    y = torch.tensor(y, dtype=torch.long).to('cuda')

    tensor_loader = TensorDataset(input_ids, attention_mask, mask_pos, y)
    data_loader = DataLoader(tensor_loader, shuffle=shuffle, batch_size=batch_size)

    return y, data_loader


def data_helper_bert(data, tokenizer, batch_size, shuffle):
    print('Loading data')
    text, tar, y, domain = data['tweet'], data['target'], data['label'], data['domain']
    prompt = prompt_template(text, tar, domain)
    y, data_loader = tokenization(tokenizer, prompt, y, batch_size, shuffle)

    return y, data_loader


def sep_test_set(input_data: list, general: bool):
    '''
    separate test set according to targets
    :param input_data: data
    :param general: generalization test set or not
    :return: list of separated test set
    '''
    if general:
        data_list = [input_data[:10238], input_data[10238:12204], input_data[12204:]]
    else:
        data_list = [input_data[:387], input_data[387:1361], input_data[1361:3041], input_data[3041:4217],
                     input_data[4217:4572], input_data[4572:4835], input_data[4835:5107], input_data[5107:5462],
                     input_data[5462:5725], input_data[5725:5997], input_data[5997:6217], input_data[6217:6502],
                     input_data[6502:6797], input_data[6797:7077], input_data[7077:7441], input_data[7441:8228],
                     input_data[8228:8837], input_data[8837:9568], input_data[9568:10237], input_data[10237:10734],
                     input_data[10734:11231], input_data[11231:11948], input_data[11948:12550]]

    return data_list