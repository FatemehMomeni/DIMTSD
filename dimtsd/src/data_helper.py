import torch
from torch.utils.data import TensorDataset, DataLoader


def tokenization(data: dict, tokenizer):
    '''
    tokenize data
    :param data: data
    :param tokenizer: BERTweet tokenizer
    :return: tokenized data
    '''
    inputs = list()
    for i in range(len(data['tweet'])):
        inputs.append(f"target: {data['target'][i]} tweet: {data['tweet'][i]} domain: {data['domain'][i]} ")

    encoded = tokenizer(inputs, add_special_tokens=True, return_attention_mask=True, max_length=128,
                        padding='max_length', truncation=True, return_tensors='pt').to('cuda')
    labels = torch.tensor([label for label in data['stance']]).to('cuda')

    return encoded.input_ids, encoded.attention_mask, encoded.token_type_ids, labels


def data_helper(cleaned_train: dict, cleaned_eval: dict, cleaned_test: dict, cleaned_test_gen: dict, tokenizer):
    print('* Tokenization...')
    train_input_ds, train_attention_mask, train_token_type_ids, y_train = tokenization(cleaned_train, tokenizer)
    eval_input_ds, eval_attention_mask, eval_token_type_ids, y_eval = tokenization(cleaned_eval, tokenizer)
    test_input_ds, test_attention_mask, test_token_type_ids, y_test = tokenization(cleaned_test, tokenizer)
    test_gen_input_ds, test_gen_attention_mask, test_gen_token_type_ids, y_test_gen = tokenization(cleaned_test_gen,
                                                                                                   tokenizer)

    train = [train_input_ds, train_attention_mask, train_token_type_ids, y_train]
    val = [eval_input_ds, eval_attention_mask, eval_token_type_ids, y_eval]
    test = [test_input_ds, test_attention_mask, test_token_type_ids, y_test]
    test_gen = [test_gen_input_ds, test_gen_attention_mask, test_gen_token_type_ids, y_test_gen]

    return train, val, test, test_gen


def data_loader(x_all: list, batch_size: int, mode: str, model_name: str, **kwargs):
    '''
    create datasets
    :param x_all: list of tokenized data
    :param batch_size: size of mini-batch
    :param mode: train, eval, or test
    :param model_name: teacher or student
    :return data loader and labels
    '''
    if model_name == 'student' and mode == 'train':
        y2 = torch.tensor(kwargs['y_train2'], dtype=torch.float).to('cuda')  # load teacher predictions
        tensor_loader = TensorDataset(x_all[0], x_all[1], x_all[2], x_all[3], y2)
    else:
        tensor_loader = TensorDataset(x_all[0], x_all[1], x_all[2], x_all[3])

    if mode == 'train':
        data_loader = DataLoader(tensor_loader, shuffle=True, batch_size=batch_size)
        data_loader_distill = DataLoader(tensor_loader, shuffle=False, batch_size=batch_size)
        return data_loader, x_all[3], data_loader_distill
    else:
        data_loader = DataLoader(tensor_loader, shuffle=False, batch_size=batch_size)
        return data_loader, x_all[3]


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
