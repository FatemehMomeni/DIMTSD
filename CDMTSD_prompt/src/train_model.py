import torch
import torch.nn as nn
import random
import numpy as np
import json
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def run_classifier():
    lr = 3e-5
    batch_size = 4
    total_epoch = 5
    random_seeds = [1, 2, 4, 5, 9, 10]

    # create normalization dictionary for preprocessing
    with open("../noslang_data.json", "r") as f:
        data1 = json.load(f)
    data2 = {}
    with open("../emnlp_dict.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            row = line.split('\t')
            data2[row[0]] = row[1].rstrip()
    normalization_dict = {**data1, **data2}

    # dataset paths
    train_path = '../datasets/train.csv'
    eval_path = '../datasets/validation.csv'
    test_path = '../datasets/test.csv'
    test_general_path = '../datasets/generalization test.csv'

    # preprocess data
    # temp_dict = {'tweet': [], 'target': [], 'label': []}
    train = clean_all(train_path, normalization_dict)
    val = clean_all(eval_path, normalization_dict)
    test = clean_all(test_path, normalization_dict)
    test_general = clean_all(test_general_path, normalization_dict)    
    
    best_val = []
    best_result = {'single': [], 'general': []}
    for seed in random_seeds:
        print("current random seed: ", seed)

        # set up the random seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        model = StanceClassifier()

        # prepare for model
        y_train, train_loader = data_helper_bert(train, model.tokenizer, batch_size, True)
        y_val, val_loader = data_helper_bert(val, model.tokenizer, batch_size, False)
        y_test, test_loader = data_helper_bert(test, model.tokenizer, batch_size, False)
        y_test_gen, test_loader_gen = data_helper_bert(test_general, model.tokenizer, batch_size, 'test')

        optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if n.startswith('bert.encoder')] , 'lr': lr},
                {'params': [p for n, p in model.named_parameters() if n.startswith('bert.pooler')] , 'lr': 1e-3},
        ]

        loss_function = nn.CrossEntropyLoss(reduction='sum')
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)

        sum_loss, val_f1_average, train_predictions = [], [], []
        test_f1_average = {'single': [[] for _ in range(23)], 'general': [[] for _ in range(3)]}

        for epoch in range(total_epoch):
            print('Epoch:', epoch)
            # training phase
            train_loss = list()
            model.train()
            for input_ids, attention_mask, mask_pos, label in train_loader:
                optimizer.zero_grad()
                output = model(input_ids, attention_mask, mask_pos)
                loss = loss_function(output, label)
                loss.requires_grad = True
                loss.backward() # retain_graph=True
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                train_loss.append(loss.item())
            sum_loss.append(sum(train_loss) / len(train['tweet']))
            print(sum_loss[epoch])  # summation of loss in this epoch

            # evaluation phase
            model.eval()
            val_predictions = list()
            with (torch.no_grad()):
                for input_ids, attention_mask, mask_pos, label in val_loader:
                    prediction = model(input_ids, attention_mask, mask_pos)
                    val_predictions.append(prediction)                
                prediction = torch.cat(val_predictions, 0)
                acc, f1_average, precision, recall = compute_f1(prediction, y_val)
                val_f1_average.append(f1_average)

            # test phase
            y_test_list = sep_test_set(y_test, False)
            y_test_gen_list = sep_test_set(y_test_gen, True)

            with torch.no_grad():
                test_predictions = list()
                for input_ids, attention_mask, mask_pos, label in test_loader:
                    prediction = model(input_ids, attention_mask, mask_pos)
                    test_predictions.append(prediction)
                prediction = torch.cat(test_predictions, 0)
                prediction_list = sep_test_set(prediction, False)
                
                test_predictions.clear()                                
                for input_ids, attention_mask, mask_pos, label in test_loader_gen:
                    prediction = model(input_ids, attention_mask, mask_pos, None)
                    test_predictions.append(prediction)
                prediction = torch.cat(test_predictions, 0)
                prediction_list_gen = sep_test_set(prediction, True)

                # measuring F1
                test_predictions = list()
                for ind in range(len(y_test_list)):
                    prediction = prediction_list[ind]
                    test_predictions.append(prediction)
                    acc, f1_average, precision, recall = compute_f1(prediction, y_test_list[ind])
                    test_f1_average['single'][ind].append(f1_average)

                # measuring F1 for generalization dataset
                test_predictions = list()
                for ind in range(len(y_test_gen_list)):
                    prediction = prediction_list_gen[ind]
                    test_predictions.append(prediction)
                    acc, f1_average, precision, recall = compute_f1(prediction, y_test_gen_list[ind])
                    test_f1_average['general'][ind].append(f1_average)

        # model that performs best on the eval set is evaluated on the test set
        best_epoch = [index for index, v in enumerate(val_f1_average) if v == max(val_f1_average)][-1]
        best_result['single'].append([f1[best_epoch] for f1 in test_f1_average['single']])
        best_result['general'].append([f1[best_epoch] for f1 in test_f1_average['general']])

         torch.save(model.state_dict(), f'/kaggle/working/pccl_{seed}.pt')

        print("*" * 20)
        print(f"dev results with seed {seed} on all epochs")
        print(val_f1_average)
        best_val.append(val_f1_average[best_epoch])

        print("*" * 20)
        print(f"test results with seed {seed} on all epochs")
        print(max(best_result['single']))
        print(best_result['single'])
        print("generalization")
        print(max(best_result['general']))
        print(best_result['general'])


if __name__ == "__main__":
    run_classifier()

