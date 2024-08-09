import argparse
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import data_helper as dh
import model_eval
import modeling
import preprocessing as pp

import warnings
warnings.filterwarnings('ignore')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="teacher", help="teacher or student")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--theta", type=float, default=0.6, help="AKD parameter")
    args = parser.parse_args()

    lr = args.lr
    batch_size = args.batch_size
    total_epoch = args.epochs
    model_name = args.model_name
    random_seeds = [1, 2, 4, 5, 9, 10]
    dropout = args.dropout
    alpha = args.alpha
    theta = args.theta

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

    best_val = []
    best_result = {'single': [], 'general': []}
    metrics = {'single': [], 'general': []}

    for seed in random_seeds:
        print('-' * 5 + f" Current Random Seed: {seed} " + '-' * 5)

        train_filename = '../datasets/train.csv'
        eval_filename = '../datasets/validation.csv'
        test_filename = '../datasets/test.csv'
        test_gen_filename = '../datasets/generalization test.csv'
        print('* Loading data...')
        cleaned_train = pp.clean_all(train_filename, normalization_dict)
        cleaned_eval = pp.clean_all(eval_filename, normalization_dict)
        cleaned_test = pp.clean_all(test_filename, normalization_dict)
        cleaned_test_gen = pp.clean_all(test_gen_filename, normalization_dict)

        if model_name == 'student':
            y_train_stu = torch.load(f'../teacher_models/seed{seed}.pt')  # load teacher predictions

        # set up the random seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        num_labels = 3  # Favor, Against and None
        model = modeling.StanceClassifier(num_labels).to('cuda')

        # prepare for model
        train, val, test, test_gen = dh.data_helper(cleaned_train, cleaned_eval, cleaned_test, cleaned_test_gen,
                                                    model.tokenizer)

        print('* Prepare Datasets...')
        if model_name == 'teacher':
            train_loader, y_train, train_loader_distill = dh.data_loader(train, batch_size, 'train', model_name)
        else:
            train_loader, y_train, train_loader_distill = dh.data_loader(train, batch_size, 'train', model_name, y_train2=y_train_stu)
        eval_loader, y_eval = dh.data_loader(val, batch_size, 'val', model_name)
        test_loader, y_test = dh.data_loader(test, batch_size, 'test', model_name)
        test_gen_loader, y_test_gen = dh.data_loader(test_gen, batch_size, 'test', model_name)

        for n, p in model.named_parameters():
            if "bert.embeddings" in n:
                p.requires_grad = False

        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if n.startswith('bert.encoder')], 'lr': lr},
            {'params': [p for n, p in model.named_parameters() if n.startswith('bert.pooler')], 'lr': 1e-3},
            {'params': [p for n, p in model.named_parameters() if n.startswith('linear')], 'lr': 1e-3},
            {'params': [p for n, p in model.named_parameters() if n.startswith('out')], 'lr': 1e-3}
        ]

        loss_function = nn.CrossEntropyLoss(reduction='sum')
        if model_name == 'student':
            loss_function2 = nn.KLDivLoss(reduction='sum')

        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)

        sum_loss, sum_loss2, val_f1_average, train_preds_distill, train_cls_distill = [], [], [], [], []
        test_f1_average = {'single': [[] for _ in range(23)], 'general': [[] for _ in range(3)]}

        for epoch in range(total_epoch):
            print(f'Epoch: {epoch}')
            train_loss, train_loss2 = [], []
            model.train()
            if model_name == 'teacher':
                for input_ids, attention_mask, token_type_ids, label in train_loader:
                    optimizer.zero_grad()
                    output = model(input_ids, attention_mask, token_type_ids)
                    loss = loss_function(output, label)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    train_loss.append(loss.item())
            else:
                for input_ids, attention_mask, token_type_ids, label, label2 in train_loader:
                    optimizer.zero_grad()
                    output1 = model(input_ids, attention_mask, token_type_ids)
                    output2 = output1
                    # 3. proposed AKD
                    output2 = torch.empty(output1.shape).fill_(0.).cuda()
                    for ind in range(len(label2)):
                        soft = max(F.softmax(label2[ind]))
                        if soft <= theta:
                            rrand = random.uniform(2, 3)  # parameter b1 and b2 in paper
                        elif theta + 0.2 > soft > theta:  # parameter a1 and a2 are theta and theta+0.2 here
                            rrand = random.uniform(1, 2)
                        else:
                            rrand = 1
                        label2[ind] = label2[ind] / rrand
                        output2[ind] = output1[ind] / rrand
                    label2 = F.softmax(label2)

                    loss = (1 - alpha) * loss_function(output1, label) + alpha * loss_function2(F.log_softmax(output2), label2)
                    loss2 = alpha * loss_function2(F.log_softmax(output2), label2)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    train_loss.append(loss.item())
                    train_loss2.append(loss2.item())
                sum_loss2.append(sum(train_loss2) / len(cleaned_train['tweet']))
                print(sum_loss2[epoch])
            sum_loss.append(sum(train_loss) / len(cleaned_train['tweet']))
            print(sum_loss[epoch])

            if model_name == 'teacher':
                # train evaluation
                model.eval()
                train_preds = []
                with torch.no_grad():
                    for input_ids, attention_mask, token_type_ids, label in train_loader_distill:
                        output1 = model(input_ids, attention_mask, token_type_ids)
                        train_preds.append(output1)
                    preds = torch.cat(train_preds, 0)
                    train_preds_distill.append(preds)

            # evaluation on val set
            model.eval()
            val_preds = []
            with torch.no_grad():
                for input_ids, attention_mask, token_type_ids, label in eval_loader:
                    pred1 = model(input_ids, attention_mask, token_type_ids)
                    val_preds.append(pred1)
                pred1 = torch.cat(val_preds, 0)
                f1_average = model_eval.compute_f1(pred1, y_eval, True)
                val_f1_average.append(f1_average)

            # evaluation on test set
            y_test_list = dh.sep_test_set(y_test, False)
            y_test_gen_list = dh.sep_test_set(y_test_gen, True)

            with torch.no_grad():
                test_preds = []
                for input_ids, attention_mask, token_type_ids, label in test_loader:
                    pred1 = model(input_ids, attention_mask, token_type_ids)
                    test_preds.append(pred1)
                test_predictions = torch.cat(test_preds, 0)
                pred1_list = dh.sep_test_set(test_predictions, False)

                test_preds.clear()
                for input_ids, attention_mask, token_type_ids, label in test_gen_loader:
                    pred1 = model(input_ids, attention_mask, token_type_ids)
                    test_preds.append(pred1)
                test_gen_predictions = torch.cat(test_preds, 0)
                pred1_list_g = dh.sep_test_set(test_gen_predictions, True)

                test_preds.clear()
                for ind in range(len(y_test_list)):
                    pred1 = pred1_list[ind]
                    test_preds.append(pred1)
                    f1_average = model_eval.compute_f1(pred1, y_test_list[ind], True)
                    test_f1_average['single'][ind].append(f1_average)

                test_preds.clear()
                for ind in range(len(y_test_gen_list)):
                    pred1 = pred1_list_g[ind]
                    test_preds.append(pred1)
                    f1_average = model_eval.compute_f1(pred1, y_test_gen_list[ind], True)
                    test_f1_average['general'][ind].append(f1_average)

                accuracy, precision, recall = model_eval.compute_f1(test_predictions, y_test, False)
                metrics['single'].append((accuracy, precision, recall))
                accuracy, precision, recall = model_eval.compute_f1(test_gen_predictions, y_test_gen, False)
                metrics['general'].append((accuracy, precision, recall))

        # model that performs best on the dev set is evaluated on the test set
        best_epoch = [index for index, v in enumerate(val_f1_average) if v == max(val_f1_average)][-1]
        best_result['single'].append([f1[best_epoch] for f1 in test_f1_average['single']])
        best_result['general'].append([f1[best_epoch] for f1 in test_f1_average['general']])

        if model_name == 'teacher':
            best_preds = train_preds_distill[best_epoch]
            torch.save(best_preds, f'../teacher_models/seed{seed}.pt')
        else:
            torch.save(model.state_dict(), f'../student_models/DIMTSD_seed{seed}.pt')
            torch.save(model, f'../student_models/seed{seed}.pt')

        print("*" * 20)
        print(f"dev results with seed {seed} on all epochs")
        print(val_f1_average)
        best_val.append(val_f1_average[best_epoch])

        print("*" * 20)
        print(f"test results with seed {seed} on all epochs")
        print(max(best_result['single']))
        print(best_result['single'])
        print(f"Accuracy: {metrics['single'][best_epoch][0]}\nPrecision: {metrics['single'][best_epoch][1]}\nRecall: {metrics['single'][best_epoch][2]}")

        print("\ngeneralization")
        print(max(best_result['general']))
        print(best_result['general'])
        print(f"Accuracy: {metrics['general'][best_epoch][0]}\nPrecision: {metrics['general'][best_epoch][1]}\nRecall: {metrics['general'][best_epoch][2]}")


if __name__ == "__main__":
    main()
