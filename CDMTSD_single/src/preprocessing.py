import preprocessor as p
import re
import wordninja
import pandas as pd


def load_data(filename: str):
    '''
    read data and convert labels to integers
    :param filename: path of csv file
    :return: pandas DataFrame
    '''
    dataset = pd.read_csv(filename, encoding='ISO-8859-1')
    dataset['Stance 1'].replace(to_replace=[r'FAVOR|support', r'NONE|comment', r'AGAINST|refute'], value=[2, 1, 0],
                                regex=True, inplace=True)
    dataset = dataset[dataset['Stance 1'] != 'unrelated']  # remove 'unrelated' label of WT-WT

    return dataset


def data_clean(strings: str, norm_dict: dict):
    '''
    clean data
    :param strings: texts to be cleaned
    :param norm_dict: normalization dictionary
    :return: list of cleaned texts
    '''
    p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.RESERVED)
    clean_data = p.clean(strings)  # using lib to clean URL,hashtags...
    clean_data = re.sub(r"#SemST", "", clean_data)
    clean_data = re.findall(r"[A-Za-z#@]+|[,.!?&/\<>=$]|[0-9]+", clean_data)
    clean_data = [[x.lower()] for x in clean_data]

    for i in range(len(clean_data)):
        if clean_data[i][0] in norm_dict.keys():
            clean_data[i] = norm_dict[clean_data[i][0]].split()
            continue
        if clean_data[i][0].startswith("#") or clean_data[i][0].startswith("@"):
            clean_data[i] = wordninja.split(clean_data[i][0])  # separate hashtags
    clean_data = [j for i in clean_data for j in i]

    return clean_data


def clean_all(filename: str, norm_dict: dict):
    '''
    load and clean csv files
    :param filename: path of csv file
    :param norm_dict: normalization dictionary
    :return: a dictionary of cleaned data
    '''
    dataset = load_data(filename)
    # convert DataFrame to list of strings ['string','string', ...]
    text = dataset['Tweet'].values.tolist()
    label = dataset['Stance 1'].values.tolist()
    target = dataset['Target 1'].values.tolist()
    domain = dataset['domain'].values.tolist()

    for i in range(len(label)):
        # clean each text to list of list of words [ ['word1','word2'], [...], ... ]
        text[i] = data_clean(text[i], norm_dict)
        target[i] = data_clean(target[i], norm_dict)
        text[i] = ' '.join(text[i])
        target[i] = ' '.join(target[i])

    return {'tweet': text, 'target': target, 'domain': domain, 'stance': label}
