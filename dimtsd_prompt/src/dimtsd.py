import torch
import json
import preprocessor as p
import re
import wordninja
import modeling
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')


with open("../noslang_data.json", "r") as f:
    data1 = json.load(f)
data2 = {}
with open("../emnlp_dict.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        row = line.split('\t')
        data2[row[0]] = row[1].rstrip()
normalization_dict = {**data1, **data2}

# Load the model
model = modeling.StanceClassifier(3).to('cuda')
model.load_state_dict(torch.load('../student_models/DIMTSD_seed4.pt'))
model.eval()

mapping = {0: 'against', 1: 'none', 2: 'favor'}

tweet = input('Please Enter tweet:')
target = input('Please Enter target:')
domain = input('Please Enter domain:')

flag = True
while flag:
    # Preprocessing
    p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.RESERVED)
    clean_data = p.clean(tweet)  # using lib to clean URL,hashtags...
    clean_data = re.sub(r"#SemST", "", clean_data)
    clean_data = re.findall(r"[A-Za-z#@]+|[,.!?&/\<>=$]|[0-9]+", clean_data)
    clean_data = [[x.lower()] for x in clean_data]

    for i in range(len(clean_data)):
        if clean_data[i][0] in normalization_dict.keys():
            clean_data[i] = normalization_dict[clean_data[i][0]].split()
            continue
        if clean_data[i][0].startswith("#") or clean_data[i][0].startswith("@"):
            clean_data[i] = wordninja.split(clean_data[i][0])  # separate hashtags
    clean_data = [j for i in clean_data for j in i]
    tweet = ' '.join(tweet)

    inputs = f"target: {target} tweet: {tweet} domain: {domain}"
    encoded = model.tokenizer(inputs, add_special_tokens=True, return_attention_mask=True, max_length=128,
                              padding='max_length', truncation=True, return_tensors='pt').to('cuda')

    with torch.no_grad():
        prediction = model(encoded.input_ids, encoded.attention_mask, encoded.token_type_ids)
        rounded_preds = torch.nn.functional.softmax(prediction)
        _, index = torch.max(rounded_preds, 1)
        print(f'Predicted Stance: {mapping[index.item()]}')

    cont = input("Do you want to continue? (Please Enter 'yes' or 'no')")
    if cont == 'no':
        flag = False
