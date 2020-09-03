import re
import json
from collections import Counter
import pickle
import csv

# train_text = [key for key in open('./data/train.txt', 'r', encoding='utf-8')]
# test_text = [key for key in open('./data/test.txt', 'r', encoding='utf-8')]

train_reader = csv.reader(open('./data/comment_trainset_2class.csv', 'r', encoding='utf-8'))
train_text = [(k[0], k[2]) for k in train_reader]

test_reader = csv.reader(open('./data/comment_testset_2class.csv', 'r', encoding='utf-8'))
test_text = [(k[0], k[2]) for k in test_reader]


all_data = train_text[1:] + test_text[1:]


data = []
for line in all_data:
    text = line[0]
    label = line[1]
    if line[1] == '-1':
        label = [1, 0]
    else:
        label = [0, 1]

    """
    带有语料清洗, 包含数据预处理, 可以根据自己的需求重载
    """
    text = re.sub("\{%.+?%\}", "", text)           # 去除 {%xxx%} (地理定位, 微博话题等)
    text = re.sub("@.+?( |$)", "", text)           # 去除 @xxx (用户名)
    text = re.sub("【.+?】", "", text)              # 去除 【xx】 (里面的内容通常都不是用户自己写的)
    text = re.sub("\[.+?\]", "", text)               # 将文本中的图标
    text = re.sub('\u200b', "", text)  # 将文本中的图标
    text = text.replace(' ', '')
    text = text.strip('\n')
    text = text.strip(' ')
    if len(text) < 5:
        pass
    else:
        data.append((text, label))

#构建字典
word = {}
# word['<PAD>'] = 0
# word['<UNK>'] = 1
j = 0
word_list = []
word = {}
for key in data:
    word_list.extend([k for k in key[0] if k != ' '])

counter = Counter(word_list)
count_pari = counter.most_common(4998)
word_, _ = list(zip(*count_pari))
word_ = list([key for key in word_])
word_new = ['<PAD>', '<UNK>']
word_new = word_new + word_
for key in word_new:
    if key not in word:
        word[key] = j
        j += 1
    else:
        pass
print(word)
with open('./data/word2id.pkl', 'wb') as fw: #将建立的字典 保存
    pickle.dump(word, fw)


writ_f = open('./data/text_label.json', 'wb')
eachline_ = json.dumps(data, ensure_ascii=False, indent=4)
eachline_ = eachline_.encode()
writ_f.write(eachline_)