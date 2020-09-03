import tensorflow as tf
import json
import random
import pickle
import numpy as np

'''

'''
max_len = 50
batch_size = 64
keep_prob = 0.5
lr = 0.001
num_epochs = 20
label_class = 2


data_all = json.load(open('./data/text_label.json', encoding='utf-8'))
word_dict = pickle.load(open('./data/word2id.pkl', 'rb'))
# print(len(data_all))
random.shuffle(data_all)
train_ = data_all[0:35000]
test_ = data_all[35000:]

def Tokenizer(text):
    word_list = []
    for key in text:
        if key not in word_dict:
            key = '<UNK>'
        word_list.append(word_dict[key])
        word_list = word_list[:max_len]
    return word_list

class data_loader(object):
    def __init__(self, data_train):
        self.data = data_train
        self.in_sentence = [Tokenizer(sentence[0]) for sentence in self.data]
        self.in_label = [sentence[1] for sentence in self.data]

        self.in_sentence = tf.keras.preprocessing.sequence.pad_sequences(self.in_sentence, max_len, padding='post', truncating='post')

        self.in_label = np.array(self.in_label, np.float32)

        self.train_num = self.in_sentence.shape[0]
        self.db_train = tf.data.Dataset.from_tensor_slices(
            (self.in_sentence, self.in_label))
        self.db_train = self.db_train.shuffle(self.train_num).batch(batch_size, drop_remainder=True)

    def get_data(self, batch_s):
        indics = np.random.randint(0, self.train_num, batch_s)
        return self.in_sentence[indics], self.in_label[indics]

class Sentimental_Double_Rnn(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.char_embedding = tf.keras.layers.Embedding(5000, 64, mask_zero=True)
        self.Rnncell = tf.keras.layers.SimpleRNNCell(64)
        Rnncell_drop = tf.nn.RNNCellDropoutWrapper(self.Rnncell, keep_prob)
        self.double_rnn = tf.keras.layers.RNN([Rnncell_drop, Rnncell_drop], return_sequences=True)
        self.drop_out = tf.keras.layers.Dropout(keep_prob)
        self.dense = tf.keras.layers.Dense(label_class)

    def call(self, inputs):
        x = self.char_embedding(inputs)
        mask = self.char_embedding.compute_mask(inputs)
        x = self.double_rnn(x, mask=mask)
        x = tf.reduce_mean(x, 1)
        x = self.drop_out(x)
        x = self.dense(x)
        x = tf.nn.sigmoid(x)
        return x

def loss_function(in_socre, pre_score):
    loss = tf.keras.losses.binary_crossentropy(y_true=in_socre, y_pred=pre_score)
    loss = tf.reduce_mean(loss)
    return loss


'''
验证模型效果
c:相似预测分数 [0,1]
pre_c: 取整，与测试集进行精度判断
'''
class Extra_result(object):
    def __init__(self, in_list):
        self.in_list = in_list
    def call(self):
        token = Tokenizer(self.in_list)
        token = np.array([token], np.int32)
        out_score = double_rnn_model(token)
        c = float(np.array(tf.argmax(out_score[0])))  #
        pre_c = 0
        if c > 0.50:
            pre_c = 1
        # print(c, pre_c)
        return c, pre_c


class Evaluate(object):
    def __init__(self):
        pass
    def evaluate(self, data):
        A, T = 1e-10, 1e-10
        for d in data:
            #print(d[0])
            extra_items = Extra_result(d[0])
            score, c = extra_items.call()
            if d[1] == [1, 0]:
                true = 0
            else:
                true = 1
            T += 1
            if c == true:
                A += 1
        P = float(A / T)
        return P


#建立模型
double_rnn_model = Sentimental_Double_Rnn()
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

#保存模型
checkpoint = tf.train.Checkpoint(optimizer=optimizer, double_rnn_model=double_rnn_model)

evaluate = Evaluate()
data_loader = data_loader(train_)
best = 0.0

for epoch in range(num_epochs):
    print('Epoch:', epoch + 1)

    num_batchs = int(data_loader.train_num / batch_size) + 1
    for batch_index in range(num_batchs):
        input_x,  input_c = data_loader.get_data(batch_size)
        with tf.GradientTape() as tape:
            score = double_rnn_model(input_x)
            loss = loss_function(input_c, score)
            if (batch_index+1) % 100 == 0:
                print("batch %d: loss %f" % (batch_index+1, loss.numpy()))

        variables = double_rnn_model.variables
        grads = tape.gradient(loss, variables)
        grads, _ = tf.clip_by_global_norm(grads, 5.0)  # 梯度裁剪 <=5
        optimizer.apply_gradients(grads_and_vars=zip(grads, variables))

    # p = evaluate.evaluate(train_)
    # print('训练集:', "p %f" % p)

    P = evaluate.evaluate(test_)
    print('测试集:', "P %f" % P)
    if round(P, 4) > best and round(P, 2) > 0.50:
        best = P
        print('saving_model')
        #model.save('./save/Entity_Relationshaip_version2.h5')
        checkpoint.save('./save/double_rnn_model/version1_checkpoints.ckpt')