import sys
sys.path.append('..')
import numpy as np
from common.functions import softmax
from ch06.rnnlm import Rnnlm
from ch06.better_rnnlm import BetterRnnlm
from dataset import ptb
from dataset import sequence
from common.time_layers import *
from common.base_model import BaseModel
import matplotlib.pyplot as plt
from dataset import sequence
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq
from ch07.seq2seq import Seq2seq
from ch07.peeky_seq2seq import PeekySeq2seq
from ch07.peeky_seq2seq import PeekyDecoder

class RnnlmGen(Rnnlm):
    def generate(self,start_id,skip_ids=None,sample_size=100):
        word_ids = [start_id]

        x = start_id
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1,1)
            score = self.predict(x)
            p = softmax(score.flatten())

            sampled = np.random.choice(len(p), size=1, p=p)
            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                word_ids.append(int(x))

        return word_ids


# corpus,word_to_id,id_to_word = ptb.load_data('train')
# vocab_size = len(word_to_id)
# corpus_size = len(corpus)
#
# model = RnnlmGen()
# model.load_params('../ch06/Rnnlm.pkl')
#
# #start文字とskip文字の設定
# start_word = 'you'
# start_id = word_to_id[start_word]
# skip_words = ['N', '<unk>', '$']
# skip_ids = [word_to_id[w] for w in skip_words]
#
# #文章作成
# word_ids = model.generate(start_id, skip_ids)
# txt = ' '.join([id_to_word[i] for i in word_ids])
# txt = txt.replace(' <eos>', '.\n')
# print(txt)



# (x_train,t_train),(x_test,t_test) = sequence.load_data('addition.txt',seed=1984)
# char_to_id, id_to_char = sequence.get_vocab()
#
# print(x_train.shape,t_train.shape) #(45000,7),(45000,7)
# print(x_test.shape,t_test.shape) #(5000,7),(5000,7)
#
# print(x_train[0]) #[3,0,2,0,0,11,5]
# print(t_train[0]) #[6,0,11,7,5]
#
# print(''.join([id_to_char[c] for c in x_train[0]]))
# print(''.join([id_to_char[c] for c in t_train[0]]))

class Encoder:
    def __init__(self,vocab_size,wordvec_size,hidden_size):
        V,D,H = vocab_size,wordvec_size,hidden_size
        rn = np.random.randn

        embed_W = (rn(V,D) / 100).astype('f')
        lstm_Wx = (rn(D,4*H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H,4*H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4*H).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx,lstm_Wh,lstm_b,stateful=False)

        self.params = self.embed.params + self.lstm.params
        self.grads = self.embed.grads + self.lstm.grads
        self.hs = None

    def forward(self,xs):
        xs = self.forward(xs)
        hs = self.lstm.forward(xs)
        self.hs = hs
        return hs[:,-1,:]

    def backward(self,dh):
        dhs = np.zeros_like(self.hs)
        dhs[:,-1,:] = dh

        dout = self.lstm.backward(dhs)
        dout = self.embed.backward(dout)
        return dout

class Decoder:
    def __init__(self,vocab_size,word_vec_size,hidden_size):
        V,D,H = vocab_size,word_vec_size,hidden_size
        rn = np.random.randn

        embed_W = (rn(V,D) / 100).astype('f')
        lstm_Wx = (rn(D,4*H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H,4*H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4*H).astype('f')
        affine_W = (rn(H,V) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx,lstm_Wh,lstm_b,stateful=True)
        self.affine = TimeAffine(affine_W,affine_b)

        self.params,self.grads = [],[]
        for layer in (self.embed,self.lstm,self.affine):
            self.params += layer.params
            self.grads += layer.grads

    def forward(self,xs,h):
        self.lstm.set_state(h)

        out = self.embed.forward(xs)
        out = self.lstm.forward(out)
        score = self.affine.forward(out)
        return score

    def backward(self,dscore):
        dout = self.affine.backward(dscore)
        dout = self.lstm.backward(dout)
        dout = self.embed.backward(dout)
        dh = self.lstm.dh #lstmレイヤのbackwardを出力としている
        return dh

    def generate(self,h,start_id,sample_size):
        sampled = []
        sample_id = start_id
        self.lstm.set_state(h)

        for _ in range(sample_size):
            x = np.array(sample_id).reshape((1,1))
            out = self.embed.forward(x)
            out = self.lstm.forward(out)
            score = self.affine.forward(out)

            sample_id = np.argmax(score.flatten())
            sampled.append(int(sample_id))

        return sampled

# class Seq2seq(BaseModel):
#     def __init__(self,vocab_size,wordvec_size,hidden_size):
#         V,D,H = vocab_size,wordvec_size,hidden_size
#         self.encoder = Encoder(V,D,H)
#         self.decoder = Decoder(V,D,H)
#         self.softmax = TimeSoftmaxWithLoss()
#
#         self.params = self.encoder.params + self.decoder.params
#         self.grads = self.encoder.grads + self.decoder.params
#
#     def forward(self,xs,ts):
#         decoder_xs,decoder_ts = ts[:,:-1],ts[:,1:]
#
#         h = self.encoder.forward(xs)
#         score = self.decoder.forward(decoder_xs,h)
#         loss = self.softmax.forward(score,decoder_ts)
#         return loss
#
#     def backward(self,dout=1):
#         dout = self.softmax.backward(dout)
#         dh = self.decoder.backward(dout)
#         dout = self.encoder.backward(dh)
#         return dout
#
#     def generate(self,xs,start_id,sample_size):
#         h = self.encoder.forward(xs)
#         sampled = self.decoder.generate(h,start_id,sample_size)
#         return sampled

# class PeekyDecoder:
#     def __init__(self,vocab_size,wordvec_size,hidden_size):
#         V,D,H = vocab_size,wordvec_size,hidden_size
#         rn = np.random.randn
#
#         embed_W = (rn(V,D) / 100).astype('f')
#         lstm_Wx = (rn(H+D,4*H) / np.sqrt(H+D)).astype('f')
#         lstm_Wh = (rn(H,4*H) / np.sqrt(H)).astype('f')
#         lstm_b = np.zeros(4*H).astype('f')
#         affine_W = (rn(H+H,V) / np.sqrt(H+H)).astype('f')
#         affine_b = np.zeros(V).astype('f')
#
#         self.embed = TimeEmbedding(embed_W)
#         self.lstm = TimeLSTM(lstm_Wx,lstm_Wh,lstm_b,stateful=True)
#         self.affine = TimeAffine(affine_W,affine_b)
#
#         self.params,self.grads = [],[]
#         for layer in (self.embed,self.lstm,self.affine):
#             self.params += layer.params
#             self.grads += layer.grads
#         self.cache = None
#
#     def forward(self,xs,h):
#         N,T = xs.shape
#         N,H = h.shape
#
#         self.lstm.set_state(h)
#
#         out = self.embed.forward(xs)
#         hs = np.repeat(h,T,axis=0).reshape(N,T,H)
#         out = np.concatenate((hs,out),axis=2)
#
#         score = self.affine.forward(out)
#         self.cache = H
#         return score

class PeekySeq2seq(Seq2seq):
    def __init__(self,vocab_size,wordvec_size,hidden_size):
        V,D,H = vocab_size,wordvec_size,hidden_size
        self.encoder = Encoder(V,D,H)
        self.decoder = PeekyDecoder(V,D,H)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads


#データセットの読み込み
(x_train,t_train),(x_test,t_test) = sequence.load_data('addition.txt')
char_to_id,id_to_char = sequence.get_vocab()

#ハイパーパラメータの設定
vocab_size = len(char_to_id)
wordvec_size = 16
hidden_size = 128
batch_size = 128
max_epoch = 25
max_grad = 5.0

#　モデル / オプティマイザ / トレーナーの生成
model = Seq2seq(vocab_size,wordvec_size,hidden_size)
optimizer = Adam()
trainer = Trainer(model,optimizer)

acc_list = []
for epoch in range(max_epoch):
    trainer.fit(x_train,t_train,max_epoch=1,batch_size =batch_size,max_grad=max_grad)

    correct_num = 0
    for i in range(len(x_test)):
        question, correct = x_test[[i]],t_test[[i]]
        verbose = i < 100
        correct_num += eval_seq2seq(model,question,correct,id_to_char,verbose)
        acc = float(correct_num) / len(x_test)
        acc_list.append(acc)
        print('val acc %.3f%%' % (acc * 100))

