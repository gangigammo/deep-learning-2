import numpy as np
import sys

import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from dataset import sequence
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq
from ch08.attention_seq2seq import AttentionSeq2seq
from ch07.seq2seq import Seq2seq
from ch07.peeky_seq2seq import PeekySeq2seq

# class WeightSum:
#     def __init__(self):
#         self.params, self.grads = [], []
#         self.cache = None
#
#     def forward(self, hs, a):
#         N, T, H = hs.shape
#
#         ar = a.reshape(N, T, 1).repeat(H, axis=2)
#         t = hs * ar
#         c = np.sum(t, axis=1)
#
#         self.cache = (hs, ar)
#         return c
#
#     def backward(self, dc):
#         hs, ar = self.cache
#         N, T, H = hs.shape
#
#         dt = dc.reshape(N, 1, H).repeat(T, axis=1)  # sumの逆伝播
#         dar = dt * hs
#         dhs = dt * ar
#         da = np.sum(dar, axis=2)
#
#         return dhs, da
#
# class AttentionWeight:
#     def __init__(self):
#         self.params,self.grads = [],[]
#         self.softmax = Softmax()
#         self.cache = None
#
#     def forward(self,hs,h):
#         N,T,H = hs.shape
#
#         hr = h.reshape(N,1,H).repeat(T,axis=1)
#         t = hs * hr
#         s = np.sum(t,axis=1)
#         a = self.softmax.forward(s)
#
#         self.cache = (hs,hr)
#         return a
#
#     def backward(self,da):
#         hs,hr = self.cache
#         N,T,H = hs.shape
#
#         ds = self.softmax.backward(da)
#         dt = ds.reshape(N,T,1).repeat(H,axis=2)
#         dhs = dt * hr
#         dhr = dt * hs
#         dh = np.sum(dhr,axis=1)
#
#         return dhs,dh
#
# class Attention:
#     def __init__(self):
#         self.params, self.grads = [],[]
#         self.attention_weight_layer = AttentionWeight()
#         self.weight_sum_layer = WeightSum()
#         self.attention_weight = None
#
#     def forward(self,hs,h):
#         a = self.attention_weight_layer.forward(hs,h)
#         out = self.weight_sum_layer.forward(hs,a)
#         self.attention_weight = a
#         return out
#
#     def backward(self,dout):
#         dhs0,da = self.weight_sum_layer.backward(dout)
#         dhs1,dh = self.attention_weight_layer.backward(da)
#         dhs = dhs0 + dhs1
#         return dhs,dh
#
# class TimeAttention:
#     def __init__(self):
#         self.params,self.grads = [],[]
#         self.layers = None
#         self.attention_weight = None
#
#     def forward(self,hs_enc,hs_dec):
#         N,T,H = hs_dec.shape
#         out = np.empty_like(hs_dec)
#         self.layers = []
#         self.attention_weight = []
#
#         for t in range(T):
#             layer = Attention()
#             out[:,t,:] = layer.forward(hs_enc,hs_dec[:,t,:])
#             self.layers.append(layer)
#             self.attention_weight.append(layer.attention_weight)
#         return out
#
#     def backward(self,dout):
#         N,T,H = dout.shape
#         dhs_enc = 0
#         dhs_dec = np.empty_like(dout)
#
#         for t in range(T):
#             layer = self.layers[t]
#             dhs, dh = layer.backward(dout[:,t,:])
#             dhs_enc += dhs
#             dhs_dec[:,t,:] = dh
#
#         return dhs_enc,dhs_dec
#
# class AttentionEncoder(Encoder):
#     def forward(self,xs):
#         xs = self.embed.forward(xs)
#         hs = self.lstm.forward(xs)
#         return hs
#
#     def backward(self,dhs):
#         dout = self.lstm.backawrd(dhs)
#         dout = self.embed.backward(dout)
#         return dout
#
# class AttentionDecoder:
#     def __init__(self,vocab_size,wordvec_size,hidden_size):
#         V,D,H = vocab_size,wordvec_size,hidden_size
#         rn = np.random.randn
#
#         embed_W = (rn(V,D) / 100).astype('f')
#         lstm_Wx = (rn(D,4*H) / np.sqrt(D)).astype('f')
#         lstm_Wh = (rn(H,4*H) / np.sqrt(H)).astype('f')
#         lstm_b = np.zeros(4*H).astype('f')
#         affine_W = (rn(2*H,V) / np.sqrt(2*H)).astype('f')
#         affine_b = np.zeros(V).astype('f')
#
#         self.embed = TimeEmbedding(embed_W)
#         self.lstm = TimeLSTM(lstm_Wx,lstm_Wh,lstm_b,stateful=True)
#         self.attention = TimeAttention()
#         self.affine = TimeAffine(affine_W,affine_b)
#         layers = [self.embed,self.lstm,self.attention,self.affine]
#
#         self.params,self.grads = [],[]
#         for layer in layers:
#             self.params += layer.params
#             self.grads += layer.grads
#
#     def forward(self,xs,enc_hs):
#         h = enc_hs[:,-1]
#         self.lstm.set_state(h)
#
#         out = self.embed.forward(xs)
#         dec_hs = self.lstm.forward(out)
#         c = self.attention.forward(enc_hs,dec_hs)
#         out = np.concatenate((c,dec_hs),axis=2)
#         score = self.affine.forward(out)
#
#         return score
#
# class AttentionSeq2seq(Seq2seq):
#     def __init__(self,vocab_size,wordvec_size,hidden_size):
#         args = vocab_size,wordvec_size,hidden_size
#         self.encoder = AttentionEncoder(*args)
#         self.decoder = AttentionDecoder(*args)
#         self.softmax = TimeSoftmaxWithLoss()
#
#         self.params = self.encoder.params + self.decoder.params
#         self.grads = self.encoder.grads + self.decoder.grads
#


# T,H = 5,4
# hs = np.random.randn(T,H)
# a = np.array([0.8,0.1,0.03,0.05,0.02])
#
# ar = a.reshape(5,1).repeat(4,axis=1)
# print(ar)
# #print(ar.shape)
# #(5,4)
#
#
# t = hs * ar
# print(t)
# #print(t.shape)
# #(5,4)
#
# c = np.sum(t,axis=0)
# print(c)
# #print(c.shape)



# N,T,H = 10,5,4
# hs = np.random.randn(N,T,H)
# a = np.random.randn(N,H)
# ar = a.reshape(N,1,H).repeat(T,axis=1)
# #ar = a.reshape(N,1,H) #ブロードキャストを使う場合
#
# t = hs * ar
# print(t.shape)
# #(10,5,4)
#
# s = np.sum(t,axis=2)
# print(s.shape)
# # (10,5)
#
# softmax = Softmax()
# a = softmax.forward(s)
# print(a.shape)
# print(a)

#データの読み込み
(x_train,t_train),(x_test,t_test) = sequence.load_data('data.txt')
char_to_id,id_to_char = sequence.get_vocab()

#入力文を反転
x_train, x_test = x_train[:,::-1],x_test[:,::-1]

#ハイパーパラメータの設定
vocab_size = len(char_to_id)
wordvec_size = 16
hidden_size = 256
batch_size = 128
max_epoch = 10
max_grad = 5.0

model = AttentionSeq2seq(vocab_size,wordvec_size,hidden_size)
optimizer = Adam()
trainer = Trainer(model,optimizer)

acc_list = []
for epoch in range(max_epoch):
    trainer.fit(x_train,t_train,max_epoch=1,batch_size=batch_size,max_grad=max_grad)

    correct_num=0
    for i in range(len(x_test)):
        question,correct = x_test[[i]],t_test[[i]]
        verbose = i<10
        correct_num += eval_seq2seq(model,question,correct,id_to_char,verbose,is_reverse=True)

    acc = float(correct_num) / len(x_test)
    acc_list.append(acc)
    print('val acc %.3f%%' % (acc*100))

model.save_params()