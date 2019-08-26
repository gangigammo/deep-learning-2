import numpy as np
import matplotlib.pyplot as plt
from common.time_layers import *
from common.base_model import BaseModel
# coding: utf-8
from common.optimizer import *
import sys
sys.path.append('..')
from common.time_layers import *
import pickle
from common.optimizer import SGD
from common.trainer import RnnlmTrainer
from common.util import eval_perplexity, to_gpu
from dataset import ptb
from ch06.better_rnnlm import BetterRnnlm


##############rnn_gradient_graph_implementation##################
# N = 2 #ミニバッチサイズ
# H = 3 #隠れ状態ベクトルの次元数
# T = 20 #時系列データの長さ
#
# dh = np.ones((N,H))
# np.random.seed(3) #再現性のため乱数のシードを固定
# Wh = np.random.randn(H,H) * 0.5
#
# norm_list = []
# for t in range(T):
#     dh = np.dot(dh,Wh.T)
#     norm = np.sqrt(np.sum(dh**2)) / N
#     norm_list.append(norm)
#
# print(norm_list)
#
# # グラフの描画
# plt.plot(np.arange(len(norm_list)), norm_list)
# plt.xticks([0, 4, 9, 14, 19], [1, 5, 10, 15, 20])
# plt.xlabel('time step')
# plt.ylabel('norm')
# plt.show()

dW1 = np.random.rand(3,3) * 10
dW2 = np.random.rand(3,3) * 10
grads = [dW1,dW2]
max_norm = 5.0
def clip_grads(grads,max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad**2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate

clip_grads(grads,max_norm)

# class LSTM:
#     def __init__(self,Wx,Wh,b):
#         self.params = [Wx,Wh,b] #４つのパラメータが格納されている
#         self.grads = [np.zeros_like(Wx),np.zeros_like(Wh),np.zeros_like(b)]
#         self.cache = None
#
#     def forward(self,x,h_prev,c_prev):
#         Wx, Wh, b = self.params
#         N, H = h_prev.shape
#
#         A = np.dot(x,Wx) + np.dot(h_prev,Wh) + b #4つのaffineレイヤが格納されている
#
#         #slice
#         f = A[:, :H]
#         g = A[:, H:2*H]
#         i = A[:,2*H:3*H]
#         o = A[:,3*H:]
#
#         f = sigmoid(f)
#         g = sigmoid(g)
#         i = sigmoid(i)
#         o = sigmoid(o)
#
#         c_next = f * c_prev + g * i
#         h_next = o * np.tanh(c_next)
#
#         self.cache = (x,h_prev,c_prev,i,f,g,o,c_next)
#         return h_next,c_next
#
#     #def backward(self){}

# class TimeLSTM:
#     def __init__(self,Wx,Wh,b,stateful=False):
#         self.params = [Wx, Wh, b]
#         self.grads = [np.zeros_like(Wx),np.zeros_like(Wh),np.zeros_like(b)]
#         self.layers = None
#
#         self.h,self.c = None,None
#         self.dh = None
#         self.stateful = stateful
#
#     def forward(self,xs):
#         Wx,Wh,b = self.params
#         N,T,D = xs.shape
#         H = Wh.shape[0]
#
#         self.layers = []
#         hs = np.empty((N,T,H),dtype='f')
#
#         if not self.stateful or self.h is None:
#             self.h = np.zeros((N,H),dtype='f')
#         if not self.stateful or self.c is None:
#             self.c = np.zeros((N,H), dtype='f')
#
#         for t in range(T):
#             layer = LSTM(*self.params)
#             self.h,self.c = layer.forward(xs[:,t,:], self.h,self.c)
#             hs[:, t, :] = self.h
#
#             self.layers.append(layer)
#         return hs
#
#     def backward(self,dhs):
#         Wx,Wh,b = self.params
#         N,T,H = dhs.shape
#         D = Wx.shape[0]
#
#         dxs = np.empty((N,T,D),dtype='f')
#         dh,dc = 0,0
#
#         grads = [0,0,0]
#         for t in reversed(range(T)):
#             layer = self.layers[t]
#             dx,dh,dc = layer.backward(dhs[:,t,:] + dh,dc)
#             dxs[:,t,:] = dx
#             for i,grad in enumerate(layer.grads):
#                 grads[i] += grad
#
#         for i,grad in enumerate(grads):
#             self.grads[i][...] = grad
#
#         self.dh = dh
#         return dxs
#
#     def set_state(self,h,c=None):
#         self.h,self.c = h,c
#
#     def reset_state(self):
#         self.h, self.c = None,None


class Rnnlm:
    def __init__(self,vocab_size=10000,wordvec_size=100,hidden_size=100):
        V,D,H = vocab_size,wordvec_size,hidden_size
        rn = np.random.randn

        #重みの初期化
        embed_W = (rn(V,D) / 100).astype('f')
        lstm_Wx = (rn(D,4*H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H,4*H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4*H).astype('f')
        affine_W = (rn(H,V) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        #レイヤの生成
        self.layers = [
            TimeEmbedding(embed_W),
            TimeLSTM(lstm_Wx,lstm_Wh,lstm_b,stateful=True),
            TimeAffine(affine_W, affine_b)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layer = self.layers[1]

        #全ての重みと勾配をリストにまとめる
        self.params,self.grads = [],[]
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self,xs):
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    def forward(self,xs,ts):
        score = self.predict(xs)
        loss = self.loss_layer.forward(score,ts)
        return loss

    def backward(self,dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        self.lstm_layer.reset_state()

    def save_params(self,file_name='Rnnlm.pkl'):
        with open(file_name,'wb') as f:
            pickle.dump(self.params,f)

    def load_params(self,file_name='Rnnlm.pkl'):
        with open(file_name, 'rb') as f:
            self.params = pickle.load(f)

# #ハイパーパラメータの設定
# batch_size = 20
# wordvec_size = 100
# hidden_size = 100 #RNNの隠れ状態ベクトルの要素数
# time_size = 35 #RNNを展開するサイズ
# lr = 20.0
# max_epoch = 4
# max_grad = 0.25
#
# #学習データの読み込み
# corpus,word_to_id,id_to_word = ptb.load_data('train')
# corpus_test, _, _ = ptb.load_data('test')
# vocab_size = len(word_to_id)
# xs = corpus[:-1]
# ts = corpus[1:]
#
# #モデルの生成
# model = Rnnlm(vocab_size,wordvec_size)
# optimizer = SGD(lr)
# trainer = RnnlmTrainer(model,optimizer)
#
# #勾配クリッピングを適応して学習
# trainer.fit(xs,ts,max_epoch,batch_size,time_size,max_grad,eval_interval=20)
# trainer.plot(ylim=(0,500))
#
# #テストデータで評価
# model.reset_state()
# ppl_test = eval_perplexity(model,corpus_test)
# print('test preplexity: ',ppl_test)
#
# #パラメータの保存
# model.save_params()