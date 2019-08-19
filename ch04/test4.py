import numpy as np
from ch04.negative_sampling_layer import UnigramSampler,SigmoidWithLoss

class Embedding:
    def __init__(self,W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self,idx):
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out

    def backward(self,dout):
        dW, = self.grads
        dW[...] = 0
        for i, word_id in enumerate(self.idx):
            dW[word_id] = dout[i]
        return None

class EmbeddingDot:
    def __init__(self,W):
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None

    def forward(self, h, idx):
        target_W = self.embed.forward(idx)
        out = np.sum(target_W * h, axis=1)

        self.cache = (h,target_W)
        return out

    def backward(self,dout):
        h,target_W = self.cache
        dout = dout.reshape(dout.shape[0],1)

        dtarget_W = dout * h
        self.embed.backward(dtarget_W)
        dh = dout * target_W
        return dh

class NegativeSamplingLoss:
    def __init__(self,W,corpus,power=0.75,sample_size=5):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus,power,sample_size)
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)]
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size+1)]
        self.params,self.grads = [],[]
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self,h,target):
        batch_size = target.shape[0]
        negative_sample = self.sampler.get_negative_sample(target)

        #正例のフォーワード
        score = self.embed_dot_layers[0].forward(h,target)
        correct_label = np.ones(batch_size, dtype=np.int32)
        loss = self.loss_layers[0].forward(score,correct_label)

        #負例のフォーワード
        negative_label = np.zeros(batch_size,dtype=np.int32)
        for i in range(self.sample_size):
            negative_target = negative_sample[:,i]
            score = self.embed_dot_layers[1+i].forward(h,negative_target)
            loss += self.loss_layers[1+i].forward(score,negative_label)

        return loss

    def backward(self,dout=1):
        dh = 0
        for l0,l1 in zip(self.loss_layers,self.embed_dot_layers):
            dscore = l0.backward(dout)
            dh += l1.backward(dscore)
        return dh

class CBOW:
    def __init__(self,vocab_size,hidden_size,window_size,corpus):
        V,H = vocab_size,hidden_size

        #重みの初期化
        W_in = 0.01 * np.random.randn(V,H).astype('f')
        W_out = 0.01 * np.random.randn(V,H).astype('f')

        #レイヤの生成
        self.in_layers = []
        for i in range(2*window_size):
            layer = Embedding(W_in)
            self.in_layers.append(layer)
        self.ns_loss = NegativeSamplingLoss(W_out,corpus,power=0.75,sample_size=5)

        #全ての重みと勾配を配列にまとめる
        layers = self.in_layers + [self.ns_loss]
        self.params ,self.grads = [],[]
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        #メンバ変数に単語の分散表現を設定
        self.word_vecs = W_in

    def forward(self,contexts,target):
        h = 0
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(contexts[:,i])
        h += 1 / len(self.in_layers)
        loss = self.ns_loss.forward(h,target)
        return loss

    def backward(self,dout=1):
        dout = self.ns_loss.backward(dout)
        dout *= 1 / len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dout)
        return None


# corpus = np.array([0,1,2,3,4,1,2,3])
# power = 0.75
# sample_size = 2
#
# sampler = UnigramSampler(corpus,power,sample_size)
# target = np.array([1,3,0])
# negative_sample = sampler.get_negative_sample(target)
# print(negative_sample)
import pickle


with open('cbow_params.pkl', 'rb') as f:
    data = pickle.load(f)
    print(data)