import sys
sys.path.append('..')
import numpy as np
from common.trainer import Trainer
from common.optimizer import Adam
from common.layers import MatMul, SoftmaxWithLoss
from common.util import preprocess, convert_one_hot

def create_context_target(corpus,window_size=1):
    target = corpus[window_size:-window_size]
    contexts = []

    for idx in range(window_size,len(corpus)-window_size):
        cs = []
        for t in range(-window_size,window_size+1):
            if t == 0:
                continue
            cs.append(corpus[idx+t])
        contexts.append(cs)

    return np.array(contexts),np.array(target)

class SimpleCBOW:
    def __init__(self,vocab_size,hidden_size):
        V, H = vocab_size, hidden_size

        #重みの初期化
        W_in = 0.01 * np.random.randn(V,H).astype('f')
        W_out = 0.01 * np.random.randn(H,V).astype('f')

        #レイヤの作成
        self.in_layer0 = MatMul(W_in)
        self.in_layer1 = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()

        #全ての重みと勾配をリストにまとめる
        layers = [self.in_layer0, self.in_layer1,self.out_layer]
        self.params,self.grads = [],[]
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        #メンバ変数に単語の分散表現を設定
        self.word_vecs = W_in

    def forward(self,contexts,target):
        h0 = self.in_layer0.forward(contexts[:,0])
        h1 = self.in_layer1.forward(contexts[:,1])
        h = (h0+h1) * 0.5
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score,target)
        return loss

    def backward(self,dout=1):
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5
        self.in_layer1.backward(da)
        self.in_layer0.backward(da)

window_size = 1
hidden_size = 5
batch_size = 3
max_eopch = 1000

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
#print(corpus)    #[0 1 2 3 4 1 5 6]
#print(id_to_word)    #{0: 'you', 1: 'say', 2: 'goodbye', 3: 'and', 4: 'i', 5: 'hello', 6: '.'}

vocab_size = len(word_to_id)
contexts, target = create_context_target(corpus,window_size)

#print(contexts) #[[0 2],[1 3],[2 4],[3 1],[4 5],[1 6]]
#print(target) #[1 2 3 4 1 5]

target = convert_one_hot(target,vocab_size)
contexts = convert_one_hot(contexts,vocab_size)

model = SimpleCBOW(vocab_size,hidden_size)
optimizer = Adam()
trainer = Trainer(model,optimizer)

trainer.fit(contexts, target, max_eopch,batch_size)
trainer.plot()

word_vecs = model.word_vecs
for word_id,word in id_to_word.items():
    print(word,word_vecs[word_id])