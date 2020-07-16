import numpy as np
import math

def sigmoid(x):
    return 1 / (1 + math.e ** (-x))

class NeuralNet:

    def __init__(self, n_in, n_mid, n_out, eta, alpha):
        #各層の出力値
        self.input = np.zeros(n_in)
        self.mid = np.zeros(n_mid)
        self.output = np.zeros(n_out)

        #中間層、出力層の重み
        # 初期値は[-1,1)の乱数
        self.weight_mid = 2*np.random.random_sample((n_in,  n_mid)) - 1
        self.weight_out = 2*np.random.random_sample((n_mid, n_out)) - 1

        #重みの更新量
        self.update_mid = np.zeros((n_in,  n_mid))
        self.update_out = np.zeros((n_mid, n_out))

        #教師信号
        self.teacher = np.identity(n_out)

        #学習定数
        self.eta = eta

        #安定化定数
        self.alpha = alpha

    def updateWeights(self,v):
        vec_m = v*self.output*(1-self.output)
        vec_o =  (vec_m * self.weight_out).dot(np.ones(self.output.size)) * self.mid*(1-self.mid)
        
        self.update_mid = self.eta * np.array(np.matrix(self.input).T * vec_o) + self.alpha * self.update_mid
        self.update_out = self.eta * np.array(np.matrix(self.mid).T * vec_m) + self.alpha * self.update_out
        
        self.weight_mid += self.update_mid
        self.weight_out += self.update_out
        
    def forward(self, data):
        self.input = data
        #入力層 to 中間層
        self.mid = sigmoid(self.input @ self.weight_mid)
        #中間層 to 出力層
        self.output = sigmoid(self.mid @ self.weight_out)
        return self.output

    def learn(self, dataset):
        mse = 0
        kn = self.output.size
        for i in range(dataset.shape[0]):
            for j in range(dataset.shape[1]):
                v = self.teacher[i] - self.forward(dataset[i][j])
                #重みの更新
                self.updateWeights(v)
                mse += np.sum((v * v))/kn
        
        #出力ユニットの平均2乗誤差
        mse /= self.input.size
        
        return mse

    def test(self, dataset):
        count = 0
        for i in range(dataset.shape[0]):
            for j in range(dataset.shape[1]):
                if(np.argmax(self.forward(dataset[i][j])) == i):
                    count += 100
        print(count/(dataset.shape[0]*dataset.shape[1]),"%")