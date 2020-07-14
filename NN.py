import numpy as np
import math

def sigmoid(x):
    return 1 / (1 + math.e ** (-x))

class NeuralNet:

    def __init__(self, n_in, n_mid, n_out):
        #各層の出力値
        self.input = np.zeros(n_in)
        self.mid = np.zeros(n_mid)
        self.output = np.zeros(n_out)
        #重み
        self.weight_mid = 2*np.random.random_sample((n_in,  n_mid)) - 1
        self.weight_out = 2*np.random.random_sample((n_mid, n_out)) - 1
        #重みの更新量
        self.update_mid = np.zeros((n_in,  n_mid))
        self.update_out = np.zeros((n_mid, n_out))
        #教師信号
        self.teacher = np.identity(n_out)

    def set_param(self, eta_, alpha_):
        self.eta = eta_
        self.alpha = alpha_

    def updateWeights(self,v):
        #(1,n_out)行ベクトル
        mid_vec = v*self.output*(1-self.output)

        out_vec =  (mid_vec * self.weight_out).dot(np.ones(20)) * self.mid*(1-self.mid)
        
        #(n_in,n_mid)行列
        self.update_mid = self.eta * np.array( np.matrix(self.input).T * out_vec) + self.alpha * self.update_mid

        #(n_mid,n_out)行列
        self.update_out = self.eta * np.array( np.matrix(self.mid).T * mid_vec) + self.alpha * self.update_out
        
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
        for i in range(dataset.shape[0]):
            for j in range(dataset.shape[1]):
                t = self.teacher[i] - self.forward(dataset[i][j])
                #重みの更新
                self.updateWeights(t)
                mse += np.sum((t * t))/len(t)
        
        #出力ユニットの平均2乗誤差
        mse /= dataset.shape[0]
        return mse


    
    # def mse(self, dataset):
    #     s = 0
    #     for i in range(dataset.shape[0]):
    #         for j in range(dataset.shape[1]):
    #             t = self.teacher[i] - self.forward(dataset[i][j])
    #             s += np.sum((t * t))/len(t)
    #     s /= dataset.shape[0]
    #     return s

    # def predict(self, data):
    #     y = self.forward(data)
    #     return np.argmax(y)
    
    def test(self, dataset):
        ans_list = []
        for i in range(dataset.shape[0]):
            count = 0
            for j in range(dataset.shape[1]):
                if(np.argmax(self.forward(dataset[i][j])) == i):
                    count += 100
            ans_list.append(count / dataset.shape[1])
        ans_list.append(np.array(ans_list).sum() / dataset.shape[0])
        return ans_list