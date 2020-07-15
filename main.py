import numpy as np
import os
import NN


def generateDataSet(who,how):
    data = np.empty((20,100,64),dtype='float32')

    for num in range(20):
        mesh = np.zeros((100,64), dtype='float32')
        with open("hira{}_{:0>2}{}.dat".format(who,num,how)) as f:
            p = np.array(f.readlines())
        for i in range(100):
            char = p[64*i:64*i+64]
            for j in range(64):
                for k in range(64):
                    if char[j][k] == '1':
                        mesh[i][8*(j//8)+k//8] += 1
        data[num][:] = mesh/64
    return data

def experiment1():
    writer0_L = generateDataSet(0,"L")
    writer0_T = generateDataSet(0,"T")
    writer1_T = generateDataSet(1,"T")

    nn = NN.NeuralNet(64,64,20)
    nn.set_param(0.8, 0.3)
    eps = 0.001
    ev = eps + 1
    while( eps < ev ):
        ev = nn.learn(writer0_L)
    
    print(nn.test(writer0_L))
    print(nn.test(writer0_T))
    print(nn.test(writer1_T))

def experiment2():
    writer1_L = generateDataSet(1,"L")
    writer0_T = generateDataSet(0,"T")
    writer1_T = generateDataSet(1,"T")

    nn = NN.NeuralNet(64,64,20)
    nn.set_param(0.8, 0.3)
    eps = 0.001
    ev = eps + 1
    while( eps < ev ):
        ev = nn.learn(writer1_L)
    
    print(nn.test(writer1_L))
    print(nn.test(writer0_T))
    print(nn.test(writer1_T))

def experiment3():
    writer0_L = generateDataSet(0,"L")
    writer1_L = generateDataSet(1,"L")
    writer0_T = generateDataSet(0,"T")
    writer1_T = generateDataSet(1,"T")

    writer01_L = np.empty((20,200,64),dtype='float32')
    writer01_T = np.empty((20,200,64),dtype='float32')

    for i in range(20):
        writer01_L[i][0:100] = writer0_L[i]
        writer01_L[i][100:200] = writer1_L[i]
        writer01_T[i][0:100] = writer0_T[i]
        writer01_T[i][100:200] = writer1_T[i]

    nn = NN.NeuralNet(64,64,20)
    nn.set_param(0.8, 0.3)
    eps = 0.001
    ev = eps + 1
    while( eps < ev ):
        ev = nn.learn(writer01_L)
    
    print(nn.test(writer01_T))


if __name__ == '__main__':
    os.chdir("Data")
    #experiment1()
    #experiment2()
    experiment3()