import numpy as np
import os
import NN

def partially_feature(p, ii, jj):
    feature = 0.0
    for i in range(8):
        for j in range(8):
            if(p[8*ii + i][8*jj + j] == '1'):
                feature += 1
    return feature / 64

def extract_feature(p):
    buf = np.empty(64, dtype='float32')
    for i in range(8):
        for j in range(8):
            buf[8*i + j] = partially_feature(p, i, j)
    return buf

def read_onefile(path):
    buf = np.empty((100,64), dtype='float32')
    with open(path) as f:
        p = np.array(f.readlines())
    for charas in range(100):
        buf[charas][:] = extract_feature(p[64*charas:64*charas+64])
    return buf



def generateDateSet(who,how):
    data = np.empty((20,100,64),dtype='float32')

    for i in range(20):
        data[i][:] = read_onefile("hira{}_{:0>2}{}.dat".format(who,i,how))
    
    return data


def experiment1():
    writer0_L = generateDateSet(0,"L")
    writer0_T = generateDateSet(0,"T")
    writer1_T = generateDateSet(1,"T")

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
    writer1_L = generateDateSet(1,"L")
    writer0_T = generateDateSet(0,"T")
    writer1_T = generateDateSet(1,"T")

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
    writer0_L = generateDateSet(0,"L")
    writer1_L = generateDateSet(1,"L")
    writer0_T = generateDateSet(0,"T")
    writer1_T = generateDateSet(1,"T")

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