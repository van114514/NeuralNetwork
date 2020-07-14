import numpy as np
import os
import NN as nn

def generateDateSet(who,how):
    data = np.empty((20,100,64),dtype='float32')


    for i in range(20):
        print("hira{}_{:0>2}{}".format(who,i,how))
    




    return data



if __name__ == '__main__':
    #experiment1()
    #experiment2()
    #experiment3()

    os.chdir("Data")
    #for file in os.listdir(currentdir):
    #    with open(file, encoding="utf-8") as f:
    #        yield f.read().replace('\n','')

    generateDateSet(0,"L")