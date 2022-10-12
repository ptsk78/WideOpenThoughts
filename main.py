from DataLoader import DataSet
from DNN_05 import DNN_05
from DNN_04 import DNN_04
from DNN_03 import DNN_03
from DNN_02 import DNN_02
from DNN_01 import DNN_01
from plots import makePlot
import os.path
import os

expNum = 0
while os.path.isdir('./exp_{0:05d}/'.format(expNum)):
    expNum += 1
curDir = './exp_{0:05d}/'.format(expNum)
os.mkdir(curDir)

ds = DataSet('Cifar')
dnn = DNN_05(10, ds.data.shape)
#dnn.loadFromFile('./exp_00004/dnn_01400.npy')

#ds = DataSet('MNIST')
#dnn = DNNMNIST(10, ds.data.shape)

opt = []
optT = []
cor = []
corT = []
round = 0
while True:
    optim, correct, confMatr, correctProbs, incorrectProbs, times, im1, im2 = dnn.go(round, ds.data, ds.labels, isTrain=True)
    optimT, correctT, confMatrT, correctProbsT, incorrectProbsT, timesT, im1T, im2T = dnn.go(round, ds.dataTest, ds.labelsTest, isTrain=False)

    opt.append(optim)
    cor.append(correct)
    optT.append(optimT)
    corT.append(correctT)
    makePlot(opt, optT, cor, corT, confMatr, correctProbs, incorrectProbs, times, confMatrT, correctProbsT, incorrectProbsT, timesT, im1, im2, '{0}result_{1:05d}.png'.format(curDir, round))

    print('{0}, {1}, {2}, {3}, {4}'.format(round, optim, correct, optimT, correctT))
    f = open('{0}debug.txt'.format(curDir), 'at')
    f.write('{0}, {1}, {2}, {3}, {4}\n'.format(round, optim, correct, optimT, correctT))
    f.close()

    if round % 100 == 0:
        dnn.saveToFile('{0}dnn_{1:05d}.npy'.format(curDir, round))

    round += 1
