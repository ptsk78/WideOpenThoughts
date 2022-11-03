from cmath import isnan
import numpy as np
import datetime
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

from DNNLayer import DNNLayerType

class DNN:
    def __init__(self, numClasses, stepSize, momentum, batchSize, regularization, relativeStepSize):
        self.firstTime = True

        self.numClasses = numClasses
        self.batchSize = batchSize
        self.stepSize = stepSize
        self.momentum = momentum
        self.regularization = regularization
        self.isLoaded = False
        self.relativeStepSize = relativeStepSize

        self.improvements = []
        self.stepsizes = []

    def go(self, round, ds, lb, isTrain = False):
        all = list(range(ds.shape[0]))

        bn = 0
        optim = 0
        correct = 0

        self.times = [[0, 0, 0] for i in range(len(self.layers))]
        self.confusionMatrix = np.zeros((self.numClasses, self.numClasses), np.int32)
        self.correctProbs =[[] for i in range(self.numClasses)]
        self.incorrectProbs =[[] for i in range(self.numClasses)]

        lastSumOptim = None
        while len(all) >= self.batchSize:
            bn += 1
            correctLabel = self.prepareInputOutput(ds, lb, all)
            if not self.isLoaded:
                self.normalizePars()
            self.forward(isTrain)

            if isTrain:
                self.backward()
                self.update()

            sumOptim, sumCorrect = self.computeStats(correctLabel)
            print('Round {0}, isTrain {1}, Batch number {2}, optim {3}, correct {4}'.format(round, isTrain, bn, sumOptim, sumCorrect))
            optim += sumOptim 
            correct += sumCorrect

            if np.isnan(sumOptim).any() or np.isinf(sumOptim).any():
                self.debug()

        for i in range(len(self.layers)):
            print("{0} {1} {2:.4f} {3:.4f} {4:.4f}".format(i, self.layers[i].layerType, self.times[i][0], self.times[i][1], self.times[i][2]))

        optim /= bn
        correct /= bn
        print(self.confusionMatrix)

        self.layers[0].output.copyToCPU()
        self.layers[1].output.copyToCPU()

        return optim, correct, self.confusionMatrix, self.correctProbs, self.incorrectProbs, self.times, np.array(self.layers[0].output.mem), np.array(self.layers[1].output.mem)

    def computeStats(self, correctLabel):
        self.layers[-1].output.copyToCPU()
        self.layers[-2].output.copyToCPU()
        where = np.argmax(self.layers[-2].output.mem, axis=1).reshape(-1)

        sumOptim = np.sum(self.layers[-1].output.mem) / self.batchSize
        sumCorrect = np.sum(np.equal(correctLabel, where)) / self.batchSize
        for i in range(len(correctLabel)):
            self.confusionMatrix[correctLabel[i], where[i]] += 1
        for i in range(self.layers[-2].output.mem.shape[0]):
            for j in range(self.layers[-2].output.mem.shape[1]):
                if j == correctLabel[i]:
                    self.correctProbs[correctLabel[i]].append(self.layers[-2].output.mem[i,j,0,0])
                else:
                    self.incorrectProbs[correctLabel[i]].append(self.layers[-2].output.mem[i,j,0,0])

        return sumOptim, sumCorrect

    def checkMem(self, i, mem, lt, whatmem):
        mem.copyToCPU()
        print('Layer {0} {1} {4} is [{2},{3}]'.format(i, lt, np.min(mem.mem), np.max(mem.mem), whatmem))
        
    def debug(self):
        print('debugging info')
        for i in range(0,len(self.layers)):
            self.checkMem(i, self.layers[i].output, self.layers[i].layerType, 'output')
            if self.layers[i].outputDer is not None:
                self.checkMem(i, self.layers[i].outputDer, self.layers[i].layerType, 'output der')
            if self.layers[i].pars is not None:
                self.checkMem(i, self.layers[i].pars, self.layers[i].layerType, 'pars')
            if self.layers[i].parsDer is not None:
                self.checkMem(i, self.layers[i].parsDer, self.layers[i].layerType, 'pars der')

        exit(1)

    def prepareInputOutput(self, ds, lb, all):
        correctLabel = []
        for i in range(self.batchSize):
            c = np.random.randint(len(all))
            self.layers[0].output.mem[i,:,:,:] = ds[all[c],:,:,:]
            
            self.layers[-1].pars.mem[i,:,:,:] = np.zeros(self.layers[-1].pars.mem.shape[1:], np.float32)
            self.layers[-1].pars.mem[i,lb[all[c]],0,0] = 1.0
            correctLabel.append(lb[all[c]])
            
            del all[c]

        correctLabel = np.array(correctLabel).reshape(-1)
        self.layers[0].output.copyToGPU()
        self.layers[-1].pars.copyToGPU()

        return correctLabel

    def normalizePars(self):
        if self.firstTime:
            for i in range(1,len(self.layers)):
                self.layers[i].normalizePars()
            self.firstTime = False

            for i in range(len(self.layers)):
                print(i, self.layers[i].layerType, None if i == 0  else self.layers[i].prev.outputSize, self.layers[i].outputSize, self.layers[i].parsSize)

    def forward(self, isTrain):
        for i in range(1,len(self.layers)):
            s0 = datetime.datetime.now()
            self.layers[i].forward(isTrain)
            s1 = datetime.datetime.now()
            self.times[i][0] += (s1-s0).total_seconds()

    def backward(self):
        for i in range(len(self.layers)-1, 0, -1):
            s0 = datetime.datetime.now()
            self.layers[i].backward()
            s1 = datetime.datetime.now()
            self.times[i][1] += (s1-s0).total_seconds()

    def update(self):
        for i in range(len(self.layers)-1, 0, -1):
            s0 = datetime.datetime.now()
            self.layers[i].update(self.stepSize, self.momentum, self.regularization, self.relativeStepSize)
            s1 = datetime.datetime.now()
            self.times[i][2] += (s1-s0).total_seconds()

    def saveToFile(self, fileName):
        with open(fileName, 'wb') as f:
            for i in range(len(self.layers)):
                if self.layers[i].pars is not None:
                    self.layers[i].pars.copyToCPU()
                    np.save(f, self.layers[i].pars.mem)

    def loadFromDir(self, dirname, newdir):
        onlyfiles = [f for f in listdir(dirname) if isfile(join(dirname, f))]    
        round = 0
        dnnfile = ''
        for file in onlyfiles:
            if file.startswith('dnn_') and file.endswith('.npy'):
                tmp = int(file[4:-4])
                if tmp>=round:
                    round = tmp
                    dnnfile = file
        round += 1
        opt = []
        optT = []
        cor = []
        corT = []
        file = open(join(dirname, 'debug.txt'), 'r')
        fileW = open(join(newdir, 'debug.txt'), 'wt')
        for i in range(round):
            line = file.readline()
            fileW.write(line)
            tmp = line.split(',')
            opt.append(float(tmp[1]))
            cor.append(float(tmp[2]))
            optT.append(float(tmp[3]))
            corT.append(float(tmp[4]))
        file.close()
        fileW.close()

        self.isLoaded = True
        with open(join(dirname, dnnfile), 'rb') as f:
            for i in range(len(self.layers)):
                if self.layers[i].pars is not None:
                    self.layers[i].pars.mem = np.load(f)
                    self.layers[i].pars.copyToGPU()
                    
        return opt, optT, cor, corT, round
