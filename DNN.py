from cmath import isnan
import numpy as np
import datetime
import matplotlib.pyplot as plt

from DNNLayer import DNNLayerType

class DNN:
    def __init__(self, numClasses, relativeStepSize, momentum, batchSize, regularization):
        self.firstTime = True

        self.numClasses = numClasses
        self.batchSize = batchSize
        self.relativeStepSize = relativeStepSize
        self.momentum = momentum
        self.regularization = regularization
        self.isLoaded = False

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
        prevCSZ = None
        while len(all) >= self.batchSize:
            bn += 1
            correctLabel = self.prepareInputOutput(ds, lb, all)
            if not self.isLoaded:
                self.normalizePars()
            self.forward(isTrain)

            if isTrain:
                self.backward()
                csz = self.update()

            sumOptim, sumCorrect = self.computeStats(correctLabel)
            print('Round {0}, isTrain {1}, Batch number {2}, optim {3}, correct {4}, ss {5}'.format(round, isTrain, bn, sumOptim, sumCorrect, self.relativeStepSize))
            optim += sumOptim 
            correct += sumCorrect

            if np.isnan(sumOptim).any() or np.isinf(sumOptim).any():
                self.debug()

            #if isTrain:
            #    if lastSumOptim is not None:
            #        self.improvements.append(lastSumOptim - sumOptim)
            #        self.stepsizes.append(prevCSZ)
            #        if len(self.improvements)>100:
            #            pars = np.polyfit(self.stepsizes, self.improvements, 2)
            #            yy = np.poly1d(pars)
            #            maxImprov = -10000
            #            maxImprovWhere = -1
            #            for i in range(11):
            #                wh = self.relativeStepSize * (0.7 + 0.6 * i / 10.0)
            #                va = yy(wh)
            #                if va > maxImprov:
            #                    maxImprov = va
            #                    maxImprovWhere = wh
            #
            #            if maxImprovWhere != -1:
            #                self.relativeStepSize = maxImprovWhere
            #            
            #            self.improvements = self.improvements[70:]
            #            self.stepsizes = self.stepsizes[70:]
            #
            #    lastSumOptim = sumOptim
            #    prevCSZ = csz

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
        curRelStepSize = self.relativeStepSize * (0.7 + 0.6 * np.random.rand())
        for i in range(len(self.layers)-1, 0, -1):
            s0 = datetime.datetime.now()
            self.layers[i].update(curRelStepSize, self.momentum, self.regularization)
            s1 = datetime.datetime.now()
            self.times[i][2] += (s1-s0).total_seconds()

        return curRelStepSize

    def saveToFile(self, fileName):
        with open(fileName, 'wb') as f:
            for i in range(len(self.layers)):
                if self.layers[i].pars is not None:
                    self.layers[i].pars.copyToCPU()
                    np.save(f, self.layers[i].pars.mem)

    def loadFromFile(self, fileName):
        self.isLoaded = True
        with open(fileName, 'rb') as f:
            for i in range(len(self.layers)):
                if self.layers[i].pars is not None:
                    self.layers[i].pars.mem = np.load(f)
                    self.layers[i].pars.copyToGPU()