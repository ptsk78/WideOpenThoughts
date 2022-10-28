from DNN import DNN
from DNNLayer import DNNLayer, DNNLayerType
import math

class DNN_MNIST_01(DNN):
    def __init__(self, numClasses, shape, batchSize = 200, relativeStepSize = 0.001, momentum = 0.0, regularization=0.0):
        DNN.__init__(self, numClasses, relativeStepSize, momentum, batchSize, regularization)

        self.numClasses = numClasses
        self.batchSize = batchSize
        self.relativeStepSize = relativeStepSize

        d1 = shape[1]
        d2 = shape[2]
        d3 = shape[3]

        self.layers = []
        self.layers.append(DNNLayer(DNNLayerType.INPUT, (batchSize, d1, d2, d3), None))
        self.layers.append(DNNLayer(DNNLayerType.PREPROCESS, (batchSize, d1, d2, d3), (batchSize, 7, 1, 1), self.layers[-1], 
        preprocessPars=[
            [-30.0 * math.pi / 180.0, 30.0 * math.pi / 180.0],
            [-0.25, 0.25],
            [-0.25, 0.25],
            [0.8, 1.2],
            [0.5, 1.0],
            [0.0, 23.0],
            [0.0, 23.0]
        ]))

        self.layers.append(DNNLayer(DNNLayerType.CONVO, (batchSize, 200, d2-7, d3-7), (d1, 8, 8, 200), self.layers[-1]))
        self.layers.append(DNNLayer(DNNLayerType.MAX, (batchSize, 200, 3, 3), None, self.layers[-1]))

        self.layers.append(DNNLayer(DNNLayerType.MATRIX, (batchSize, 130, 1, 1), (130, 200 * 3 * 3 + 1, 1, 1), self.layers[-1]))
        self.layers.append(DNNLayer(DNNLayerType.SIGMOID, (batchSize, 130, 1, 1), None, self.layers[-1]))
        self.layers.append(DNNLayer(DNNLayerType.DROPOUT, (batchSize, 130, 1, 1), (batchSize, 130, 1, 1), self.layers[-1], extraPar=0.5))

        self.layers.append(DNNLayer(DNNLayerType.MATRIX, (batchSize, numClasses, 1, 1), (numClasses, 130 + 1, 1, 1), self.layers[-1]))
        self.layers.append(DNNLayer(DNNLayerType.SOFTMAX, (batchSize, numClasses, 1, 1), None, self.layers[-1]))

        self.layers.append(DNNLayer(DNNLayerType.FINAL, (batchSize, 1, 1, 1), (batchSize, numClasses, 1, 1), self.layers[-1]))
