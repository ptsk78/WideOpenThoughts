from DNN import DNN
from DNNLayer import DNNLayer, DNNLayerType
import math

class DNN_ADS_01(DNN):
    def __init__(self, numClasses, shape, batchSize = 1000, stepSize = 0.001, momentum = 0.0, regularization=0.0):
        DNN.__init__(self, numClasses, stepSize, momentum, batchSize, regularization, True)

        d1 = shape[1]
        d2 = shape[2]
        d3 = shape[3]
        print(d1,d2,d3)

        self.layers = []
        self.layers.append(DNNLayer(DNNLayerType.INPUT, (batchSize, d1, d2, d3), None))

        self.layers.append(DNNLayer(DNNLayerType.MATRIX, (batchSize, 100, 1, 1), (100, d1 * d2 * d3 + 1, 1, 1), self.layers[-1]))
        self.layers.append(DNNLayer(DNNLayerType.FRELU, (batchSize, 100, 1, 1), None, self.layers[-1]))

        self.layers.append(DNNLayer(DNNLayerType.DROPOUT, (batchSize, 100, 1, 1), (batchSize, 100, 1, 1), self.layers[-1], extraPar=0.5))

        self.layers.append(DNNLayer(DNNLayerType.MATRIX, (batchSize, 50, 1, 1), (50, 100 + 1, 1, 1), self.layers[-1]))
        self.layers.append(DNNLayer(DNNLayerType.SIGMOID, (batchSize, 50, 1, 1), None, self.layers[-1]))

        self.layers.append(DNNLayer(DNNLayerType.MATRIX, (batchSize, numClasses, 1, 1), (numClasses, 50 + 1, 1, 1), self.layers[-1]))
        self.layers.append(DNNLayer(DNNLayerType.SOFTMAX, (batchSize, numClasses, 1, 1), None, self.layers[-1]))

        self.layers.append(DNNLayer(DNNLayerType.FINAL, (batchSize, 1, 1, 1), (batchSize, numClasses, 1, 1), self.layers[-1]))
