from DNN import DNN
from DNNLayer import DNNLayer, DNNLayerType
import math

class DNN_03(DNN):
    def __init__(self, numClasses, shape, batchSize = 50, relativeStepSize = 0.001, momentum = 0.9, regularization=0.0):
        DNN.__init__(self, numClasses, relativeStepSize, momentum, batchSize, regularization)

        d1 = shape[1]
        d2 = shape[2]
        d3 = shape[3]

        self.layers = []
        self.layers.append(DNNLayer(DNNLayerType.INPUT, (batchSize, d1, d2, d3), None))
        self.layers.append(DNNLayer(DNNLayerType.PREPROCESS, (batchSize, d1, d2, d3), (batchSize, 5, 1, 1), self.layers[-1], 
        preprocessPars=[
            [-40.0* math.pi / 180.0, 40.0* math.pi / 180.0],
            [-0.4,0.4],
            [-0.4,0.4],
            [0.6,1.4],
            [-1.0,1.0]
        ]))

        self.layers.append(DNNLayer(DNNLayerType.CONVO, (batchSize, 8, d2-2, d3-2), (d1, 3, 3, 8), self.layers[-1]))
        d1 = 8
        d2 -= 2
        d3 -= 2
        self.layers.append(DNNLayer(DNNLayerType.FRELU, (batchSize, d1, d2, d3), None, self.layers[-1]))
        self.layers.append(DNNLayer(DNNLayerType.CONVO, (batchSize, 16, d2-2, d3-2), (d1, 3, 3, 16), self.layers[-1]))
        d1 = 16
        d2 -= 2
        d3 -= 2
        self.layers.append(DNNLayer(DNNLayerType.FRELU, (batchSize, d1, d2, d3), None, self.layers[-1]))
        self.layers.append(DNNLayer(DNNLayerType.CONVO, (batchSize, 32, d2-2, d3-2), (d1, 3, 3, 32), self.layers[-1]))
        d1 = 32
        d2 -= 2
        d3 -= 2
        self.layers.append(DNNLayer(DNNLayerType.FRELU, (batchSize, d1, d2, d3), None, self.layers[-1]))
        self.layers.append(DNNLayer(DNNLayerType.CONVO, (batchSize, 512, d2-6, d3-6), (d1, 7, 7, 512), self.layers[-1]))
        d1 = 256
        d2 -= 6
        d3 -= 6
        self.layers.append(DNNLayer(DNNLayerType.MAX, (batchSize, d1, 3, 3), None, self.layers[-1]))
        d2 = 3
        d3 = 3

        self.layers.append(DNNLayer(DNNLayerType.DROPOUT, (batchSize, d1 * d2 * d3, 1, 1), (batchSize, d1 * d2 * d3, 1, 1), self.layers[-1], extraPar=0.25))
        self.layers.append(DNNLayer(DNNLayerType.MATRIX, (batchSize, 200, 1, 1), (200, d1 * d2 * d3 + 1, 1, 1), self.layers[-1]))
        self.layers.append(DNNLayer(DNNLayerType.SIGMOID, (batchSize, 200, 1, 1), None, self.layers[-1]))
        self.layers.append(DNNLayer(DNNLayerType.DROPOUT, (batchSize, 200, 1, 1), (batchSize, 200, 1, 1), self.layers[-1], extraPar=0.25))
        self.layers.append(DNNLayer(DNNLayerType.MATRIX, (batchSize, numClasses, 1, 1), (numClasses, 200 + 1, 1, 1), self.layers[-1]))
        self.layers.append(DNNLayer(DNNLayerType.SOFTMAX, (batchSize, numClasses, 1, 1), None, self.layers[-1]))

        self.layers.append(DNNLayer(DNNLayerType.FINAL, (batchSize, 1, 1, 1), (batchSize, numClasses, 1, 1), self.layers[-1]))
