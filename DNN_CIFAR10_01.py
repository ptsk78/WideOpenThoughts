from DNN import DNN
from DNNLayer import DNNLayer, DNNLayerType
import math

class DNN_CIFAR10_01(DNN):
    def __init__(self, numClasses, shape, batchSize = 50, relativeStepSize = 0.0001, momentum = 0.9, regularization=0.0):
        DNN.__init__(self, numClasses, relativeStepSize, momentum, batchSize, regularization, True)

        d1 = shape[1]
        d2 = shape[2]
        d3 = shape[3]

        self.layers = []
        self.layers.append(DNNLayer(DNNLayerType.INPUT, (batchSize, d1, d2, d3), None))
        self.layers.append(DNNLayer(DNNLayerType.PREPROCESS, (batchSize, d1, d2, d3), (batchSize, 9, 1, 1), self.layers[-1], 
        preprocessPars=[
            [-30.0 * math.pi / 180.0, 30.0 * math.pi / 180.0],
            [-0.25, 0.25],
            [-0.25, 0.25],
            [0.8, 1.2],
            [-1.0, 1.0],
            [0.0, 27.0],
            [5.0, 5.5],
            [0.0, 27.0],
            [5.0, 5.5]
        ]))

        s1 = 96
        s2 = 192

        self.layers.append(DNNLayer(DNNLayerType.FINCONVO, (batchSize, s1, d2-2, d3-2), (d1, 3, 3, s1), self.layers[-1]))
        d1 = s1
        d2 -= 2
        d3 -= 2
        self.layers.append(DNNLayer(DNNLayerType.FRELU, (batchSize, d1, d2, d3), None, self.layers[-1]))

        self.layers.append(DNNLayer(DNNLayerType.FINCONVO, (batchSize, s1, d2-2, d3-2), (d1, 3, 3, s1), self.layers[-1]))
        d1 = s1
        d2 -= 2
        d3 -= 2
        self.layers.append(DNNLayer(DNNLayerType.FRELU, (batchSize, d1, d2, d3), None, self.layers[-1]))

        self.layers.append(DNNLayer(DNNLayerType.FINCONVO, (batchSize, s1, d2-2, d3-2), (d1, 3, 3, s1), self.layers[-1]))
        d1 = s1
        d2 -= 2
        d3 -= 2
        self.layers.append(DNNLayer(DNNLayerType.FRELU, (batchSize, d1, d2, d3), None, self.layers[-1]))

        self.layers.append(DNNLayer(DNNLayerType.MAX, (batchSize, d1, d2 // 2, d3 // 2), None, self.layers[-1]))
        d2 = d2 // 2
        d3 = d3 // 2

        self.layers.append(DNNLayer(DNNLayerType.FINCONVO, (batchSize, s2, d2-2, d3-2), (d1, 3, 3, s2), self.layers[-1]))
        d1 = s2
        d2 -= 2
        d3 -= 2
        self.layers.append(DNNLayer(DNNLayerType.FRELU, (batchSize, d1, d2, d3), None, self.layers[-1]))

        self.layers.append(DNNLayer(DNNLayerType.FINCONVO, (batchSize, s2, d2-2, d3-2), (d1, 3, 3, s2), self.layers[-1]))
        d1 = s2
        d2 -= 2
        d3 -= 2
        self.layers.append(DNNLayer(DNNLayerType.FRELU, (batchSize, d1, d2, d3), None, self.layers[-1]))

        self.layers.append(DNNLayer(DNNLayerType.FINCONVO, (batchSize, s2, d2-1, d3-1), (d1, 2, 2, s2), self.layers[-1]))
        d1 = s2
        d2 -= 1
        d3 -= 1
        self.layers.append(DNNLayer(DNNLayerType.FRELU, (batchSize, d1, d2, d3), None, self.layers[-1]))

        self.layers.append(DNNLayer(DNNLayerType.MAX, (batchSize, d1, d2 // 2, d3 // 2), None, self.layers[-1]))
        d2 = d2 // 2
        d3 = d3 // 2

        self.layers.append(DNNLayer(DNNLayerType.FINCONVO, (batchSize, s2, d2-2, d3-2), (d1, 3, 3, s2), self.layers[-1]))
        d1 = s2
        d2 -= 2
        d3 -= 2
        self.layers.append(DNNLayer(DNNLayerType.FRELU, (batchSize, d1, d2, d3), None, self.layers[-1]))

        self.layers.append(DNNLayer(DNNLayerType.FINCONVO, (batchSize, s2, d2, d3), (d1, 1, 1, s2), self.layers[-1]))
        d1 = s2
        self.layers.append(DNNLayer(DNNLayerType.FRELU, (batchSize, d1, d2, d3), None, self.layers[-1]))

        self.layers.append(DNNLayer(DNNLayerType.FINCONVO, (batchSize, s2, d2, d3), (d1, 1, 1, s2), self.layers[-1]))
        d1 = s2
        self.layers.append(DNNLayer(DNNLayerType.FRELU, (batchSize, d1, d2, d3), None, self.layers[-1]))

        self.layers.append(DNNLayer(DNNLayerType.AVER, (batchSize, d1, 1, 1), None, self.layers[-1]))
        d2 = 1
        d3 = 1

        self.layers.append(DNNLayer(DNNLayerType.DROPOUT, (batchSize, d1 * d2 * d3, 1, 1), (batchSize, d1 * d2 * d3, 1, 1), self.layers[-1], extraPar=0.25))

        self.layers.append(DNNLayer(DNNLayerType.MATRIX, (batchSize, numClasses, 1, 1), (numClasses, d1 * d2 * d3 + 1, 1, 1), self.layers[-1]))
        self.layers.append(DNNLayer(DNNLayerType.SOFTMAX, (batchSize, numClasses, 1, 1), None, self.layers[-1]))

        self.layers.append(DNNLayer(DNNLayerType.FINAL, (batchSize, 1, 1, 1), (batchSize, numClasses, 1, 1), self.layers[-1]))
