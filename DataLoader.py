import numpy as np
import struct

def unpickleCifar(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict[b'data'], dict[b'labels']

def getCifarData():
    data = []
    labels = []
    for i in range(1,6):
        dataT, labelsT = unpickleCifar('./data/cifar-10-batches-py/data_batch_{}'.format(i))
        data.extend(dataT)
        labels.extend(labelsT)

    dataTest, labelsTest = unpickleCifar('./data/cifar-10-batches-py/test_batch')

    return (np.array(data, np.float32).reshape(-1, 3, 32, 32)-0) / 255.0, np.array(labels, np.uint8), (np.array(dataTest, np.float32).reshape(-1, 3, 32, 32)-0) / 255.0, np.array(labelsTest, np.uint8)

def readLabelFile(fileName):
    with open(fileName, 'rb') as file:
        r = file.read(4)
        if r != b'\x00\x00\x08\x01':
            raise Exception('wrong input label file {0}'.format(fileName)) 
        r = file.read(4)
        ni = r[0] * 256 * 256 * 256 + r[1] * 256 * 256 + r[2] * 256 + r[3]

        ret = np.array(struct.unpack('B'*(ni), file.read(ni)), np.uint8)

    return ret

def readImageFile(fileName):
    with open(fileName, 'rb') as file:
        r = file.read(4)
        if r != b'\x00\x00\x08\x03':
            raise Exception('wrong input label file {0}'.format(fileName)) 
        r = file.read(4)
        ni = r[0] * 256 * 256 * 256 + r[1] * 256 * 256 + r[2] * 256 + r[3]

        r = file.read(4)
        x1 = r[0] * 256 * 256 * 256 + r[1] * 256 * 256 + r[2] * 256 + r[3]

        r = file.read(4)
        x2 = r[0] * 256 * 256 * 256 + r[1] * 256 * 256 + r[2] * 256 + r[3]

        ret = np.array(struct.unpack('B'*(ni * x1 * x2), file.read(ni * x1 * x2)), np.float32).reshape(ni,1,x1,x2)
        ret -= 127.5
        ret /= 127.5
 
    return ret

def getMNISTData():
    return readImageFile('./data/mnist/train-images-idx3-ubyte'), readLabelFile('./data/mnist/train-labels-idx1-ubyte'), readImageFile('./data/mnist/t10k-images-idx3-ubyte'), readLabelFile('./data/mnist/t10k-labels-idx1-ubyte')

class DataSet:
    def __init__(self, dataSetName):
        if dataSetName == 'Cifar':
            self.data, self.labels, self.dataTest, self.labelsTest = getCifarData()
        elif dataSetName == 'MNIST':
            self.data, self.labels, self.dataTest, self.labelsTest = getMNISTData()
