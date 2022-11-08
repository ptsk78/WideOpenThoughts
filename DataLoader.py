import numpy as np
import struct
import sys
from DataDownloader import download

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
        ret /= 255.0
 
    return ret

def getMNISTData():
    return readImageFile('./data/mnist/train-images-idx3-ubyte'), readLabelFile('./data/mnist/train-labels-idx1-ubyte'), readImageFile('./data/mnist/t10k-images-idx3-ubyte'), readLabelFile('./data/mnist/t10k-labels-idx1-ubyte')

def readADSFile(fn):
    ret = []

    f = open(fn, 'rt')
    while True:
        tmp = f.readline()
        if len(tmp) == 0:
            break
        tmp2 = tmp.split(',')
        if len(tmp2)==15:
            ret.append(tmp2)
    
    f.close()
        
    return ret

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

def translateADSData(data, feats, featuresMinMax):
    numFeats = 0
    for i in range(len(feats)):
        if i<len(feats)-1:
            if len(feats[i]) == 0:
                numFeats +=1
            else:
                numFeats += len(feats[i])

    retData = np.zeros((len(data), numFeats, 1, 1), np.float32)
    retLabels = np.zeros((len(data)), np.uint8)
    
    for i in range(len(data)):
        j = 0
        for ii in range(len(feats)-1):
            if len(feats[ii]) == 0:
                retData[i, j, 0, 0] = (float(data[i][ii].strip()) - featuresMinMax[ii][0]) / (featuresMinMax[ii][1]- featuresMinMax[ii][0]) 
                j += 1
            else:
                retData[i, j + feats[ii][data[i][ii].strip()], 0, 0] = float(1.0)
                j += len(feats[ii])

        # in test they have '.' at the end - remove it
        tmp = data[i][-1].strip()
        if tmp.endswith('.'):
            tmp = tmp[:-1]
        retLabels[i] = feats[-1][tmp]
    
    return retData, retLabels

def getADSData():
    train = readADSFile('./data/ads/adult.data')
    test = readADSFile('./data/ads/adult.test')
    
    features = [{} for i in range(15)]
    featuresMinMax = [[sys.float_info.max, sys.float_info.min] for i in range(15)]
    
    # find possible values
    for i in range(15):
        if not isfloat(train[0][i]):
            for j in range(len(train)):
                x = train[j][i].strip()
                if x not in features[i]:
                    features[i][x] = len(features[i])
            for j in range(len(test)):
                x = test[j][i].strip()
                # in test they have '.' at the end - remove it
                if x not in features[i] and x[:-1] not in features[i]:
                    print(x)
                    print(features[i])
                    print('unexpected')
                    exit(1)
        else:
            for j in range(len(train)):
                x = float(train[j][i].strip())
                if featuresMinMax[i][0] > x:
                    featuresMinMax[i][0] = x
                if featuresMinMax[i][1] < x:
                    featuresMinMax[i][1] = x
                
    d, dl = translateADSData(train, features, featuresMinMax)
    dt, dtl = translateADSData(test, features, featuresMinMax)
    
    return d, dl, dt, dtl

class DataSet:
    def __init__(self, dataSetName):
        download(dataSetName)
        if dataSetName == 'Cifar':
            self.data, self.labels, self.dataTest, self.labelsTest = getCifarData()
            self.isImage = True
        if dataSetName == 'MNIST':
            self.data, self.labels, self.dataTest, self.labelsTest = getMNISTData()
            self.isImage = True
        if dataSetName == 'ADS':
            self.data, self.labels, self.dataTest, self.labelsTest = getADSData()
            self.isImage = False
