import numpy as np
from GPUMemory import GPUMemory
from enum import Enum
import pyopencl as cl

class DNNLayerType(Enum):
    INPUT = 1
    PREPROCESS = 2
    NOISE = 3
    CONVO = 4
    RELU = 5
    FRELU = 6
    FFRELU = 7
    FFFRELU = 8
    AVER = 9
    MAX = 10
    MATRIX = 11
    DROPOUT = 12
    SIGMOID = 13
    SOFTMAX = 14
    SUM = 15
    FINAL = 16

class DNNLayer:
    forwardBaseCode = """
__kernel void forward(
    __global const float *input, __global const float *input2, __global const float *pars, __global float *output, int i1, int i2, int i3, int i4, int o1, int o2, int o3, int o4, int isTrain, float extraPar)
{
    uint gid = get_global_id(0);
    uint ngid = gid;

    """
    backwardBaseCode = """
__kernel void backward(
    __global const float *input, __global const float *input2, __global const float *pars, __global const float *output, __global const float *outputDer, __global float *parsDer, __global float *inputDer, __global float *inputDer2, int i1, int i2, int i3, int i4, int o1, int o2, int o3, int o4, float extraPar)
{
    uint gid = get_global_id(0);
    uint ngid = gid;

    """

    def __init__(self, layerType, outputSize, parsSize, prev=None, prev2=None, preprocessPars = None, extraPar = 0.0):
        self.outputSize = outputSize
        self.parsSize = parsSize
        self.prev = prev
        self.prev2 = prev2
        self.layerType = layerType
        self.preprocessPars = preprocessPars
        self.extraPar = extraPar

        self.forwardCode = None
        self.backwardCode = None
        self.backwardInputCode = None
        self.backwardParsCode = None


        if parsSize is not None:
            #self.pars = GPUMemory((2.0 * np.random.rand(*parsSize) - 1.0).astype(np.float32))
            if self.layerType == DNNLayerType.CONVO:
                #self.pars = GPUMemory(np.random.rand(*parsSize).astype(np.float32))
                self.pars = GPUMemory((2.0 * np.random.rand(*parsSize) - 1.0).astype(np.float32))
            else:
                self.pars = GPUMemory((2.0 * np.random.rand(*parsSize) - 1.0).astype(np.float32))
            self.parsDer = GPUMemory(np.zeros(parsSize, np.float32))
        else:
            self.pars = None
            self.parsDer = None

        self.output = GPUMemory(np.zeros(outputSize, np.float32))
        if self.layerType is not DNNLayerType.INPUT:
            self.outputDer = GPUMemory(np.zeros(outputSize, np.float32))
        else:
            self.outputDer = None

        self.loadCode()

        if self.forwardCode is not None:
            self.prgForward = cl.Program(GPUMemory.ctx, self.forwardCode).build()
        else:
            self.prgForward = None
        if self.backwardCode is not None:
            self.prgBackward = cl.Program(GPUMemory.ctx, self.backwardCode).build()
        else:
            self.prgBackward = None
        if self.backwardInputCode is not None:
            self.prgBackwardInput = cl.Program(GPUMemory.ctx, self.backwardInputCode).build()
        else:
            self.prgBackwardInput = None
        if self.backwardParsCode is not None:
            self.prgBackwardPars = cl.Program(GPUMemory.ctx, self.backwardParsCode).build()
        else:
            self.prgBackwardPars = None
        self.prgUpdate = cl.Program(GPUMemory.ctx, self.updateCode).build()

    def normalizePars(self):
        self.forward(True)
        if self.layerType == DNNLayerType.MATRIX or self.layerType == DNNLayerType.CONVO:
            self.output.copyToCPU()
            #norm = np.sum(np.abs(self.output.mem)) / (self.output.mem.shape[0] * self.output.mem.shape[1] * self.output.mem.shape[2] * self.output.mem.shape[3])
            norm = np.max(np.abs(self.output.mem)) + 1.0
            self.pars.mem /= norm
            self.pars.copyToGPU()
            self.forward(True)


    def forward(self, isTrain):
        if self.layerType == DNNLayerType.DROPOUT or self.layerType == DNNLayerType.NOISE or self.layerType == DNNLayerType.PREPROCESS:
            if self.layerType == DNNLayerType.NOISE:
                self.pars.mem = (np.random.rand(*self.parsSize).astype(np.float32) * 2.0 - 1.0) * self.extraPar
            elif self.layerType == DNNLayerType.PREPROCESS:
                self.pars.mem = np.random.rand(*self.parsSize).astype(np.float32)
                for i in range(self.parsSize[1]):
                    self.pars.mem[:,i,:,:] *= self.preprocessPars[i][1] - self.preprocessPars[i][0]
                    self.pars.mem[:,i,:,:] += self.preprocessPars[i][0]
            else:
                self.pars.mem = np.random.rand(*self.parsSize).astype(np.float32)
            self.pars.copyToGPU()
        knl = self.prgForward.forward
        knl.set_scalar_arg_dtypes([None, None, None, None, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32, np.float32])
        knl(GPUMemory.queue, (self.outputSize[0]*self.outputSize[1]*self.outputSize[2]*self.outputSize[3],), None, 
        self.prev.output.memGPU, 
        None if self.prev2 is None else self.prev2.output.memGPU,
        None if self.pars is None else self.pars.memGPU, 
        self.output.memGPU, 
        np.int32(self.prev.output.mem.shape[0]), np.int32(self.prev.output.mem.shape[1]), np.int32(self.prev.output.mem.shape[2]), np.int32(self.prev.output.mem.shape[3]),
        np.int32(self.output.mem.shape[0]), np.int32(self.output.mem.shape[1]), np.int32(self.output.mem.shape[2]), np.int32(self.output.mem.shape[3]), 
        np.int32(1 if isTrain else 0), np.float32(self.extraPar))
        GPUMemory.queue.finish()

    def executeBackward(self, knl, ks):
        knl.set_scalar_arg_dtypes([None, None, None, None, None, None, None, None, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32, np.float32])
        knl(GPUMemory.queue, ks, None, 
        self.prev.output.memGPU, 
        None if self.prev2 is None else self.prev2.output.memGPU,
        None if self.pars is None else self.pars.memGPU, 
        self.output.memGPU, 
        self.outputDer.memGPU, 
        None if self.parsDer is None else self.parsDer.memGPU, 
        None if self.prev is None or self.prev.outputDer is None else self.prev.outputDer.memGPU,
        None if self.prev2 is None or self.prev2.outputDer is None else self.prev2.outputDer.memGPU,
        np.int32(self.prev.output.mem.shape[0]), np.int32(self.prev.output.mem.shape[1]), np.int32(self.prev.output.mem.shape[2]), np.int32(self.prev.output.mem.shape[3]),
        np.int32(self.output.mem.shape[0]), np.int32(self.output.mem.shape[1]), np.int32(self.output.mem.shape[2]), np.int32(self.output.mem.shape[3]),
        np.float32(self.extraPar))
        GPUMemory.queue.finish()

    def backward(self):
        if self.prgBackward is not None and self.prev.outputDer is not None:
            self.executeBackward(self.prgBackward.backward, (self.outputSize[0]*self.outputSize[1]*self.outputSize[2]*self.outputSize[3],))
        if self.prgBackwardPars is not None:
            self.executeBackward(self.prgBackwardPars.backward, (self.parsSize[0]*self.parsSize[1]*self.parsSize[2]*self.parsSize[3],))
        if self.prgBackwardInput is not None and self.prev.outputDer is not None:
            self.executeBackward(self.prgBackwardInput.backward, (self.prev.outputSize[0]*self.prev.outputSize[1]*self.prev.outputSize[2]*self.prev.outputSize[3],))

    def update(self, relativeStepSize, momentum, regularization):
        if self.pars is not None and self.parsDer is not None:
            self.pars.copyToCPU()
            self.parsDer.copyToCPU()
            sumpars = np.sum(np.abs(self.pars.mem)) + 0.001
            sumparsDer = np.sum(np.abs(self.parsDer.mem)) + 0.001
            stepSize = sumpars / sumparsDer * relativeStepSize
            knl = self.prgUpdate.update
            knl.set_scalar_arg_dtypes([None, None, np.float32, np.float32, np.float32])
            knl(GPUMemory.queue, (self.parsSize[0]*self.parsSize[1]*self.parsSize[2]*self.parsSize[3],), None, 
            self.pars.memGPU, self.parsDer.memGPU, np.float32(stepSize), np.float32(momentum), np.float32(regularization))
            GPUMemory.queue.finish()

        if self.outputDer is not None:
            self.outputDer.reset()

    def loadCode(self):
        self.updateCode = """
__kernel void update(
    __global float *pars, __global float *parsDer, float stepSize, float momentum, float regularization)
{
    uint gid = get_global_id(0);
    
    pars[gid] -= (parsDer[gid] + 2.0f * regularization * pars[gid]) * stepSize;
    parsDer[gid] *= momentum;
}
        """
        if self.layerType == DNNLayerType.INPUT:
            self.forwardCode = None
            self.backwardCode = None
        if self.layerType == DNNLayerType.CONVO:
            self.forwardCode = DNNLayer.forwardBaseCode + """
    int x4 = ngid % o4;
    ngid /= o4;
    int x3 = ngid % o3;
    ngid /= o3;
    int x2 = ngid % o2;
    ngid /= o2;
    int x1 = ngid;

    float res = 0.0f;
    for(int q1=0;q1<i2;q1++)
    {
        for(int q2=0;q2<i3-o3+1;q2++)
        {
            for(int q3=0;q3<i4-o4+1;q3++)
            {
                res += input[x1 * i2 * i3 * i4 + q1 * i3 * i4 + (x3 + q2) * i4 + (x4 + q3)] * pars[q1 * (i3-o3+1) * (i4-o4+1) * o2 + q2 * (i4-o4+1) * o2 + q3 * o2 + x2];
            }
        }
    }

    output[gid] = res;
}
            """
            self.backwardCode = None
            self.backwardInputCode = DNNLayer.backwardBaseCode + """
    int x4 = ngid % i4;
    ngid /= i4;
    int x3 = ngid % i3;
    ngid /= i3;
    int x2 = ngid % i2;
    ngid /= i2;
    int x1 = ngid;

    float res = 0.0f;
    for(int q1=0;q1<o2;q1++)
    {
        for(int q2=0;q2<i3-o3+1;q2++)
        {
            for(int q3=0;q3<i4-o4+1;q3++)
            {
                if((x3 - q2)>=0 && (x3 - q2)<o3 && (x4 - q3)>=0 && (x4 - q3)<o4)
                {
                    res += outputDer[x1 * o2 * o3 * o4 + q1 * o3 * o4 + (x3 - q2) * o4 + (x4 - q3)] * pars[x2 * (i3-o3+1) * (i4-o4+1) * o2 + q2 * (i4-o4+1) * o2 + q3 * o2 + q1];
                }
            }
        }
    }

    inputDer[gid] += res;
}            
            """
            self.backwardParsCode = DNNLayer.backwardBaseCode + """
    int x4 = ngid % o2;
    ngid /= o2;
    int x3 = ngid % (i4-o4+1);
    ngid /= (i4-o4+1);
    int x2 = ngid % (i3-o3+1);
    ngid /= (i3-o3+1);
    int x1 = ngid;

    float res = 0.0f;
    for(int q1=0;q1<i1;q1++)
    {
        for(int q2=0;q2<o3;q2++)
        {
            for(int q3=0;q3<o4;q3++)
            {
                res += outputDer[q1 * o2 * o3 * o4 + x4 * o3 * o4 + q2 * o4 + q3] * input[q1 * i2 * i3 * i4 + x1 * i3 * i4 + (q2 + x2) * i4 + (q3 + x3)];
            }
        }
    }

    parsDer[gid] = res;
}   
            """
        elif self.layerType == DNNLayerType.RELU:
            self.forwardCode = DNNLayer.forwardBaseCode + """ 
    if(input[gid] > 0.0f)
    {
        output[gid] = input[gid];
    }
    else
    {
        output[gid] = 0.0f;
    }
}            
            """
            self.backwardCode = DNNLayer.backwardBaseCode + """
    if(input[gid] > 0.0f)
    {
        inputDer[gid] += outputDer[gid];
    }
}     
            """
        elif self.layerType == DNNLayerType.FRELU:
            self.forwardCode = DNNLayer.forwardBaseCode + """ 
    if(input[gid] > 0.0f)
    {
        output[gid] = input[gid];
    }
    else
    {
        output[gid] = 0.01f * input[gid];
    }
}            
            """
            self.backwardCode = DNNLayer.backwardBaseCode + """
    if(input[gid] > 0.0f)
    {
        inputDer[gid] += outputDer[gid];
    }
    else
    {
        inputDer[gid] += 0.01f * outputDer[gid];
    }
}     
            """
        elif self.layerType == DNNLayerType.FFRELU:
            self.forwardCode = DNNLayer.forwardBaseCode + """ 
    output[gid] = 0.01f * input[gid] + 0.99f * log(1.0f + exp(input[gid]));
}            
            """
            self.backwardCode = DNNLayer.backwardBaseCode + """
    inputDer[gid] += outputDer[gid] * (0.01f + 0.99f / (1.0f + exp(-input[gid])));
}     
            """
        elif self.layerType == DNNLayerType.FFFRELU:
            self.forwardCode = DNNLayer.forwardBaseCode + """ 
    output[gid] = 0.5f * input[gid] + 0.5f * sqrt(1.0f + input[gid] * input[gid]);
}            
            """
            self.backwardCode = DNNLayer.backwardBaseCode + """
    inputDer[gid] += outputDer[gid] * (0.5f + 0.5f * input[gid] / sqrt(1.0f + input[gid] * input[gid]));
}     
            """
        elif self.layerType == DNNLayerType.SUM:
            self.forwardCode = DNNLayer.forwardBaseCode + """ 
    output[gid] = input[gid] + input2[gid];
}            
            """
            self.backwardCode = DNNLayer.backwardBaseCode + """
    inputDer[gid] += outputDer[gid];
    inputDer2[gid] += outputDer[gid];
}     
            """            
        elif self.layerType == DNNLayerType.PREPROCESS:
            self.parsDer = None
            self.forwardCode = """
void transf(float x, float y, float *xoo, float *yoo, float p0, float p1, float p2, float p3, float p4, int o3, int o4)
{
    float xx = (x / (o3+0.0f)) * 2.0f - 1.0f;
    float yy = (y / (o4+0.0f)) * 2.0f - 1.0f;

    float xo = xx * cos(p0) - yy * sin(p0);
    float yo = xx * sin(p0) + yy * cos(p0);

    xo *= p3;
    yo *= p3;

    xo += p1;
    yo += p2;

    if(p4 < 0.0f)
    {
        yo *= -1.0f;
    }

    xo *= 0.5f;
    xo += 0.5f;
    yo *= 0.5f;
    yo += 0.5f;
    xo *= o3;
    yo *= o4;

    *xoo = xo;
    *yoo = yo;
}
            """ + DNNLayer.forwardBaseCode + """ 
    if(isTrain != 1)
    {
        output[gid] = input[gid];
    }
    else
    {
        int x4 = ngid % o4;
        ngid /= o4;
        int x3 = ngid % o3;
        ngid /= o3;
        int x2 = ngid % o2;
        ngid /= o2;
        int x1 = ngid;

        float sum = 0.0f;
        int count = 0;

        float xo, yo;

        for(int xx=0;xx<11;xx++)
        {
            for(int yy=0;yy<11;yy++)
            {
                transf(x3 + 0.05f + 0.9f * (xx + 0.0f) / 10.0f, x4 + 0.05f + 0.9f * (yy + 0.0f) / 10.0f, &xo, &yo, pars[x1 * 5 + 0], pars[x1 * 5 + 1], pars[x1 * 5 + 2], pars[x1 * 5 + 3], pars[x1 * 5 + 4], o3, o4);
                int xxxx = (int)xo;
                int yyyy = (int)yo;

                if(xxxx<0)
                {
                    xxxx = 0;
                }
                if(xxxx>=o3)
                {
                    xxxx = o3-1;
                }
                if(yyyy<0)
                {
                    yyyy = 0;
                }
                if(yyyy>=o4)
                {
                    yyyy = o4-1;
                }                
                count++;
                sum += input[x1 * o2 * o3 * o4 + x2 * o3 * o4 + xxxx * o4 + yyyy];
            }
        }

        if(count==0)
        {
            output[gid] = 0.0f;
        }
        else
        {
            output[gid] = sum / (0.0f + count);
        }
    }
}            
            """
            self.backwardCode = None
        elif self.layerType == DNNLayerType.NOISE:
            self.parsDer = None
            self.forwardCode = DNNLayer.forwardBaseCode + """ 
    if(isTrain == 1)
    {
        output[gid] = input[gid] + pars[gid];
    }
    else
    {
        output[gid] = input[gid];
    }
}            
            """
            self.backwardCode = DNNLayer.backwardBaseCode + """
    inputDer[gid] += outputDer[gid];
}     
            """            
        elif self.layerType == DNNLayerType.DROPOUT:
            self.parsDer = None
            self.forwardCode = DNNLayer.forwardBaseCode + """ 
    if(pars[gid] > extraPar || isTrain != 1)
    {
        output[gid] = input[gid];
    }
    else
    {
        output[gid] = 0.0f;
    }
}            
            """
            self.backwardCode = DNNLayer.backwardBaseCode + """
    if(pars[gid] > extraPar)
    {
        inputDer[gid] += outputDer[gid];
    }
}     
            """            
        elif self.layerType == DNNLayerType.AVER:
            self.forwardCode = DNNLayer.forwardBaseCode + """ 
    int x4 = ngid % o4;
    ngid /= o4;
    int x3 = ngid % o3;
    ngid /= o3;
    int x2 = ngid % o2;
    ngid /= o2;
    int x1 = ngid;

    int s1 = x3 * i3 / o3;
    int e1 = (x3 + 1) * i3 / o3;
    int s2 = x4 * i4 / o4;
    int e2 = (x4 + 1) * i4 / o4;

    float res = 0.0f;
    for(int q1=s1;q1<e1;q1++)
    {
        for(int q2=s2;q2<e2;q2++)
        {
                res += input[x1 * i2 * i3 * i4 + x2 * i3 * i4 + q1 * i4 + q2];
        }
    }

    output[gid] = res / (0.0f + (e1-s1) * (e2-s2));
}            
            """
            self.backwardCode = DNNLayer.backwardBaseCode + """
    int x4 = ngid % o4;
    ngid /= o4;
    int x3 = ngid % o3;
    ngid /= o3;
    int x2 = ngid % o2;
    ngid /= o2;
    int x1 = ngid;

    int s1 = x3 * i3 / o3;
    int e1 = (x3 + 1) * i3 / o3;
    int s2 = x4 * i4 / o4;
    int e2 = (x4 + 1) * i4 / o4;

    for(int q1=s1;q1<e1;q1++)
    {
        for(int q2=s2;q2<e2;q2++)
        {
                inputDer[x1 * i2 * i3 * i4 + x2 * i3 * i4 + q1 * i4 + q2] += outputDer[gid] / (0.0f + (e1-s1) * (e2-s2));
        }
    }
}
            """
        elif self.layerType == DNNLayerType.MAX:
            self.forwardCode = DNNLayer.forwardBaseCode + """ 
    int x4 = ngid % o4;
    ngid /= o4;
    int x3 = ngid % o3;
    ngid /= o3;
    int x2 = ngid % o2;
    ngid /= o2;
    int x1 = ngid;

    int s1 = x3 * i3 / o3;
    int e1 = (x3 + 1) * i3 / o3;
    int s2 = x4 * i4 / o4;
    int e2 = (x4 + 1) * i4 / o4;

    float res = input[x1 * i2 * i3 * i4 + x2 * i3 * i4 + s1 * i4 + s2];
    for(int q1=s1;q1<e1;q1++)
    {
        for(int q2=s2;q2<e2;q2++)
        {
            if(res < input[x1 * i2 * i3 * i4 + x2 * i3 * i4 + q1 * i4 + q2])
            {
                res = input[x1 * i2 * i3 * i4 + x2 * i3 * i4 + q1 * i4 + q2];
            }
        }
    }

    output[gid] = res;
}            
            """
            self.backwardCode = DNNLayer.backwardBaseCode + """
    int x4 = ngid % o4;
    ngid /= o4;
    int x3 = ngid % o3;
    ngid /= o3;
    int x2 = ngid % o2;
    ngid /= o2;
    int x1 = ngid;

    int s1 = x3 * i3 / o3;
    int e1 = (x3 + 1) * i3 / o3;
    int s2 = x4 * i4 / o4;
    int e2 = (x4 + 1) * i4 / o4;

    int w1 = s1;
    int w2 = s2;
    float res = input[x1 * i2 * i3 * i4 + x2 * i3 * i4 + s1 * i4 + s2];
    for(int q1=s1;q1<e1;q1++)
    {
        for(int q2=s2;q2<e2;q2++)
        {
            if(res < input[x1 * i2 * i3 * i4 + x2 * i3 * i4 + q1 * i4 + q2])
            {
                res = input[x1 * i2 * i3 * i4 + x2 * i3 * i4 + q1 * i4 + q2];
                w1 = q1;
                w2 = q2;
            }
        }
    }

    inputDer[x1 * i2 * i3 * i4 + x2 * i3 * i4 + w1 * i4 + w2] += outputDer[gid];
}
            """
        elif self.layerType == DNNLayerType.MATRIX:
            self.forwardCode = DNNLayer.forwardBaseCode + """ 
    int x2 = ngid % o2;
    ngid /= o2;
    int x1 = ngid;

    float res = pars[x2 * (i2 * i3 * i4 + 1)];
    for(int q1=0;q1<i2 * i3 * i4;q1++)
    {
        res += input[x1 * i2 * i3 * i4 + q1] * pars[x2 * (i2 * i3 * i4 + 1) + q1 + 1];
    }

    output[gid] = (float)res;
}            
            """
            self.backwardInputCode = DNNLayer.backwardBaseCode + """
    int x2 = ngid % (i2 * i3 * i4);
    ngid /= (i2 * i3 * i4);
    int x1 = ngid;

    float res = 0.0f;
    for(int q1=0;q1<o2;q1++)
    {
        res += outputDer[x1 * o2 + q1] * pars[q1 * (i2 * i3 * i4 + 1) + x2 + 1];
    }

    inputDer[gid] += res;
}
            """
            self.backwardParsCode = DNNLayer.backwardBaseCode + """
    int x2 = ngid % (i2 * i3 * i4 + 1);
    ngid /= (i2 * i3 * i4 + 1);
    int x1 = ngid;

    float res = 0.0f;
    for(int q1=0;q1<i1;q1++)
    {
        if(x2==0)
        {
            res += outputDer[q1 * o2 + x1];
        }
        else
        {
            res += outputDer[q1 * o2 + x1] * input[q1 * (i2 * i3 * i4) + x2 - 1];
        }
    }

    parsDer[gid] = res;
}
            """
        elif self.layerType == DNNLayerType.SOFTMAX:
            self.forwardCode = DNNLayer.forwardBaseCode + """ 
    int x2 = ngid % o2;
    ngid /= o2;
    int x1 = ngid;

    float max = -FLT_MAX;
    for(int q1=0;q1<i2;q1++)
    {
        if(max < input[x1 * i2 + q1])
        {
            max = input[x1 * i2 + q1];
        }
    }

    float sum = 0.0f;
    for(int q1=0;q1<i2;q1++)
    {
        sum += exp(input[x1 * i2 + q1] - max);
    }

    output[gid] = exp(input[gid] - max) / sum;
}            
            """
            self.backwardCode = DNNLayer.backwardBaseCode + """
    int x2 = ngid % o2;
    ngid /= o2;
    int x1 = ngid;

    inputDer[gid] += outputDer[gid] * output[gid] * (1.0f - output[gid]);
}
            """            
        elif self.layerType == DNNLayerType.SIGMOID:
            self.forwardCode = DNNLayer.forwardBaseCode + """ 
    output[gid] = 1.0f / (1.0f + exp(input[gid]));
}               
            """
            self.backwardCode = DNNLayer.backwardBaseCode + """
    inputDer[gid] += outputDer[gid] * output[gid] * (output[gid] - 1.0f);
}            
            """
        elif self.layerType == DNNLayerType.FINAL:
            self.forwardCode = DNNLayer.forwardBaseCode + """ 
    float res = 0.0f;
    for(int q1=0;q1<i2;q1++)
    {
        if(pars[gid * i2 + q1] > 0.5f)
        {
            res += -log(0.0000001f + max(0.0f, input[gid * i2 + q1]));
        }
        else
        {
            res += -log(1.0000001f - min(1.0f, input[gid * i2 + q1]));
        }
    }

    output[gid] = res;
}                  
            """
            self.backwardCode = DNNLayer.backwardBaseCode + """
    for(int q1=0;q1<i2;q1++)
    {
        if(pars[gid * i2 + q1] > 0.5f)
        {
            inputDer[gid * i2 + q1] += -1.0f / (0.0000001f + max(0.0f, input[gid * i2 + q1]));
        }
        else
        {
            inputDer[gid * i2 + q1] += 1.0f / (1.0000001f - min(1.0f, input[gid * i2 + q1]));
        }
    }
}
            """
            self.parsDer = None
