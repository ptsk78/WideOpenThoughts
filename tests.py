import numpy as np
from DNNLayer import DNNLayerType, DNNLayer

def relErr(f1, f2):
    dif = np.median(np.abs(f1 - f2)) / np.median(np.maximum(np.abs(f1),np.abs(f2)))
    
    return dif

def convo_forward(inp, pars):
    ret = np.zeros((inp.shape[0], pars.shape[3], inp.shape[2] - pars.shape[1] + 1, inp.shape[3] - pars.shape[2] + 1)).astype(np.float128)
    for i1 in range(ret.shape[0]):
        for i2 in range(ret.shape[1]):
            for i3 in range(ret.shape[2]):
                for i4 in range(ret.shape[3]):
                    for j1 in range(inp.shape[1]):
                        for j2 in range(pars.shape[1]):
                            for j3 in range(pars.shape[2]):
                                ret[i1,i2,i3,i4] += np.float128(pars[j1, j2, j3, i2]) * np.float128(inp[i1, j1, i3 + j2, i4 + j3])

    return ret

def convo_backward(inp, pars, outDer):
    ret = np.zeros((inp.shape[0], pars.shape[3], inp.shape[2] - pars.shape[1] + 1, inp.shape[3] - pars.shape[2] + 1)).astype(np.float128)
    retInpDer = np.zeros(inp.shape).astype(np.float128)
    retParDer = np.zeros(pars.shape).astype(np.float128)
    for i1 in range(ret.shape[0]):
        for i2 in range(ret.shape[1]):
            for i3 in range(ret.shape[2]):
                for i4 in range(ret.shape[3]):
                    for j1 in range(inp.shape[1]):
                        for j2 in range(pars.shape[1]):
                            for j3 in range(pars.shape[2]):
                                retParDer[j1, j2, j3, i2] += np.float128(outDer[i1,i2,i3,i4]) * np.float128(inp[i1, j1, i3 + j2, i4 + j3])
                                retInpDer[i1, j1, i3 + j2, i4 + j3] += np.float128(outDer[i1,i2,i3,i4]) * np.float128(pars[j1, j2, j3, i2])

    return retInpDer, retParDer

def matrix_forward(inp, pars):
    ret = np.zeros((inp.shape[0], pars.shape[0], 1, 1)).astype(np.float128)
    tmpinp = inp.reshape(inp.shape[0], -1)
    for i1 in range(inp.shape[0]):
        for i2 in range(pars.shape[0]):
            ret[i1, i2] += np.float128(pars[i2, 0, 0, 0])
            for i3 in range(tmpinp.shape[1]):
                ret[i1, i2] += np.float128(pars[i2, 1+i3, 0, 0]) * np.float128(tmpinp[i1, i3])

    return ret

def matrix_backward(inp, pars, outDer):
    ret = np.zeros((inp.shape[0], pars.shape[0], 1, 1)).astype(np.float128)
    retParDer = np.zeros(pars.shape).astype(np.float128)
    tmpinp = inp.reshape(inp.shape[0], -1)
    retInpDer = np.zeros(tmpinp.shape).astype(np.float128)
    for i1 in range(inp.shape[0]):
        for i2 in range(pars.shape[0]):
            retParDer[i2, 0, 0, 0] += np.float128(outDer[i1, i2])
            for i3 in range(tmpinp.shape[1]):
                retParDer[i2, 1+i3, 0, 0] += np.float128(outDer[i1, i2]) * np.float128(tmpinp[i1, i3])
                retInpDer[i1, i3] += np.float128(outDer[i1, i2]) * np.float128(pars[i2, 1+i3, 0, 0])
    retInpDer = retInpDer.reshape(inp.shape)
    
    return retInpDer, retParDer

def test_layer(layerType):
    d1 = 10
    d2 = 5
    d3 = 15
    d4 = 13
    l1 = DNNLayer(DNNLayerType.MAX, (d1, d2, d3, d4), None, None)
    if layerType == DNNLayerType.CONVO:
        l2 = DNNLayer(DNNLayerType.CONVO, (d1, 20, d3-4, d4-1), (d2, 5, 2, 20), l1)
    elif layerType == DNNLayerType.MATRIX:
        l2 = DNNLayer(DNNLayerType.MATRIX, (d1, 17, 1, 1), (17, d2 * d3 * d4 + 1, 1, 1), l1)
    else:
        return False        

    l1.output.mem = np.random.rand(*l1.output.mem.shape).astype(np.float32)
    l1.output.copyToGPU()
    l2.forward(True)
    l2.output.copyToCPU()

    if layerType == DNNLayerType.CONVO:
        output2 = convo_forward(l1.output.mem, l2.pars.mem)
    elif layerType == DNNLayerType.MATRIX:
        output2 = matrix_forward(l1.output.mem, l2.pars.mem)
    else:
        return False         

    print('   Dif {0} forward: {1}'.format(layerType, relErr(output2, l2.output.mem)))
    print('      Output [{0},{1}] [{2},{3}]]'.format(np.min(output2), np.max(output2), np.min(l2.output.mem), np.max(l2.output.mem)))
    if relErr(output2, l2.output.mem) > 1.e-6:
        return False

    l2.outputDer.mem = np.random.rand(*l2.outputDer.mem.shape).astype(np.float32)
    l2.outputDer.copyToGPU()
    l2.backward()
    l2.parsDer.copyToCPU()
    l1.outputDer.copyToCPU()

    if layerType == DNNLayerType.CONVO:
        inpDer2, parDer2 = convo_backward(l1.output.mem, l2.pars.mem, l2.outputDer.mem)
    elif layerType == DNNLayerType.MATRIX:
        inpDer2, parDer2 = matrix_backward(l1.output.mem, l2.pars.mem, l2.outputDer.mem)
    else:
        return False         

    print('   Dif {0} backward inp: {1}'.format(layerType, relErr(inpDer2, l1.outputDer.mem)))
    print('   Dif {0} backward par: {1}'.format(layerType, relErr(parDer2, l2.parsDer.mem)))
    print('      InputDer [{0},{1}] [{2},{3}]]'.format(np.min(inpDer2), np.max(inpDer2), np.min(l1.outputDer.mem), np.max(l1.outputDer.mem)))
    print('      ParsDer [{0},{1}] [{2},{3}]]'.format(np.min(parDer2), np.max(parDer2), np.min(l2.parsDer.mem), np.max(l2.parsDer.mem)))

    if relErr(inpDer2, l1.outputDer.mem) > 1.e-6:
        return False
    if relErr(parDer2, l2.parsDer.mem) > 1.e-6:
        return False

    return True

tests = [DNNLayerType.CONVO, DNNLayerType.MATRIX]#, DNNLayerType.MAX]
def run_tests():

    for lt in tests:
        if not test_layer(lt):
            print('Failed {0}'.format(lt))
            return
        else:
            print('Passed {0}'.format(lt))

    print('All tests PASSED')

run_tests()
