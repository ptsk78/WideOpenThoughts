# Wide Open Thoughts

Experimental unfinished code. (if you find good DNN for CIFAR-10 dataset with this framework - test accuracy over 90%, feel free to do pull request - it would be appreciated :) - it obviously needs some tuning to do for CIFAR-10)

This framework was written in [Python](https://www.python.org/) and [PyOpenCL](https://documen.tician.de/pyopencl/) from scratch to train neural networks and to show you don't need a lot of code to start such research project from scratch.

| File  | Number of lines |
| ------------- | ------------- |
| DNN.py | 194 |
| DNNLayer.py | 814 |
| DataDownloader.py | 52 |
| DataLoader.py | 157 |
| GPUMemory.py | 27 |
| main.py | 50 |
| plots.py | 90 |
|  |  |
| Total | 1384 |

Some results on three datasets:

| DataSet  | DNN | Test Accuracy |
| ------------- | ------------- |------------- |
| [MNIST](http://yann.lecun.com/exdb/mnist/)  | DNN_MNIST_01.py | >99% |
| [ADS](https://archive.ics.uci.edu/ml/datasets/adult)  | DNN_ADS_01.py  | ~85% |
| [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)  | DNN_CIFAR10_01.py  | ~86.5% |

