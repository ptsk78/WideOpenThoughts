import os
import requests
import re
import gzip
import shutil
import tarfile

def gunzip_something(gzipped_file_name, work_dir):
    filename = os.path.split(gzipped_file_name)[-1]
    filename = re.sub(r"\.gz$", "", filename, flags=re.IGNORECASE)

    with gzip.open(gzipped_file_name, 'rb') as f_in:
        with open(os.path.join(work_dir, filename), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

def downloadFile(url):
    response = requests.get(url)
    open('./data{0}'.format(url[url.rindex('/'):]), "wb").write(response.content)
    if url.endswith('.tar.gz'):
        file = tarfile.open('./data{0}'.format(url[url.rindex('/'):]))
        file.extractall('./data/')
        file.close()
    elif url.endswith('.tar.gz'):
        shutil.unpack_archive('./data{0}'.format(url[url.rindex('/'):]), './data/mnist/', 'gz')
    else:
        shutil.copy('./data{0}'.format(url[url.rindex('/'):]), './data/ads/'.format(url[url.rindex('/'):]))

def download(dataSetName):
    if not os.path.exists('./data'):
        os.mkdir('./data')
    urls = []
    if dataSetName == 'Cifar':
        if not os.path.exists('./data/cifar-10-batches-py/'):
            urls = ['https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz']
            os.mkdir('./data/cifar-10-batches-py')
    elif dataSetName == 'MNIST':
        shutil.register_unpack_format('gz', ['.gz', ], gunzip_something)
        if not os.path.exists('./data/mnist'):
            urls = ['http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
                    'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
                    'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
                    'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz']
            os.mkdir('./data/mnist')
    elif dataSetName == 'ADS':
        if not os.path.exists('./data/ads'):
            urls = ['https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
                    'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test']
            os.mkdir('./data/ads')

    for url in urls:
        downloadFile(url)

