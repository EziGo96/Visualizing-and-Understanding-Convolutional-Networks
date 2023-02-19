'''
Created on 17-Feb-2023

@author: EZIGO
'''
'''load mnist'''
import numpy as np
import struct

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = path+"/"+kind+"-labels.idx1-ubyte"
    images_path = path+"/"+kind+"-images.idx3-ubyte"
        
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 28,28,1)
 
    return images, labels