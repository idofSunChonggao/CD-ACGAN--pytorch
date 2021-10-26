import math
import numpy as np
from csv_io import read_from_csv, write2csv
import random
import matplotlib.pyplot as plt
import seaborn

#1. Normalization 
def normalization(ipds):
    M = max(ipds)
    m = min(ipds)
    data = []
    for x in ipds:
        x_ = ((x - M) + (x - m)) / (M - m)
        data.append(x_)
    return data
#2. Polar coordinates
def polarEncode(data):
    data_polar = []
    for s in data:
        theta = math.acos(s)
        # print(math.cos(theta * 2))
        data_polar.append(theta)
    return data_polar

# 3. GAF images generation
def caculateGAF(data_polar):
    n = len(data_polar)
    GAF = np.zeros((n, n), dtype=np.float64)
    for i in range(n):  
        for j in range(i + 1):
            if i == j:
                GAF[i][j] = math.cos(data_polar[i] * 2)
                continue
            theta_i = data_polar[i]
            theta_j = data_polar[j]
            G = math.cos(theta_i + theta_j)
            GAF[i][j] = G
            GAF[j][i] = G
    return GAF
    



# Output dataset for CD-ACGAN
def generateTrainSet(file, gaf_size=200, amount=10000):
    # Please fill in the original IPDs path
    normal_ipds = read_from_csv('')
    ipctc_ipds = read_from_csv('')
    tr_ipds = read_from_csv('')
    jitterbug_ipds = read_from_csv('')
    ln_ipds = read_from_csv('')
    print("GAF images dataset is generating, amount:{}, size:{}".format(amount, gaf_size))
    train_amount = amount // 50
    for i in range(10):
        print('This is the {} time generation'.format(i + 1))
        dataset = normalDataSet(normal_ipds, gaf_size, train_amount)
        print("normal has finished")
        ipctc_dataset = IPCTCDataSet(ipctc_ipds, gaf_size, train_amount)
        dataset = np.append(dataset, ipctc_dataset, axis=0)
        print("ipctc has finished")
        trctc_dataset = TRDataSet(tr_ipds, gaf_size, train_amount)
        dataset = np.append(dataset, trctc_dataset, axis=0)
        print("trctc has finished")
        jitterbug_dataset = JitterBugDataSet(jitterbug_ipds, gaf_size, train_amount)
        dataset = np.append(dataset, jitterbug_dataset, axis=0)
        print("jitterbug has finished")
        ln_dataset = L2N1DataSet(ln_ipds, gaf_size, train_amount)
        dataset = np.append(dataset, ln_dataset, axis=0)
        print("ln has finished")
        write2csv(file, dataset)
        print("{} GAF images have been generated, {} in total".format((i + 1) * 5 * train_amount, amount))
        

def normalDataSet(ipds, size, amount):
    dataset = np.zeros((amount, 1 + size * size))
    for i in range(amount):
        start = random.randint(100, 10000) * 24
        ipds_ = cutIpdsRandom(ipds, start, size)
        if len(ipds_) < size:
            continue
        ipds_ = ipds_[0:size]
        data = normalization(ipds_)
        data_polar = polarEncode(data)
        GAF = caculateGAF(data_polar)
        dataset[i][0] = 0
        dataset[i, 1:] = GAF.flatten()
    return dataset


def IPCTCDataSet(ipds, size, amount):
    dataset = np.zeros((amount, 1 + size * size))
    for i in range(amount):
        start = random.randint(0, 10000) * 24
        ipds_ = cutIpdsRandom(ipds, start, size)
        if len(ipds_) < size:
            continue
        ipds_ = ipds_[0:size]  
        data = normalization(ipds_)
        data_polar = polarEncode(data)
        GAF = caculateGAF(data_polar)
        dataset[i][0] = 1
        dataset[i, 1:] = GAF.flatten()
    return dataset


def TRDataSet(ipds, size, amount):
    dataset = np.zeros((amount, 1 + size * size))
    for i in range(amount):
        start = random.randint(0, 10000) * 50
        ipds_ = cutIpdsRandom(ipds, start, size)
        if len(ipds_) < size:
            continue
        ipds_ = ipds_[0:size]  
        data = normalization(ipds_)
        data_polar = polarEncode(data)
        GAF = caculateGAF(data_polar)
        dataset[i][0] = 2
        dataset[i, 1:] = GAF.flatten()
    return dataset


def JitterBugDataSet(ipds, size, amount):
    dataset = np.zeros((amount, 1 + size * size))
    for i in range(amount):
        start = random.randint(0, 10000) * 30
        end = start + size
        ipds_ = cutIpdsRandom(ipds, start, size)
        if len(ipds_) < size:
            continue
        ipds_ = ipds_[0:size]  
        data = normalization(ipds_)
        data_polar = polarEncode(data)
        GAF = caculateGAF(data_polar)
        dataset[i][0] = 3
        dataset[i, 1:] = GAF.flatten()
    return dataset


def L2N1DataSet(ipds, size, amount):
    dataset = np.zeros((amount, 1 + size * size))
    for i in range(amount):
        start = random.randint(0, 10000) * 40
        ipds_ = cutIpdsRandom(ipds, start, size)
        if len(ipds_) < size:
            continue
        ipds_ = ipds_[0:size] 
        data = normalization(ipds_)
        data_polar = polarEncode(data)
        GAF = caculateGAF(data_polar)
        dataset[i][0] = 4
        dataset[i, 1:] = GAF.flatten()
    return dataset


def num2ColorRGB(num):
    colors = ["darkblue", "navy", "blue", "darkgreen", "green", "yellow", "gold", "orange", "tomato",
              "red"]
    steps = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for i in range(10):
        if steps[i] <= num <= steps[i + 1]:
            color = colors[i]
            h = seaborn.xkcd_rgb[color]
            r = int(h[1:3], 16)
            g = int(h[3:5], 16)
            b = int(h[5:7], 16)
            return r, g, b
    return 0, 0, 0

def cutIpdsRandom(ipds, start, size):
    end = start + size
    ipds_ = ipds[start:end]
    return ipds_
    
def visualGAF(GAF):
    n = len(GAF[0])
    image = np.zeros((n, n, 3), np.uint8)
    for i in range(n):  
        for j in range(n):  
            rgb = num2ColorRGB(GAF[i][j])
            image[i][j][0] = rgb[0]
            image[i][j][1] = rgb[1]
            image[i][j][2] = rgb[2]
    # colorbar
    cm = plt.cm.get_cmap('RdYlBu_r')
    x = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    y = x
    sc = plt.scatter(x, y, c=x, vmin=-1.0, vmax=1.0, cmap=cm)
    plt.colorbar(sc)
    plt.axis('off')
    plt.imshow(image)




