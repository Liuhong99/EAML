import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np
import  torchvision.datasets as datasets 
import  torchvision.models as models 
import  torchvision.transforms as transforms
import  os

def setGPU(i):
    global os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "%s"%(i)
    gpus = [x.strip() for x in (str(i)).split(',')]
    NGPU = len(gpus)
    print('gpu(s) to be used: %s'%str(gpus))
    return NGPU

def imageDB(root, domain_num, interv):
    imageDB = []
    for i in range(0,domain_num,interv):
        imageDB.append(np.load(root + 'mnist_train_img_' +str(i) + '.npy')) 
    return(imageDB)

def shuffle(X, Y):
    m = X.shape[0]
    permutation = list(np.random.permutation(m))
    X_shuffle = X[permutation]
    Y_shuffle = Y[permutation]
    X_shuffle,  Y_shuffle
    return X_shuffle,  Y_shuffle

def get_mini_batches(imgDB, labelvec ,domain_num, domain_length,s_size, q_size, len_epoch, channel, imgsize):
    mini_batches = []
    for i in range(len_epoch):
        xspt = np.zeros((domain_length,  s_size, channel, imgsize, imgsize))
        yspt = np.zeros((domain_length,  s_size))
        xqry = np.zeros((domain_length,   q_size, channel, imgsize, imgsize))
        yqry = np.zeros((domain_length,   q_size))
        
        sel_domain_ind = np.sort(np.random.choice(domain_num,domain_length,False))
        for j in range(domain_length):
            randind1 = np.random.choice(imgDB[j].shape[0],s_size+q_size,False)
            xspt0 = imgDB[sel_domain_ind[j]][randind1[0:s_size]]
            xqry0 = imgDB[sel_domain_ind[j]][randind1[s_size:s_size + q_size]]
            yspt0 = labelvec[randind1[0:s_size]]
            yqry0 = labelvec[randind1[s_size:s_size + q_size]]
            xspt[j] = xspt0
            yspt[j] = yspt0
            xqry[j] = xqry0
            yqry[j] = yqry0
        mini_batches.append((xspt,yspt,xqry,yqry))
    return mini_batches

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)


def DAN(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    bs_s = source.shape[0]
    bs_t = target.shape[0]
    
    k_mat = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    loss = (torch.sum(k_mat[0:bs_s,0:bs_s]) - torch.sum(torch.diagonal(k_mat[0:bs_s,0:bs_s]))) / bs_s / (bs_s - 1) + (torch.sum(k_mat[bs_t:bs_s+bs_t,bs_t:bs_s+bs_t]) - torch.sum(torch.diagonal(k_mat[bs_t:bs_s+bs_t,bs_t:bs_s+bs_t]))) / bs_t / (bs_t - 1) - torch.sum(k_mat[0:bs_s,bs_t:bs_s+bs_t]) / bs_s / bs_t * 2
    return loss