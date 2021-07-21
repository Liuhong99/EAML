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
import  argparse
import  os
from    model import *
from    utils import *
from    lip_reg import *


def main(args: argparse.Namespace):

    balance_in = args.balance_in
    balance_out = args.balance_out
    inner_iter = args.inner_iter
    inner_lr = args.lr_in
    outer_lr = args.lr_out
    gpu_ind = args.gpuid
    domain_length = args.domain_length
    print_freq = args.print_freq
    rootdir = args.root
    num_epoch = args.epochs
    batch_size = args.batch_size
    lip_balance = args.lip_balance
    jth = args.lip_jth
    save_dir = args.save
    train_label_path = args.train_label_path

    setGPU(gpu_ind)

    imgDB = imageDB(rootdir,60,3)
    labelvec = np.load(train_label_path)

    ds = datasets.MNIST('./', train=True, transform=transforms.Compose([
                                transforms.Resize((28,28)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ]), target_transform=None, download=True)

    train_loader = DataLoader(
            ds, batch_size = batch_size * (domain_length + 1), shuffle = True,
            num_workers = 2, pin_memory = True,)



    config = [
            ('conv2d', [20, 1, 5, 5, 1, 0]),
            ('max_pool2d', [2, 2, 0]),
            ('relu', [True]),
            ('conv2d', [50, 20, 5, 5, 1, 0]),
            ('max_pool2d', [2, 2, 0]),
            ('relu', [True]),
            ('flatten',[]),
            
        ]

    config1 = [
            ('linear', [500,800]),
            ('relu', [True]),
            ('linear', [10,500]),
        ]


    feature_extractor = Learner(config, 1,84).cuda()
    cls = Learner(config1, 1,84).cuda()

    optimizer = torch.optim.SGD(feature_extractor.parameters(), lr=outer_lr, weight_decay=0.0005 ,momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    tt = get_mini_batches(imgDB, labelvec = labelvec, domain_num = 20, domain_length = domain_length, s_size = batch_size, q_size = batch_size, len_epoch = 500, channel = 1, imgsize = 28)





    epoch = 0
    step = 0
    lr_r = outer_lr
    while epoch < num_epoch:
        if epoch > int(num_epoch / 2) & epoch <= int(num_epoch / 4 * 3):
            lr_r = 0.1 * outer_lr
        if epoch > int(num_epoch / 4 * 3):
            lr_r = 0.01 * outer_lr

        for g in optimizer.param_groups:
            g['lr'] = lr_r
            
        for (i, ((xspt, yspt, xqry, yqry),(im_s0,lb_s0)))in enumerate(zip(tt,train_loader)):
            if im_s0.shape[0] != batch_size * (domain_length + 1):
                continue
            losslst = []
            im_s = im_s0[domain_length * batch_size:domain_length * batch_size + batch_size].cuda()
            lb_s = lb_s0[domain_length * batch_size:domain_length * batch_size + batch_size].cuda()
            fss = feature_extractor(im_s)
            logits = cls(fss)

            reg_loss = 0
            
            if lip_balance != 0:
                margins = compute_margins(logits, lb_s)
                norm_sq_dict = get_grad_hl_norms({'ft':fss}, torch.mean(margins), cls, create_graph=True, only_inputs=True)
                
                
                for val in norm_sq_dict.values():
                    j = val[1]
                    j_ind = j > jth
                    if torch.sum(j_ind) > 0:
                        reg_loss += torch.mean(j[j_ind])

            for domain_ind in range(domain_length):
                im_source1q = torch.from_numpy(xqry[domain_ind].astype('float32')).cuda()
                label_source1q = torch.from_numpy(yqry[domain_ind].astype('int64')).cuda()
                logitt = cls(feature_extractor(im_source1q))
                loss1 = DAN(logits,logitt)
                losslst.append(loss1)
        
            dloss = sum(losslst) / domain_length
            ce = F.cross_entropy(logits, lb_s)
            
            if lip_balance == 0:
                loss =  balance_out * dloss + ce
            else:
                loss =  balance_out * dloss + ce + lip_balance * reg_loss
            
            step += 1
            if step % print_freq ==0:
                if reg_loss == 0:
                    print(epoch,ce.data.cpu().numpy(),dloss.data.cpu().numpy())
                else:
                    print(epoch,ce.data.cpu().numpy(),dloss.data.cpu().numpy(),reg_loss.data.cpu().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tt = get_mini_batches(imgDB, labelvec, domain_num = 20, domain_length = domain_length, s_size = batch_size, q_size = batch_size, len_epoch = 500, channel = 1, imgsize = 28)
        epoch += 1
        torch.save(feature_extractor.state_dict(), save_dir)


    torch.save(feature_extractor.state_dict(), save_dir)

if __name__ == '__main__':
    dataset_names = sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )

    parser = argparse.ArgumentParser(description='EAML')
    parser.add_argument('--root', metavar='DIR',
                        help='root path of rotated dataset')

    parser.add_argument('--epochs', default=500, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr-in',  default=3e-2, type=float,
                        metavar='LR', help='initial learning rate in the inner loop')
    parser.add_argument('--lr-out', default=3e-3, type=float,
                        metavar='LR', help='initial learning rate in the outer loop')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--balance-in', default=0.5, type=float,
                        help='the trade-off hyper-parameter for transfer loss in the inner loop')
    parser.add_argument('--balance-out', default=1.0, type=float,
                        help='the trade-off hyper-parameter for transfer loss in the outer loop')
    parser.add_argument('--gpuid', default=0, type=int,
                        help='GPU to be used')
    parser.add_argument('--inner-iter', default=3, type=int,
                        help='iteration of each inner loop')
    parser.add_argument('--domain-length', default=10, type=int,
                        help='length of target trajectories')
    parser.add_argument('--lip-balance', default = 0.2, type=float,
                        help='balance of regularization')
    parser.add_argument('--lip-jth', default = 0.01, type=float,
                        help='thresh of regularization')
    parser.add_argument('--save', default = 'JAN.pth', type=str,
                        help='save path')
    parser.add_argument('--train-label-path', default = 'mnist_train_label.npy', type=str,
                        help='train label path')
    
    args = parser.parse_args()
    print(args)
    main(args)