# EAML
Code release for "Learning to Adapt to Evolving Domains" (NeurIPS 2020)

## Prerequisites
- PyTorch >= 0.4.0 (with suitable CUDA and CuDNN version)
- torchvision >= 0.2.1
- Python3
- Numpy
- argparse
- PIL

## Dataset
Rotated MNIST: [https://drive.google.com/file/d/1eaw42sg4Cgm34790AW_SKGCSkFosugl2/view?usp=sharing](https://drive.google.com/file/d/1eaw42sg4Cgm34790AW_SKGCSkFosugl2/view?usp=sharing)


## Training
```
EAML Rotated MNIST

%run eaml.py rot_mnist_28/ --lip-balance 0.2 --lip-jth 0.01 --epochs 500 --lr-in 0.03 --lr-out 0.003 

```
```
JAN Rotated MNIST

%run JAN.py rot_mnist_28/ --lip-balance 0.2 --lip-jth 0.01 --epochs 500 --lr-in 0.03 --lr-out 0.003

```



## Acknowledgement
This code is implemented based on the [JAN (Joint Adaptation Networks)](https://github.com/thuml/Xlearn/blob/master/pytorch/src/loss.py) code, and it is our pleasure to acknowledge their contributions.
The meta-learning code is adapted from [https://github.com/dragen1860/MAML-Pytorch/](https://github.com/dragen1860/MAML-Pytorch/).

## Citation
If you use this code for your research, please consider citing:
```
@inproceedings{NEURIPS2020_fd69dbe2,
 author = {Liu, Hong and Long, Mingsheng and Wang, Jianmin and Wang, Yu},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
 pages = {22338--22348},
 publisher = {Curran Associates, Inc.},
 title = {Learning to Adapt to Evolving Domains},
 url = {https://proceedings.neurips.cc/paper/2020/file/fd69dbe29f156a7ef876a40a94f65599-Paper.pdf},
 volume = {33},
 year = {2020}
}


```

## Contact
If you have any problem about our code, feel free to contact
- h-l17@mails.tsinghua.edu.cn