# EAML
Code release for "Learning to Adapt to Evolving Domains" (NeurIPS 2020)

## Prerequisites
- PyTorch >= 0.4.0 (with suitable CUDA and CuDNN version)
- torchvision >= 0.2.1
- Python3
- Numpy
- argparse
- PIL

## Training
```
Rotated MNIST

%run eaml.py /workspace/liuhong/rot_mnist_28/ --lip-balance 0.2 --lip-jth 0.01 --epochs 1000 --lr-in 0.03 --lr-out 0.003 --save 'pretrained_lip_0.04.pth'

```

## Acknowledgement
This code is implemented based on the [JAN (Joint Adaptation Networks)](https://github.com/thuml/Xlearn/blob/master/pytorch/src/loss.py) code, and it is our pleasure to acknowledge their contributions.
The meta-learning code is adapted from [https://github.com/dragen1860/MAML-Pytorch/](https://github.com/dragen1860/MAML-Pytorch/).

## Citation
If you use this code for your research, please consider citing:
```
@inproceedings{Wang19TransNorm,
    title = {Transferable Normalization: Towards Improving Transferability of Deep Neural Networks},
    author = {Wang, Ximei and Jin, Ying and Long, Mingsheng and Wang, Jianmin and Jordan, Michael I},
    booktitle = {Advances in Neural Information Processing Systems 33},
    year = {2020}
}
```

## Contact
If you have any problem about our code, feel free to contact
- h-l17@mails.tsinghua.edu.cn