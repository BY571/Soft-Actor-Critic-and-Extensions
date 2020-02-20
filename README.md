# Soft-Actor-Critic-and-Extensions
PyTorch implementation of **Soft-Actor-Critic** and **PER** + **ERE**
_____________
This repository includes the newest Soft-Actor-Critic version ([Paper 2019](https://arxiv.org/abs/1812.05905)) as well as extensions for SAC:
- **P**rioritized **E**xperience **R**eplay ([PER](https://arxiv.org/abs/1511.05952))
- **E**mphasizing **R**ecent **E**xperience without Forgetting the Past([ERE](https://arxiv.org/abs/1906.04009))

In implementation of ERE the authors used and older version of SAC, whereas this repository contains the newest version of SAC as well as a Proportional Prioritization implementation of PER. 

## How to use:

## Results 

![Pendulum](imgs/SAC_PENDULUM.jpg)


## Author
- Sebastian Dittert

**Feel free to use this code for your own projects or research.**
```
@misc{SAC,
  author = {Dittert, Sebastian},
  title = {PyTorch Implementation of Soft-Actor-Critic-and-Extensions},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/BY571/Soft-Actor-Critic-and-Extensions}},
}
```
