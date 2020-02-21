# Soft-Actor-Critic-and-Extensions
PyTorch implementation of **Soft-Actor-Critic** and **PER** + **ERE**
_____________
This repository includes the newest Soft-Actor-Critic version ([Paper 2019](https://arxiv.org/abs/1812.05905)) as well as extensions for SAC:
- **P**rioritized **E**xperience **R**eplay ([PER](https://arxiv.org/abs/1511.05952))
- **E**mphasizing **R**ecent **E**xperience without Forgetting the Past([ERE](https://arxiv.org/abs/1906.04009))

In implementation of ERE the authors used and older version of SAC, whereas this repository contains the newest version of SAC as well as a Proportional Prioritization implementation of PER. 

## How to use:

*Run regular SAC:* `python SAC.py -env Pendulum-v0 -ep 200`

*Run SAC + PER:* `python SAC_PER.py -env Pendulum-v0 -ep 200`

*Run SAC + ERE + PER:* `python SAC_ERE_PER.py -env Pendulum-v0 -ep`

For further input arguments and hyperparameter check the code.


## Results 
It can be seen that the extensions not always bring improvements to the algorithm. This is depending on the environment and from environment to environment different - as the authors mention in their paper (ERE).

![Pendulum](imgs/SAC_PENDULUM.jpg)

![LLC](imgs/SAC_LLC.jpg)

- All runs without hyperparameter-tuning

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
