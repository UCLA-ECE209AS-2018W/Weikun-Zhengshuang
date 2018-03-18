# over-the-air_speech_recogniztion_attack
--------------------------------------------------------------------------------

## Introduction
The course project for 209AS - Special Topics in Circuits and Embedded Systems: 
Security and Privacy for Embedded Systems, Cyber-Physical Systems, and Internet 
of Things.

The project uses deep neural networks (U-Net) to build a deep learning model 
that remove electronic noise and air noise to transmit adversarial examples 
over-the-air. Our contribution is to make the adversarial example attack to be 
a practical attack. In the further, this deep neural networks (U-Net) can also 
use to defense adversarial learning attack as well due to its strong noise 
remove ability.

For more problem details, please go to
[project website](https://weikunhan.github.io).

## Requirements and Dependencies
The following packages are required (the version numbers that have been tested 
are given for reference):

* Python 2.7 or 3.6
* Tensorflow 1.0.1
* Numpy 1.12.1
* Librosa 0.5.0
* tqdm 4.11.2 (only for preprocessing datasets)
* Sox 1.2.7 (only for preprocessing datasets)
