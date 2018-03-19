# Over-the-Air Speech Recogniztion Attack
======

## About This Project
------
The course final project for UCLA EE209AS Winter 2018 - Special Topics in Circuits and Embedded Systems: 
Security and Privacy for Embedded Systems, Cyber-Physical Systems, and Internet of Things by Professor. Mani Srivastava.

Team members: Weikun Han, Zhengshuang Ren

## Introduction
------
Recent progress in intelligent home assitant devices such as Google Home and Amazon Alexa
is changing people's daily lives and allows users to interact with their home 
devices in a smarter and more convenient. Such devices are integrated with speech recognition 
models mostly based on Deep Learning Neural Networks to recoginize users' voice commands. 
The advantage of using Deep Learning Models is their higher accuracy on recoginizing users' commands 
correctly than traditional machine learning algorithms. However, such devices bring new security concerns 
since they are operating users' private home devices and transimitting sensitive data and information about users'
private personal lives. Vulnerabilities of these devices may be exploited and used to cause users' property loss. 


Recent research has shown that deep learning models are easy to be fooled by attackers to perform untargeted or 
even targeted attacks by generating adversarial exampes to produce wrong recognized commands and to actuate users' 
home devices in unwanted ways. Moustafa Alzantot[1] and Nicholas Carlini[2] have demonstrated the vulnerabilities 
of such speech recognition models by generating adversarial examples to perform targeted attacks with high successful 
rates. Notice in [1], the author is performing black-box attacks without knowing the details about the recognition 
neural network whereas in [2], the attack is performed in white-box attacks leveraging the sturcture and details about the network.
However, they achieved the research-purpose attacks by deploying adversarial example files into the home assistant 
devices, but in practical attacks, over-the-air attacks are more realistic to perform since the attackers may not have physical 
access to the devices. In this project, we proposed a way to generate the adversarial examples before the air channel so that 
the attackers are able to perform over-the-air attacks based on the adversarial examples they have from [1] or [2]. The main idea 
of our project is to mimic the air channel characteristics, the characteristics of the speaker used to play the adversarial 
examples and the microphone on the home assistant devices in order to predict and construct the original adversarial examples which 
will result in the ones in [1],[2] after passing through the speaker-air-microphone channel. We leveraged the power of deep learning 
neural network to mimic the speaker-air-microphone channel to provide high accuracy and avoid the complicated analysis of 
speaker/microphone circuits and the acoustic air channel.


The project uses deep neural networks (U-Net) to build a deep learning model 
that remove electronic noise and air noise to transmit adversarial examples 
over-the-air. Our contribution is to make the adversarial example attack to be 
a practical attack. In the further, this deep neural networks (U-Net) can also 
use to defense adversarial learning attack as well due to its strong noise 
remove ability.

For more problem details, please go to
[project website](https://ucla-ece209as-2018w.github.io/Weikun-Zhengshuang/).

## Background
------
### Speaker-air-microphone channel

### U-Net

## Neural Network Structure
------


## Design and Implementation
------
### Data Preparation




## Evaluation
------


## Discussion
------


## Future Works
------


## Related Works
------



## Requirements and Dependencies
------
The following packages are required (the version numbers that have been tested 
are given for reference):

* Python 2.7 or 3.6
* Tensorflow 1.0.1
* Numpy 1.12.1
* Librosa 0.5.0
* tqdm 4.11.2 (only for preprocessing datasets)
* Sox 1.2.7 (only for preprocessing datasets)

## References
------