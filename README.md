# Over-the-Air Speech Recogniztion Attack

## About This Project
### Overview
------
The course final project for UCLA EE209AS Winter 2018 - Special Topics in Circuits and Embedded Systems: 
Security and Privacy for Embedded Systems, Cyber-Physical Systems, and Internet of Things by Professor. Mani Srivastava.

* **Team members**: Weikun Han, Zhengshuang Ren 
* **Link to project website**: [project website](https://ucla-ece209as-2018w.github.io/Weikun-Zhengshuang/).
* Please refer to **/documentation** folder for **proposal** and **midterm report**.

### Requirements and Dependencies
------
The following packages are required (the version numbers that have been tested 
are given for reference):

* Python 2.7 or 3.6
* Tensorflow 1.0.1
* Numpy 1.12.1
* Librosa 0.5.0
* tqdm 4.11.2 (only for preprocessing datasets)
* Sox 1.2.7 (only for preprocessing datasets)


## Introduction
Recent progress in intelligent home assitant devices such as Google Home and Amazon Alexa
is changing people's daily lives and allows users to interact with their home 
devices in a smarter and more convenient. Such devices are integrated with speech recognition 
models mostly based on Deep Learning Neural Networks to recoginize users' voice commands. 
The advantage of using Deep Learning Models is their higher accuracy on recoginizing users' commands 
correctly than traditional machine learning algorithms. However, such devices bring new security concerns 
since they are operating users' private home devices and transimitting sensitive data and information about users'
private personal lives. Vulnerabilities of these devices may be exploited and used to cause users' property loss. 

### Problem Statement
------
Recent research has shown that deep learning models are easy to be fooled by attackers to perform untargeted or 
even targeted attacks by generating adversarial exampes to produce wrong recognized commands and to actuate users' 
home devices in unwanted ways. Moustafa Alzantot[1] and Nicholas Carlini[2] have demonstrated the vulnerabilities 
of such speech recognition models by generating adversarial examples to perform targeted attacks with high successful 
rates. Notice in [1], the author is performing black-box attacks without knowing the details about the recognition 
neural network whereas in [2], the attack is performed in white-box attacks leveraging the sturcture and details about the network.
However, they achieved the research-purpose attacks by deploying adversarial example files into the home assistant 
devices, but in practical attacks, over-the-air attacks are more realistic to perform since the attackers may not have physical 
access to the devices.  

### Objective
------
In this project, we proposed a way to generate the adversarial examples before the air channel so that 
the attackers are able to perform over-the-air attacks based on the adversarial examples they have from [1] or [2]. 
The main idea of our project is to mimic the air channel characteristics, the characteristics of the speaker used to play the adversarial 
examples and the microphone on the home assistant devices listening to the commands in order to predict and construct the original adversarial examples which will result in the ones in [1],[2] after passing through the speaker-air-microphone channel. We leveraged the power of deep learning neural network to mimic the speaker-air-microphone channel to provide high accuracy and avoid the complicated analysis of speaker/microphone circuits and the acoustic air channel.

## Background
### Speaker-Air-Microphone channel
------
Since we already have some high successful rate adversarial examples, why can't we perform the attack simply by playing them through speakers?
The answer is because the adversarial examples are carefully designed to achieve targeted attacks by manipulating bits in the audio files. Minor 
modifications on the adversarial examples may lead to low successful rate or even failure of the attacks. Playing the adversarial example audio 
files with speakers will let the voice pass through a complete speaker-air-microphone channel, where each single part will change the contents in 
the attack voice in ways that cannot be easily anticipated. 


<img src="https://github.com/UCLA-ECE209AS-2018W/Weikun-Zhengshuang/raw/master/img/SAM_channel.PNG" aligen="center" width="600"/>


#### Speaker
The amplifiers in the speaker circuits will shift the DC offset and amplify the audio singals and the analog components will add noise to the 
audio signals. Both will contribute to the changes in adversarial examples after they are played. 
#### Air channel
The environmental surrounding noise and air vibration will further add changes to the adversarial examples.
#### Microphone
Components in microphone circuits such as ADC and DSP filters will greatly modify the adersarial examples as well. 

### U-Net
------
U-Net[3] is a noval deep learning neural network structure that firstly was introduced for biomedical images processing.
The structure consists of several levels of residual blocks for downsamling and upsampling. Minimal features extraction 
is done through the downsampling path and background noise is added/removed through upsampling reconstruction. The residual 
blocks are served for a fine-turning purpose during reconstruction. This network structure is proved to provide high accuracy 
and validation loss in biomedical image processing applications of artifacts and background noise removal and super-resolution 
applications. In [4], it is used also for audio dataset in audio super-resolution/noise removal application. We expect it to work 
well on noise addition application as well. 

<img src="https://github.com/UCLA-ECE209AS-2018W/Weikun-Zhengshuang/raw/master/img/U-Net_bio.PNG" aligen="center" width="600" />

## Neural Network Structure





~~[INSERT Figure here to illustrate the OUR U-Net structure]~~
![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Our U-Net")

## Design and Implementation

### Data Preparation
------
#### Experiments Setup
<img src="https://github.com/UCLA-ECE209AS-2018W/Weikun-Zhengshuang/raw/master/img/experiment_setup.PNG" aligen="center" width="600" />
#### Data Processing
process .wav file with librosa 
trim and splice 
train/valid/test separation

### Network training and testing
------







## Evaluation
------


## Discussion
------


## Future Works
------


## Related Works
------





## References
[1] Moustafa Alzantot: https://arxiv.org/pdf/1801.00554.pdf  
[2] Nicholas Carlini: https://arxiv.org/pdf/1801.01944.pdf  
[3] Biomedical U-Net: https://arxiv.org/abs/1505.04597  
[4] Audio SR U-Net: https://blog.insightdatascience.com/using-deep-learning-to-reconstruct-high-resolution-audio-29deee8b7ccd  
[5] Subpixel convolutions: https://arxiv.org/abs/1609.05158  
[6] EnglishSpeechUpsampler GitHub Repo: https://github.com/jhetherly/EnglishSpeechUpsampler  