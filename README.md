# Over-the-Air Speech Recogniztion Attack

## About This Project
### Overview
------
The course final project for UCLA EE209AS Winter 2018 - Special Topics in Circuits and Embedded Systems: 
Security and Privacy for Embedded Systems, Cyber-Physical Systems, and the Internet of Things by Professor. Mani Srivastava.
In this project, we use deep learning (audio U-Net) to build model remove electronic noise and air noise during adversarial example transmission over the air. The contribution of this project are:
* make the adversarial example attack can transmit over-the-air, which eventually be a practical attack.
* found audio U-Net is also a possible defense for adversarial example attack due to strong ability to remove noise.

### Responsibility
------
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
neural network whereas in [2], the attack is performed in white-box attacks leveraging the sturcture and details about the 
network.
However, they achieved the research-purpose attacks by deploying adversarial example files into the home assistant 
devices, but in practical attacks, over-the-air attacks are more realistic to perform since the attackers may not have 
physical access to the devices.  

### Objective
------
In this project, we proposed a way to generate the adversarial examples before the air channel so that 
the attackers are able to perform over-the-air attacks based on the adversarial examples they have from [1] or [2]. 
The main idea of our project is to mimic the air channel characteristics, the characteristics of the speaker used to play 
the adversarial examples and the microphone on the home assistant devices listening to the commands in order to predict and 
construct the original adversarial examples which will result in the ones in [1],[2] after passing through the speaker-air-
microphone channel. We leveraged the power of deep learning neural network to mimic the speaker-air-microphone channel to 
provide high accuracy and avoid the complicated analysis of speaker/microphone circuits and the acoustic air channel.

## Background
### Speaker-Air-Microphone channel
------
Since we already have some high successful rate adversarial examples, why can't we perform the attack simply by playing them 
through speakers? The answer is because the adversarial examples are carefully designed to achieve targeted attacks by 
manipulating bits in the audio files. Minor modifications on the adversarial examples may lead to low successful rate or 
even failure of the attacks. Playing the adversarial example audio files with speakers will let the voice pass through a 
complete speaker-air-microphone channel, where each single part will change the contents in the attack voice in ways that 
cannot be easily anticipated. 

<img src="https://github.com/UCLA-ECE209AS-2018W/Weikun-Zhengshuang/raw/master/img/SAM_channel.PNG" aligen="center" width="600"/>

#### Speaker
The amplifiers in the speaker circuits will shift the DC offset and amplify the audio singals and the analog components will 
add noise to the audio signals. Both will contribute to the changes in adversarial examples after they are played. 
#### Air channel
The environmental surrounding noise and air vibration will further add changes to the adversarial examples.
#### Microphone
Components in microphone circuits such as ADC and DSP filters will greatly modify the adersarial examples as well. 

### U-Net
------
U-Net[3] is a noval deep learning neural network structure that firstly was introduced for biomedical images processing.
The structure consists of several levels of residual blocks for downsamling and upsampling. Minimal features extraction 
is done through the downsampling path and background noise is added/removed through upsampling reconstruction. The residual 
blocks are served for a fine-turning purpose during reconstruction. This network structure is proved to provide high 
accuracy and validation loss in biomedical image processing applications of artifacts and background noise removal and 
super-resolution applications. 

<img src="https://github.com/UCLA-ECE209AS-2018W/Weikun-Zhengshuang/raw/master/img/U-Net_bio.PNG" aligen="center" width="600" />

### Audio U-Net
------
Recently, the U-Net deep neural networks have been applying to enhance the quality audio signal[7]. These deep neural 
networks use same ideas as U-Net which can transfer inputs encoded at low sampling rates into higher-quality signals with an 
increased resolution in the time domain. The structure of audio U-Net is as blow figure. This technique has applications in 
telephony, compression, and text-to-speech generation and suggests new architectures for generative models of audio. In [4], 
it is used also for the audio dataset in audio super-resolution/noise removal application. We expect it to work well on 
noise addition application as well. 

<img src="https://github.com/UCLA-ECE209AS-2018W/Weikun-Zhengshuang/raw/master/img/audio_u_net.png" aligen="center" width="600" />

## Design and Implementation
### Data Preparation
------
The datasets we use for this project is TED-LIUM Corpus[8]. First, since is datasets is very large, you need down it and run 
[preprocessing_audio.ipynb](https://github.com/UCLA-ECE209AS-2018W/Weikun-Zhengshuang/blob/master/preprocessing_audio.ipynb) 
to extract audio into .wav format. Next, you can generate difference noise sample to train the 
audio U-Net. The way we generate noise sample is showed in experiments setup and data processing. You free to design noise 
sample base on the different project purpose. We provide some example noise on [GitHub](https://github.com/UCLA-ECE209AS-2018W/Weikun-Zhengshuang/tree/master/datasets), but we cannot upload full datasets due to the limitation of upload size. 

#### Experiments Setup
The blow figure shows the environment of noise sample generation. To  generate noise is the sample, our settings of this 
experiment is:
* software media player: Groove Music (Window 10), volume setting 4/100
* speaker: SoundLink Mini II (Bose), volume setting 100/100
* recorder: Snowball (Blue), default
* software media recorder: Arecord (Ubuntu 17.10), setting as blow figure
* distance between the speaker and recorder: 30cm

<img src="https://github.com/UCLA-ECE209AS-2018W/Weikun-Zhengshuang/raw/master/img/experiment_setup.JPG" aligen="center" width="600" />

<img src="https://github.com/UCLA-ECE209AS-2018W/Weikun-Zhengshuang/raw/master/img/record_setup.png" aligen="center" width="600" />

#### Data Preprocessing
After recording the noise sample, you also need the use professor audio editor to remove the audio offset. Here, we use Audacity to do it. The audio most off before removing is showed as the blow.

<img src="https://github.com/UCLA-ECE209AS-2018W/Weikun-Zhengshuang/raw/master/img/processing_recorded_audio_1.PNG" aligen="center" width="600" />

Here, you need to make sure noise sample and original sample have the same offset, which can check easily as the figure below.

<img src="https://github.com/UCLA-ECE209AS-2018W/Weikun-Zhengshuang/raw/master/img/processing_recorded_audio_2.PNG" aligen="center" width="600" />

Finally, you need remove have offset of last part of noise sample. The way to remove it very easy, it showed as blow figure.
Here, you have successfully got noise sample. Remember save same audio format as the orignal audio's format.

<img src="https://github.com/UCLA-ECE209AS-2018W/Weikun-Zhengshuang/raw/master/img/processing_recorded_audio_3.PNG" aligen="center" width="600" />

<img src="https://github.com/UCLA-ECE209AS-2018W/Weikun-Zhengshuang/raw/master/img/processing_recorded_audio_4.PNG" aligen="center" width="600" />

#### Create train, validation, and test datasets
After you get the noise sample datasets, you need first trim your original and noise sample into audio clips. In this 
project, we set the audio clips is 5s long period. You can set any period time depending on your project. To trim the 
original and noise sample, you could directly run [preprocessing_audio_split.ipynb](https://github.com/UCLA-ECE209AS-2018W/Weikun-Zhengshuang/blob/master/preprocessing_audio_split.ipynb). 

Next, you need to prepare the train, validation, and test datasets. To do that, we directly create a .csv file to record 
each original and noise audio 5s clip file path (here, we name it as origal_noise_pairs). In this way, after we record all 
pairs of original and noise audio, we can shuffle each pair in .csv and split into train, validation, and test datasets. To 
save all file pairs and shuffle it, you can run [preprocessing_split_train_test.ipynb](https://github.com/UCLA-ECE209AS-2018W/Weikun-Zhengshuang/blob/master/preprocessing_split_train_test.ipynb). In the project, we use 60% for training, 20% for 
validation, and 20% for testing. 

### Model
------
Our model is basically same as the audio U-Net as [7]. The total downsampling cell is 8 in our DNNs, and we use 8 upsampling 
cell as well. Furthermore, we use batch normalization instead of using dropout. Moreover, we add another stacking layer and 
one more 1-dimensional convolution layer at the end of DNNs, which can get better performance. The basic structure of our 
model is as shown as the blow. 

<img src="https://github.com/UCLA-ECE209AS-2018W/Weikun-Zhengshuang/raw/master/img/network.PNG" aligen="center" width="600" />

The bottom figure shows that details in each downsampling and upsampling cell. As mentioned before, we use batch 
normalization instead of using dropout, because batch normalization has better performance. The bottleneck cell and 
downsample cell have the same structure, and meaning of bottleneck cell is same as stacking layer plus 1-dimensional 
convolution layer at the end of whole DNNs. [5] demonstrated a novel method to generate super-resolution image. They call it 
as subpixel shuffle layer (some people called dimensional shuffle layer). This layer makes zero padding for each pixel of 
the image, which can generate high resolution as the result. The audio U-Net use this subpixel shuffle layer to generate 
high sampling rate result. 

<img src="https://github.com/UCLA-ECE209AS-2018W/Weikun-Zhengshuang/raw/master/img/network2.PNG" aligen="center" width="600" />

## Evaluation
The mehod we used to evaluate our result do not base on the means square errors (MSE).For our project is eliminate noise, the MSE between noise sample and original is tiny. Therefore, we use some 


## Discussion

### Conventional Approach
The most conventional and straightforward approach for this problem would be to use the combination of Digital Signal Processing(DSP), Circuit Analysis and Physics 
to build a model to anticipate the changes that would happen to the adversarial examples so that proper manipulation can be made in order to do reverse-engineering. 
However, this approach requires very intense knowledge in all three domains and introduce much high complexity. Moreover, there are uncontrollable noise in speaker/mircrophone 
circuits and air acoustic channel which are bottlenecks to improve the accuracy. Additionally, this approach does not guarantee the resulting audio still recognizable to human ears 
because this approach can not consider the meaningfulness of the words or sentences it processes since it only focuses on circuits/physics.

### DataSet
In this project, we used TED-LIUM dataset for training and testing our network. The reasons as listed as followings:
* This dataset consists of hours of TED speech recording, other dataset only have seconds-long or minutes long audio file.
* Due to its long duration, we can simplify and shorten data processing so that we can focuse more on network tuning and improving.
* Meanwhile, it does not sacrifice its quality and it's meaningful for our network training, meaning that it is simliar to our objective input data
* Other DataSet can be considered to use as test input in future work

### Loss Function
We used mean sqaure error (MSE) as our loss function, the most fundamental loss function to calculate training loss and validation loss. The main purpose is to 
try to minimize the bit-wise difference between output data and label data, since the objective for this network is to get outputs which are as close to 
the generated adversarial examples as possible. MSE can serve for this objective well enough. Some other popular loss functions such as L2 or gradient-related loss function 
consider more about neighboring bits in order to achieve better loss from a larger picture. Intuitively, we do not care much about the beighboring bits since we 
want to get optimal bit-wise difference since less bit-wise difference to lead to higher successful rate.  


## Related work

CommanderSong[6]
Yuan et al. brings up this idea to 

Very noisy 

DolphinAtack[7]


## Future Works

In this project, we only used a portion of the TED-LIUM dataset for network training, validating and testing due to timing constraint and training time consideration.
To improve the model, more data should be used to train the network in order to achieve better performance, bias/variance tradeoff and avoid overfitting problem.
Other than TED-LIUM dataset, other dataset should also be considered to improve the variablity of the training dataset to further improve the trained model.

We did not have time to test whether this model is able to generate successful adversarial examples before SAM channel and we only demonstrated the the accuracy we can 
obtain on testing dataset. Combining our work and adversarial examples generation works such as Moustafa Alzantot's work or Nicholas Carlini's work is definitely 
the further direction of this project. Either we can keep the separated training and tune our network based on testing result or we can combine these two network and train 
it as one model. 

## Conclution

Current research in the field does not provide an efficient approach for air attacks targeting home assistant devices. DolphinAttack requires very special hardware equipment 
to generate ultrasound whereas CommanderSong generates extremely noisy results which are easily noticed. We proposed a new approach based on deep neural network to simulate the 
speaker-air-microphone channel in order to provide ability for reverse engineering and generate adversarial examples before SAM channel which will result in successful adversarial 
examples after passing through SAM channel. Our results achieve -------- ~~Fill in evaluation result here~~.

## References

[1] Moustafa Alzantot, Did you hear that? Adversarial Examples Against Automatic Speech Recognition: https://arxiv.org/pdf/1801.00554.pdf  
[2] Nicholas Carlini, Audio Adversarial Examples: Targeted Attacks on Speech-to-Text: https://arxiv.org/pdf/1801.01944.pdf  
[3] Olaf Ronneberger, U-Net: Convolutional Networks for Biomedical Image Segmentation: https://arxiv.org/pdf/1505.04597.pdf  
[4] Jeffrey Hetherly, Using Deep Learning to Reconstruct High-Resolution Audio: https://blog.insightdatascience.com/using-deep-learning-to-reconstruct-high-resolution-audio-29deee8b7ccd  
[5] Wenzhe Shi, Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network: https://arxiv.org/abs/1609.05158  

[6] Xuejing Yuan, CommanderSong: A Systematic Approach for Practical Adversarial Voice Recognition: https://arxiv.org/abs/1801.08535
[7] Guoming Zhang, DolphinAtack: Inaudible Voice Commands: https://arxiv.org/abs/1708.09537


[8] Volodymyr Kuleshov, Audio Super Resolution using Neural Networks, https://arxiv.org/pdf/1708.00853.pdf   
[9] TED-LIUM Corpus, http://www-lium.univ-lemans.fr/en/content/ted-lium-corpus

