# Course Project for Machine Learning

This repository contains the code and data we used in the course project of machine learning[2019 Spring] in SJTU.



## Dataset

The dataset we used is 'FER2013' from <https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data>. 

Image Properties: 

- 48 x 48 pixels (2304 bytes)
-  labels: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral 
- The training set consists of 28,709 examples. The public test set consists of 3,589 examples. The private test set consists of another 3,589 examples.

You should download the 'fer2013.csv' file from the website above and put it in the 'data' folder.

The 'images' folder contains some images we saved from 'FER2013' dataset for adversarial examples visiualization.



## Code

We implement both SVM and CNN classfications on the dataset, and we use FGSM algorithm to get adversarial examples to attack the network. 

### csvPreprocess.py

This file reads the original 'fer2013.csv' file,  preprecesses the data and saves the data in '.npy' files.

### my_CNN.py

This file defines the structure of CNN.

### FGSM.py

This file contains some functions we used to implement FGSM algorithm.

### main.py

This file includes the whole process we described in the final report: SVM, CNN and FGSM. 



## Links

### DeepFool

<https://github.com/LTS4/DeepFool>

This link implements the DeepFool algorithm to get adversarial examples.

### VGG19

<https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch>

This link implements a high accuracy claasficatoin on 'FER2013' dataset with VGG19.




