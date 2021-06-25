<p align="center">
  <img width="400" src="https://www.utem.edu.my/image/newlogo/LogoJawi.png" alt="UTeM logo">
</p>

# Artificial Intelligence Project

# FACE RECOGNITION ATTENDANCE SYSTEM


## 1.0 PROJECT SUMMARY
<p align="center">
Project Title: Face Recognition Attendance System

 _____________________________________________________________
</p>  
 
**Team Members:** 


![fs_education_mob_img_banner](https://user-images.githubusercontent.com/81746626/123410939-78be6180-d5e2-11eb-8bd6-853172aac359.jpeg)



:curly_haired_woman: Nurfarzana Amirah Binti Adnan			(B031910024)

:curly_haired_woman: Syaqirah Nur Fatihah Binti Sazalee		(B031910082)

:curly_haired_woman: Hamizah Binti Yusni				(B031910035)

:curly_haired_woman: Nurul Hidayati Rahmah Binti Mohd Hashim	(B031910380)










- **1.1  Objectives:**

:pushpin: Break out the project goal into more specific objectives

:pushpin: Cost-effective

:pushpin: Time saving

:pushpin: Easy to manage


##  2.0 ABSTRACT 

With the advancement of technologies each and every day, humanity is slowly going towards contactless everything.

It is quite evident that the future ahead of us will become so much advance that maybe 90%+ things that we are doing right now will be either automated or become contactless.

One such advancement will be the facial recognition technology or the FR tech, which is the prime focus of this article. 


Facial recognition technology is a system or software which is capable enough to verify the identity of a person from analyzing an image or video footage.

Some of the technologies or software are so advanced that even blurred pictures are sometimes rendered enough and analyzed to know the identity of the person.

So much are the advantages of this system that it would take a long article to note down each and every one of them.

But today, our prime focus will be on one of the many applications of facial recognition technology, and that is using face recognition based attendance system.

![fs_education_mob_img_banner](https://user-images.githubusercontent.com/66559983/115048576-4a4e4700-9f0c-11eb-922c-1bf62aa36da9.png)

<p align="center">
Figure 1 shows the AI output of detecting face for attandance.
</p>

## 3.0  DATASET

In this project, we’ll discuss our Face Recognition Attandance System, detailing how our computer vision/deep learning pipeline will be implemented.

From there, we’ll review the dataset we’ll be using to train our custom face attandance detector.

I’ll then show you how to implement a Python script to train a face attandance detector on our dataset using Keras and TensorFlow.

We’ll use this Python script to train a face attandance detector and review the results.

Given the trained face attandance detector, we’ll proceed to implement two more additional Python scripts used to:

- Detect face attandance in images
- Detect face attandance in real-time video streams

We’ll wrap up the post by looking at the results of applying our face attandance detector.


There is Three-phase of face attandance detector as shown in Figure 2:

(![0_oJIRaoERCUHoyylG_](https://user-images.githubusercontent.com/66559983/115047080-c21b7200-9f0a-11eb-88f2-0c6228cc25c7.png)

Figure 2: Phases and individual steps for building a face attandance detector with computer vision and deep learning 

In order to train a custom face attandance detector, we need to break our project into three distinct phases, each with its own respective sub-steps (as shown by Figure 1 above):

- Face Detection and Data Gathering : Here we’ll focus on loading our face attandance detection dataset from disk.

- Train the Recognizer : Training a model (using Keras/TensorFlow) on this dataset, and then serializing the face attandance detector to disk.

- Face Recognition : Once the face attandance detector is trained, we can then move on to loading the attandance detector, performing face detection, and then classifying each face with their own specific id number.

We’ll review each of these phases and associated subsets in detail in the remainder of this tutorial, but in the meantime, let’s take a look at the dataset we’ll be using to train our face attandance detector.


Our face attandance detection dataset as shown in Figure 3:

![1_qK_TqFBjxq45_vfpgdJrwQ](https://user-images.githubusercontent.com/66559983/115047537-30f8cb00-9f0b-11eb-9334-3a3098c34cce.png)


Figure 3: A face attandance detection dataset consists of their own face id images. 

The dataset we’ll be using here today was created by PyImageSearch reader Prajna Bhandary.

This dataset consists of 40 images belonging to :

- face id : 0 until face id : 39


Our goal is to train a custom deep learning model to record the attandance by recognise their face.

How was our face attandance dataset created?
Prajna, like me, has been feeling down and depressed about the state of the world — thousands of people are dying each day, and for many of us, there is very little (if anything) we can do.

To help keep her spirits up, Prajna decided to distract herself by applying computer vision and deep learning to solve a real-world problem:

- Best case scenario — she could use her project to help others
- Worst case scenario — it gave her a much needed mental escape


## 4.0   PROJECT STRUCTURE

The following directory is our structure of our project:
- $ tree --dirsfirst --filelimit 10
- .
- ├── dataset
- │   └── face id 0 until 39.
- ├── examples
- │   ├── example_01.png
- │   ├── example_02.png
- │   └── example_03.png
- ├── face_detector
- │   ├── deploy.prototxt
- │   └── res10_300x300_ssd_iter_140000.caffemodel
- ├── detect_mask_image.py
- ├── detect_mask_video.py
- ├── mask_detector.model
- ├── plot.png
- └── train_mask_detector.py
- 5 directories, 10 files


The dataset/ directory contains the data described in the “Our COVID-19 face mask detection dataset” section.

Three image examples/ are provided so that you can test the static image face mask detector.

We’ll be reviewing three Python scripts in this tutorial:

- train_mask_detector.py: Accepts our input dataset and fine-tunes MobileNetV2 upon it to create our mask_detector.model. A training history plot.png containing accuracy/loss curves is also produced
- detect_mask_image.py: Performs face mask detection in static images
- detect_mask_video.py: Using your webcam, this script applies face mask detection to every frame in the stream

In the next two sections, we will train our face mask detector.



## 5.0   TRAINING THE FACE ATTANDANCE DETECTION

We are now ready to train our face attandance detector using Keras, TensorFlow, and Deep Learning.

From there, open up a terminal, and execute the following command:

- $ python train_mask_detector.py --dataset dataset
- [INFO] loading images...
- [INFO] compiling model...
- [INFO] training head...
- Train for 34 steps, validate on 276 samples
- Epoch 1/20
- 34/34 [==============================] - 30s 885ms/step - loss: 0.6431 - accuracy: 0.6676 - val_loss: 0.3696 - val_accuracy: 0.8242
- Epoch 2/20
- 34/34 [==============================] - 29s 853ms/step - loss: 0.3507 - accuracy: 0.8567 - val_loss: 0.1964 - val_accuracy: 0.9375
- Epoch 3/20
- 34/34 [==============================] - 27s 800ms/step - loss: 0.2792 - accuracy: 0.8820 - val_loss: 0.1383 - val_accuracy: 0.9531
- Epoch 4/20
- 34/34 [==============================] - 28s 814ms/step - loss: 0.2196 - accuracy: 0.9148 - val_loss: 0.1306 - val_accuracy: 0.9492
- Epoch 5/20
- 34/34 [==============================] - 27s 792ms/step - loss: 0.2006 - accuracy: 0.9213 - val_loss: 0.0863 - val_accuracy: 0.9688
- ...
- Epoch 16/20
- 34/34 [==============================] - 27s 801ms/step - loss: 0.0767 - accuracy: 0.9766 - val_loss: 0.0291 - val_accuracy: 0.9922
- Epoch 17/20
- 34/34 [==============================] - 27s 795ms/step - loss: 0.1042 - accuracy: 0.9616 - val_loss: 0.0243 - val_accuracy: 1.0000
- Epoch 18/20
- 34/34 [==============================] - 27s 796ms/step - loss: 0.0804 - accuracy: 0.9672 - val_loss: 0.0244 - val_accuracy: 0.9961
- Epoch 19/20
- 34/34 [==============================] - 27s 793ms/step - loss: 0.0836 - accuracy: 0.9710 - val_loss: 0.0440 - val_accuracy: 0.9883
- Epoch 20/20
- 34/34 [==============================] - 28s 838ms/step - loss: 0.0717 - accuracy: 0.9710 - val_loss: 0.0270 - val_accuracy: 0.9922
- [INFO] evaluating network...

|      |    precision    | recall| f1-score | support |
|------|-----------------|-------|----------|---------|
|with_mask|0.99|1.00|0.99|138|
|without_mask|1.00|0.99|0.99|138|
|accuracy| | |0.99|276|
|macro avg|0.99|0.99|0.99|276|
|weighted avg|0.99|0.99|0.99|276|


![Figure 4](https://www.pyimagesearch.com/wp-content/uploads/2020/04/face_mask_detector_plot.png)

Figure 4: Figure 10: COVID-19 face mask detector training accuracy/loss curves demonstrate high accuracy and little signs of overfitting on the data

As you can see, we are obtaining ~99% accuracy on our test set.

Looking at Figure 4, we can see there are little signs of overfitting, with the validation loss lower than the training loss. 

Given these results, we are hopeful that our model will generalize well to images outside our training and testing set.


## 6.0  RESULT AND CONCLUSION

Detecting face attandance with OpenCV in real-time

You can then launch the attandance detector in real-time video streams using the following command:
- $ python detect_mask_video.py
- [INFO] loading face detector model...
- INFO] loading face mask detector model...
- [INFO] starting video stream...



![0_WtQqicLOVn1TukA2_](https://user-images.githubusercontent.com/66559983/115048873-a6b16680-9f0c-11eb-9cd1-7772e80c9885.png)

Figure 5: Attandance detector in real-time video streams

In Figure 5, you can see that our face attandance detector is capable of running in real-time (and is correct in its predictions as well.



## 7.0   PROJECT PRESENTATION 

In this project, you learned how to create a face attandance detector using OpenCV, Keras/TensorFlow, and Deep Learning.

To create our face attandance detector, we trained 40 peoples who already registered in the attandance system.

We fine-tuned MobileNetV2 on our attandance dataset and obtained a classifier that is ~99% accurate.

We then took this face attandance classifier and applied it to both images and real-time video streams by:

- Detecting faces in images/video
- Extracting each individual face
- Applying our face attandance classifier

Our face attandance detector is accurate, and since we used the MobileNetV2 architecture, it’s also computationally efficient, making it easier to deploy the model to embedded systems (Raspberry Pi, Google Coral, Jetosn, Nano, etc.).

[![demo](https://www.youtube.com/watch?v=sz25xxF_AVE/0.jpg)](https://www.youtube.com/watch?v=sz25xxF_AVE "demo")





