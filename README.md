<p align="center">
  <img width="1000" src="http://ftmk.utem.edu.my/web/wp-content/uploads/2020/07/Logo-KPTFTMK2-1024x206.png" alt="logo">
</p>

# Artificial Intelligence Project

# FACE RECOGNITION ATTENDANCE SYSTEM


## 1.0 PROJECT SUMMARY
<p align="center">
Project Title: Face Recognition Attendance System

 _____________________________________________________________
</p>  
 
**Team Members:** 

<p align="center">
  <img width="500" src="https://user-images.githubusercontent.com/81746626/123410939-78be6180-d5e2-11eb-8bd6-853172aac359.jpeg" alt="example">

 _____________________________________________________________
  </p>

**1.1  Problem Statement :**

:clipboard: Traditional student attendance marking

:clipboard: technique is often facing a lot of trouble and take time.

:clipboard: Traditional method are usually disturbing learning and teaching process.

:clipboard: Class with many students might be difficult to pass the attendance sheet.

:clipboard: Student can tick attendance for friend who is absent.

<p align="center">
_____________________________________________________________
</p>

**1.2  Objectives:**

:pushpin: To automate user identification via face detection and recognition.

:pushpin: To detect face attendance in image.

:pushpin: To detect face attendance in real-time video stream.

:pushpin: To record attendance of identified student.

_____________________________________________________________
 
##  2.0 BACKGROUND 

With the advancement of technologies each and every day, humanity is slowly going towards contactless everything. It is quite evident that the future ahead of us will become so much advance that maybe 90%+ things that we are doing right now will be either automated or become contactless. One such advancement will be the facial recognition technology or the FR tech, which is the prime focus of this article. 

Facial recognition technology is a system or software which is capable enough to identify an identity of family members, friends or anyone by analyzing an image or video footage. Some of the technologies or software are so advanced that even blurred pictures are sometimes rendered enough and analyzed to know the identity of the person.  In order to overcome human limitations on memorizing each of human face, this face recognition attendance system would be able to do a face recognition and store the image in database with high processing speed.

In this article, our focus will be on one of the many applications of facial recognition technology, which is **Face Recognition Attendance System**.

<p align="center">
  <img width="500" src="https://user-images.githubusercontent.com/66559983/123439769-a4524380-d604-11eb-859f-689e8bd42dca.png" alt="example">
</p>

<p align="center">
Figure 1 shows the AI output of detecting face for attandance.
</p>


## 3.0  DATASET

In this project, we will discuss about our Face Recognition Attandance System which will detailing how our computer vision or deep learning pipeline will be implemented. From there, we will review the dataset to train our custom face attandance detector. The implementation of a Python script will be shown in this article to train a face attandance detector on our dataset using Keras and TensorFlow. We will use this Python script to train a face attandance detector and review the results.

We will proceed to implement two additional Python scripts by using:

:mag: Detect face attandance in images

:mag: Detect face attandance in real-time video streams

We will wrap up the post by looking at the results of applying our face attandance detector.


There is Three-phase of face attandance detector as shown in Figure 2:

<p align="center">
  <img width="700" src="https://user-images.githubusercontent.com/66559983/123443079-23954680-d608-11eb-8d2c-f7f7a96c0250.png" alt="Figure 2">
</p>

<p align="center">
Figure 2: Phases and individual steps for building a face attendance detector with computer vision and deep learning.
</p>
  
  
In order to train a custom face attandance detector, we need to break our project into three distinct phases, each with its own respective sub-steps :

- Face Detection and Data Gathering : Here we will focus on loading our face attandance detection dataset from disk.

- Train the Recognizer : Training a model (using Keras/TensorFlow) on this dataset, and then serializing the face attandance detector to disk.

- Face Recognition : Once the face attandance detector is trained, we can then move on to loading the attandance detector, performing face detection, and then classifying each face with their own specific id number.

We will review each of these phases and associated subsets in detail in the remainder of this tutorial. In the meantime, we will review the dataset used to train our face attandance detector.


Our face attandance detection dataset as shown in Figure 3:

<p align="center">
  <img width="700" src="https://user-images.githubusercontent.com/66559983/123445596-9e5f6100-d60a-11eb-9ddd-481eb2c828f7.png" alt="Figure 2">
</p>

<p align="center">
Figure 3: A face attendance detection dataset consists of their own face id images.
</p>

## 4.0   PROJECT STRUCTURE

The following directory is our structure of our project:

- ├── Dataset
- │ ├──   Dayah.jpg
- │ ├──  Dayah1.jpg
- │ ├──  Dayah2.jpg
- │ ├──  Dayah3.jpg
- │ ├──  Dayah4.jpg
- │ ├──  Mirah.jpg
- │ ├──  Mirah1.jpg
- │ ├──  Mirah2.jpg
- │ ├──  Mirah3.jpg
- │ ├──  Mirah4.jpg
- │ ├──  Miza.jpg
- │ ├──  Miza1.jpg
- │ ├──  Miza2.jpg
- │ ├──  Miza3.jpg
- │ ├──  Miza4.jpg
- │ ├──  Syaqirah.jpeg
- │ ├──  Syaqirah1.jpeg
- │ ├──  Syaqirah2.jpeg
- │ ├──  Syaqirah3.jpeg
- │ └── Syaqirah4.jpeg
- ├──  ImagesAttendance
- │ ├──  Hamizah binti Yusni (B0319190035).png
- │ ├──  Nurfarzana Amirah binti Adnan (B031910024).jpg
- │ ├──  Nurul Hidayati Rahmah binti Mohd Hashim (B031910380).JPG
- │ └── Syaqirah Nur Fatihah binti Sazalee (B031910082).jpeg
- ├──  face-train.py
- ├──  Attendance.csv
- └── AttendanceProject.py

- 2 directories, 28 files
Two Python scripts:
	:label: face-train.py: Accepts our input dataset.
	:label: AttendanceProject.py: Using your webcam, this script applies attendance face recognition to every frame in the stream.


## 5.0   TRAINING THE FACE ATTANDANCE DETECTION

Let’s try this face recognition attendance out on some of our own images now.
We will run our script at PyCharm and webcam will prompt out:
Example:

<p align="center">
  <img width="700" src="https://user-images.githubusercontent.com/66559983/123445596-9e5f6100-d60a-11eb-9ddd-481eb2c828f7.png" alt="Figure 2">
</p>



|                                                      CODE                                                          |
|--------------------------------------------------------------------------------------------------------------------|
|    |
|"C:\Users\Hamizah Yusni\PycharmProjects\FaceRecognitionAttendance\venv\Scripts\python.exe"|
|"C:/Users/Hamizah Yusni/PycharmProjects/FaceRecognitionAttendance/AttendanceProject.py"|
|    |
|['Hamizah binti Yusni (B031910035).png', 'Nurfarzana Amirah binti Adnan (B031910024).jpg','Nurul Hidayati Rahmah binti Mohd Hashim (B031910380).JPG', 'Syaqirah Nur Fatihah binti Sazalee (B031910082).jpeg']|
|    |
|['Hamizah binti Yusni (B031910035)', 'Nurfarzana Amirah binti Adnan (B031910024)','Nurul Hidayati Rahmah binti Mohd Hashim (B031910380)', 'Syaqirah Nur Fatihah binti Sazalee (B031910082)']|
|    |
|Encoding Complete|

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





