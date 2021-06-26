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
  <img width="650" src="https://user-images.githubusercontent.com/81746626/123410939-78be6180-d5e2-11eb-8bd6-853172aac359.jpeg" alt="example">

 _____________________________________________________________
  </p>

**1.1  Problem Statement :**

:clipboard: Traditional student attendance marking

:clipboard: technique is often facing a lot of trouble and take time.

:clipboard: Traditional method are usually disturbing learning and teaching process.

:clipboard: Class with many students might be difficult to pass the attendance sheet.

:clipboard: Student can tick attendance for friend who is absent.

_____________________________________________________________


**1.2  Objectives:**

:pushpin: To automate user identification via face detection and recognition.

:pushpin: To detect face attendance in image.

:pushpin: To detect face attendance in real-time video stream.

:pushpin: To record attendance of identified student.

<br /> <br /> 
_____________________________________________________________
##  2.0 BACKGROUND 


With the advancement of technologies each and every day, humanity is slowly going towards contactless everything. It is quite evident that the future ahead of us will become so much advance. One such advancement will be the facial recognition technology or the FR tech, which is the prime focus of this project. 

Facial recognition technology is a system or software which is capable enough to identify an identity of family members, friends or anyone by analyzing an image or video footage. In order to overcome human limitations on memorizing each of human face, this face recognition attendance system would be able to do a face recognition and store the image in database with high processing speed. <br /> <br />

In this project, our focus will be on one of the many applications of facial recognition technology, which is **Face Recognition Attendance System**.

<p align="center">
  <img width="700" src="(https://user-images.githubusercontent.com/66559983/123521045-c1f7d980-d6e6-11eb-98c3-dc3a21bcd510.png" alt="example">
</p>


<p align="center">
Figure 1 shows the AI output of detecting face for attandance.
</p>

<br /> <br /> 
_____________________________________________________________
## 3.0  DATASET

In this project, we will discuss about our Face Recognition Attandance System which will detailing how our computer vision or deep learning pipeline will be implemented. From there, we will review the dataset to train our custom face attandance detector. The implementation of a Python script will be shown in this article to train a face attandance detector on our dataset using Keras and TensorFlow. We will use this Python script to train a face attandance detector and review the results.

We will proceed to implement two additional Python scripts by using:

:mag: Detect face attandance in images

:mag: Detect face attandance in real-time video streams

We will wrap up the post by looking at the results of applying our face attandance detector.

<br /> <br />


There is Three-phase of face attandance detector as shown in Figure 2:

<p align="center">
  <img width="700" src="https://user-images.githubusercontent.com/66559983/123520019-40ea1380-d6e1-11eb-8504-9a79f1327446.png" alt="Figure 2">
</p>

<p align="center">
Figure 2: Phases and individual steps for building a face attendance detector with computer vision and deep learning.
</p>
  
<br /> <br /> 
  
In order to train a custom face attandance detector, we need to break our project into three distinct phases, each with its own respective sub-steps :

:black_nib: Face Detection and Data Gathering : Here we will focus on loading our face attandance detection dataset from disk.

:black_nib: Train the Recognizer : Training a model (using Keras/TensorFlow) on this dataset, and then serializing the face attandance detector to disk.

:black_nib: Face Recognition : Once the face attandance detector is trained, we can then move on to loading the attandance detector, performing face detection, and then classifying each face with their own specific id number.

_____________________________________________________________


We will review each of these phases and associated subsets in detail in the remainder of this tutorial. In the meantime, we will review the dataset used to train our face attandance detector. <br /> <br />


Our face attandance detection dataset as shown in Figure 3:

<p align="center">
  <img width="700" src="https://user-images.githubusercontent.com/66559983/123445596-9e5f6100-d60a-11eb-9ddd-481eb2c828f7.png" alt="Figure 2">
</p>

<p align="center">
Figure 3: A face attendance detection dataset consists of their own face id images.
</p>

<br /> <br /> 
_____________________________________________________________
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
- │ ├──  Hamizah binti Yusni - (B0319190035).png
- │ ├──  Nurfarzana Amirah binti Adnan - (B031910024).jpg
- │ ├──  Nurul Hidayati Rahmah binti Mohd Hashim - (B031910380).JPG
- │ └── Syaqirah Nur Fatihah binti Sazalee - (B031910082).jpeg
- ├──  face-train.py
- ├──  Attendance.csv
- └── AttendanceProject.py

- 2 directories, 28 files

Two Python scripts:

:label: face-train.py: Accepts our input dataset.

:label: AttendanceProject.py: Using your webcam, this script applies attendance face recognition to every frame in the stream.

<br /> <br /> 
_____________________________________________________________
## 5.0   TRAINING THE FACE ATTANDANCE DETECTION

Let’s try this face recognition attendance out on some of our own images now.
We will run our script at PyCharm and webcam will prompt out:

<br />

_____________________________________________________________

:camera: Example 1 : Python command to execute & evaluate the student, Hamizah picture based on the dataset.

|                                                      CODE                                                          |
|--------------------------------------------------------------------------------------------------------------------|
|    |
|"C:\Users\Hamizah Yusni\PycharmProjects\FaceRecognitionAttendance\venv\Scripts\python.exe"|
|"C:/Users/Hamizah Yusni/PycharmProjects/FaceRecognitionAttendance/face-train.py"|
|    |
|[True] [0.40082248]|
|    |
|[True] [0.57053437]|
|    |
|[True] [0.63591632]|
|    |
|[True] [0.58169129]|

<br />


**Output 1:** 

<p align="center">
  <img width="500" src="https://user-images.githubusercontent.com/66559983/123520173-00d76080-d6e2-11eb-9d96-3110eafb4b4a.png" alt="Training 1">
</p>

<p align="center">
Figure 4: Program output after a student, Hamizah compare her face with her own face and other students.
</p>

<br />

- Hamizah got her own face as the lowest face distance, which means that the system is accurate.

<br />

_____________________________________________________________


:camera: Example 2: Python command to execute & evaluate the student, Amirah picture based on the dataset.


|                                                      CODE                                                          |
|--------------------------------------------------------------------------------------------------------------------|
|    |
|"C:\Users\Hamizah Yusni\PycharmProjects\FaceRecognitionAttendance\venv\Scripts\python.exe"|
|"C:/Users/Hamizah Yusni/PycharmProjects/FaceRecognitionAttendance/face-train.py"|
|    |
|[True] [0.52788831]|
|    |
|[True] [0.28396551]|
|    |
|[True] [0.5750813]|
|    |
|[False] [0.74549696]|

<br />

**Output 2:**

<p align="center">
  <img width="500" src="https://user-images.githubusercontent.com/66559983/123520189-1482c700-d6e2-11eb-9971-86bcbda3eca3.png" alt="Training 2">
</p>

<p align="center">
Figure 5: Program output after a student, Amirah compare her face with her own face and other students.
</p>

<br />

- Amirah got her own face as the lowest face distance, which means that the system is accurate.

<br />

_____________________________________________________________


:camera: Example 3: Python command to execute & evaluate the student, Hidayati picture based on the dataset.

|                                                      CODE                                                          |
|--------------------------------------------------------------------------------------------------------------------|
|    |
|"C:\Users\Hamizah Yusni\PycharmProjects\FaceRecognitionAttendance\venv\Scripts\python.exe"|
|"C:/Users/Hamizah Yusni/PycharmProjects/FaceRecognitionAttendance/face-train.py"|
|    |
|[False] [0.63179693]|
|    |
|[True] [0.56561713]|
|    |
|[True] [0.32760166]|
|    |
|[False] [0.64222176]|

<br />

**Output 3:**

<p align="center">
  <img width="500" src="https://user-images.githubusercontent.com/66559983/123520729-a4c20b80-d6e4-11eb-98d9-79e957d57dfb.png" alt="Training 3">
</p>


<p align="center">
Figure 6: Program output after a student, Hidayati compare her face with her own face and other students.
</p>

<br /> 

- Hidayati got her own face as the lowest face distance, which means that the system is accurate.

<br />

_____________________________________________________________


:camera: Example 4: Python command to execute & evaluate the student, Syaqirah picture based on the dataset.

|                                                      CODE                                                          |
|--------------------------------------------------------------------------------------------------------------------|
|    |
|"C:\Users\Hamizah Yusni\PycharmProjects\FaceRecognitionAttendance\venv\Scripts\python.exe"|
|"C:/Users/Hamizah Yusni/PycharmProjects/FaceRecognitionAttendance/face-train.py"|
|    |
|[False] [0.6255457]|
|    |
|[False] [0.76353881]|
|    |
|[False] [0.64580961]|
|    |
|[True] [0.31279863]|



**Output 4:**

<p align="center">
  <img width="500" src="https://user-images.githubusercontent.com/66559983/123520261-7c391200-d6e2-11eb-8eb2-fe0b04afe3b6.png" alt="Training 4">
</p>


<p align="center">
Figure 7: Program output after a student, Syaqirah compare her face with her own face and other students.
</p>

<br /> 

- Syaqirah got her own face as the lowest face distance, which means that the system is accurate.

<br />

_____________________________________________________________



However, sometimes it shown a wrong name to the face. To overcome this situation:

:bulb: Gather more faces of every student to help recognize the faces more accurate.

:bulb: Get faces of every student with different side of face to each of the photos.

:bulb: Implementing better training of the model

By these given results, we hope that our model will generalize every face recognize at the webcam outside our training and testing.

<br /> <br /> 
_____________________________________________________________
## 6.0  RESULT AND CONCLUSION

Detecting face attandance with OpenCV in real-time

You can then launch the face recognition attendance detector in real-time video streams using the following command:


|                                        $ python AttendanceProject.py                                            |
|-----------------------------------------------------------------------------------------------------------------|



<p align="center">
  <img width="500" src="https://user-images.githubusercontent.com/66559983/123453322-fc437700-d611-11eb-9ee9-fc4d5eed2fac.png" alt="Result">
</p>

<p align="center">
Figure 11: Face Recognition Attendance in real-time video streams.
</p>

<br /> <br />

In Figure 11, you can see that our Face Recognition Attendance detector is capable of running in real-time (and is correct in its predictions as well.)

<br /> <br /> 
_____________________________________________________________
## 7.0   PROJECT PRESENTATION 

In this project, creating a face attandance detector using OpenCV, Keras/TensorFlow, and Deep Learning is very important to learn and expert. <br /> <br />


:computer_mouse: OpenCV 

Open Source Computer Vision Library is an open source computer vision and machine learning software library. OpenCV was built to provide a common infrastructure for computer vision applications and to accelerate the use of machine perception in the commercial products. Being a BSD-licensed product, OpenCV makes it easy for businesses to utilize and modify the code. It has C++, Python, Java and MATLAB interfaces and supports Windows, Linux, Android and Mac OS. <br /> <br />


:computer_mouse: Computer Vision

Computer Vision, often abbreviated as CV, is a field of artificial intelligence (AI) that enables computers and systems to derive meaningful information from digital images, videos and other visual inputs — and take actions or make recommendations based on that information. If AI enables computers to think, computer vision enables them to see, observe and understand. <br /> <br />


We then took this face attandance classifier and applied it to both images and real-time video streams by:

:trackball: Detecting faces in images/video

:trackball: Extracting each individual face

:trackball: Applying our face attandance classifier <br /> <br />



Our face attandance detector is accurate, and since we used the MobileNetV2 architecture, it is also computationally efficient, making it easier to deploy the model to embedded systems (Raspberry Pi, Google Coral, Jetosn, Nano, etc.).

This is the link of video presentation from our group : 

[![Presentation](https://user-images.githubusercontent.com/66559983/123519312-f2d31100-d6dc-11eb-86af-0802d9a021b7.PNG)](https://youtu.be/pbvcZpHbYLA "Group H - Face Recognition Attendance System (Presentation & Demo)")






