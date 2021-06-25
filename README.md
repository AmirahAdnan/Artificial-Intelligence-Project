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


With the advancement of technologies each and every day, humanity is slowly going towards contactless everything. It is quite evident that the future ahead of us will become so much advance that maybe 90%+ things that we are doing right now will be either automated or become contactless. One such advancement will be the facial recognition technology or the FR tech, which is the prime focus of this article. 

Facial recognition technology is a system or software which is capable enough to identify an identity of family members, friends or anyone by analyzing an image or video footage. Some of the technologies or software are so advanced that even blurred pictures are sometimes rendered enough and analyzed to know the identity of the person.  In order to overcome human limitations on memorizing each of human face, this face recognition attendance system would be able to do a face recognition and store the image in database with high processing speed. <br /> <br />

In this article, our focus will be on one of the many applications of facial recognition technology, which is **Face Recognition Attendance System**.

<p align="center">
  <img width="700" src="https://user-images.githubusercontent.com/66559983/123439769-a4524380-d604-11eb-859f-689e8bd42dca.png" alt="example">
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
  <img width="700" src="https://user-images.githubusercontent.com/66559983/123443079-23954680-d608-11eb-8d2c-f7f7a96c0250.png" alt="Figure 2">
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
<br /> <br /> 
_____________________________________________________________
## 5.0   TRAINING THE FACE ATTANDANCE DETECTION

Let’s try this face recognition attendance out on some of our own images now.
We will run our script at PyCharm and webcam will prompt out:
Example of python command to execute & evaluate the student at the webcam based on the dataset :

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


_____________________________________________________________


**Output 1:** 

<p align="center">
  <img width="500" src="https://user-images.githubusercontent.com/66559983/123449307-1f6c2780-d60e-11eb-885d-dd215861ddb8.png" alt="Training 1">
</p>

<p align="center">
Figure 4: Program output after a student put their face at the webcam.
</p>


_____________________________________________________________


**Output 2:**

<p align="center">
  <img width="500" src="https://user-images.githubusercontent.com/66559983/123449352-2c891680-d60e-11eb-8bf9-3d594dd11399.png" alt="Training 2">
</p>

<p align="center">
Figure 5: Program output after a student put their face at the webcam.
</p>


_____________________________________________________________


**Output 3:**

<p align="center">
  <img width="500" src="https://user-images.githubusercontent.com/66559983/123449364-301c9d80-d60e-11eb-9f86-9d2fdcce0162.png" alt="Training 3">
</p>

<p align="center">
Figure 6: Program output after a student put their face at the webcam.
</p>

_____________________________________________________________


**Output 4:**

<p align="center">
  <img width="500" src="https://user-images.githubusercontent.com/66559983/123449378-327ef780-d60e-11eb-8758-158517732e23.png" alt="Training 4">
</p>

<p align="center">
Figure 7: Program output after a student put their face at the webcam.
</p>

_____________________________________________________________


**Output 5:**

<p align="center">
  <img width="500" src="https://user-images.githubusercontent.com/66559983/123449387-34e15180-d60e-11eb-951e-f06503cbe025.png" alt="Training 5">
</p>

<p align="center">
Figure 8: Program output after some students put their face at the same time at the webcam.
</p>

_____________________________________________________________


**Output 6:**

<p align="center">
  <img width="500" src="https://user-images.githubusercontent.com/66559983/123449394-36127e80-d60e-11eb-8f4d-a749cf6f4506.png" alt="Training 6">
</p>

<p align="center">
Figure 9: Program output after a student put their face while wearing face mask at the webcam.
</p>


_____________________________________________________________


**Output 7:**


<p align="center">
  <img width="500" src="https://user-images.githubusercontent.com/66559983/123459606-192f7880-d619-11eb-9493-1426e92d795f.png" alt="Training 7">
</p>

<p align="center">
Figure 10: Program output after stop the system which shown the attendance list on .csv file.
</p>

<br /> <br />

This output can also be open using Microsoft Excel.


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





