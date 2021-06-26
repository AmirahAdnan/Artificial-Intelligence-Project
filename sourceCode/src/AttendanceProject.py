import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

#specify the path
path = 'ImagesAttendance'
#array that hold the images
images = []
#take name of the images
imageNames = []
#take list of images from the path
imageList = os.listdir(path)
print(imageList)
#insert data in the arrays
for cl in imageList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    imageNames.append(os.path.splitext(cl)[0])
print(imageNames)

#encode the images
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    #print(encodeList)
    return encodeList

#insert the data into attendance file
def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        dateList = []
        nameList = []
        now = datetime.now()
        dateString = now.strftime('%d/%b/%Y')
        timeString = now.strftime('%H:%M:%S')
        for line in myDataList:
            entryDate = line.split(',')
            dateList.append(entryDate[0])
        if dateString not in dateList:
           f.writelines(f'\n\n{dateString},')
        for line in myDataList:
            entryName = line.split(' > ')
            nameList.append(entryName[0])
        if name not in nameList:
           f.writelines(f'\n{name} - {timeString}')

encodeListKnown = findEncodings(images)
print('Encoding Complete')

#return video from the first webcam on your computer
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    #find the data that match
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        print(matches)
        matchIndex = np.argmin(faceDis)
        #print(matchIndex)
        print(faceDis)

        #set the name of the match data that appear on webcam and attendance
        if matches[matchIndex]:
            name = imageNames[matchIndex].upper()
            shortformname = imageNames[matchIndex].partition('- ')[2]
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,shortformname,(x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255),2)
            markAttendance(name)

    cv2.imshow('WebCam', img)
    cv2.waitKey(1)