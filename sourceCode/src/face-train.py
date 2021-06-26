import cv2
import face_recognition

#get a face of Hamizah to compare with 4 students
#imgMiza = face_recognition.load_image_file('Dataset/Hamizah/Miza.jpg')
#imgMiza = cv2.cvtColor(imgMiza, cv2.COLOR_BGR2RGB)
#imgMizaT = face_recognition.load_image_file('Dataset/Hamizah/Miza2.jpg')
#imgMizaT = cv2.cvtColor(imgMizaT, cv2.COLOR_BGR2RGB)
#imgMirahT = face_recognition.load_image_file('Dataset/Amirah/Mirah2.jpg')
#imgMirahT = cv2.cvtColor(imgMirahT, cv2.COLOR_BGR2RGB)
#imgDayahT = face_recognition.load_image_file('Dataset/Hidayati/Dayah2.jpg')
#imgDayahT = cv2.cvtColor(imgDayahT, cv2.COLOR_BGR2RGB)
#imgSyaqT = face_recognition.load_image_file('Dataset/Syaqirah/Syaqirah2.jpg')
#imgSyaqT = cv2.cvtColor(imgSyaqT, cv2.COLOR_BGR2RGB)

##get a face of Amirah to compare with 4 students
#imgMirah = face_recognition.load_image_file('Dataset/Amirah/Mirah.jpg')
#imgMirah = cv2.cvtColor(imgMirah, cv2.COLOR_BGR2RGB)
#imgMizaT = face_recognition.load_image_file('Dataset/Hamizah/Miza4.jpg')
#imgMizaT = cv2.cvtColor(imgMizaT, cv2.COLOR_BGR2RGB)
#imgMirahT = face_recognition.load_image_file('Dataset/Amirah/Mirah4.jpg')
#imgMirahT = cv2.cvtColor(imgMirahT, cv2.COLOR_BGR2RGB)
#imgDayahT = face_recognition.load_image_file('Dataset/Hidayati/Dayah4.jpg')
#imgDayahT = cv2.cvtColor(imgDayahT, cv2.COLOR_BGR2RGB)
#imgSyaqT = face_recognition.load_image_file('Dataset/Syaqirah/Syaqirah4.jpg')
#imgSyaqT = cv2.cvtColor(imgSyaqT, cv2.COLOR_BGR2RGB)

##get a face of Hidayati to compare
#imgDayah = face_recognition.load_image_file('Dataset/Hidayati/Dayah.jpg')
#imgDayah = cv2.cvtColor(imgDayah, cv2.COLOR_BGR2RGB)
#imgMizaT = face_recognition.load_image_file('Dataset/Hamizah/Miza3.jpg')
#imgMizaT = cv2.cvtColor(imgMizaT, cv2.COLOR_BGR2RGB)
#imgMirahT = face_recognition.load_image_file('Dataset/Amirah/Mirah3.jpg')
#imgMirahT = cv2.cvtColor(imgMirahT, cv2.COLOR_BGR2RGB)
#imgDayahT = face_recognition.load_image_file('Dataset/Hidayati/Dayah3.jpg')
#imgDayahT = cv2.cvtColor(imgDayahT, cv2.COLOR_BGR2RGB)
#imgSyaqT = face_recognition.load_image_file('Dataset/Syaqirah/Syaqirah3.jpg')
#imgSyaqT = cv2.cvtColor(imgSyaqT, cv2.COLOR_BGR2RGB)

#get a face of Syaqirah to compare
imgSyaq = face_recognition.load_image_file('Dataset/Syaqirah/Syaqirah.jpg')
imgSyaq = cv2.cvtColor(imgSyaq, cv2.COLOR_BGR2RGB)
imgMizaT = face_recognition.load_image_file('Dataset/Hamizah/Miza1.jpg')
imgMizaT = cv2.cvtColor(imgMizaT, cv2.COLOR_BGR2RGB)
imgMirahT = face_recognition.load_image_file('Dataset/Amirah/Mirah1.jpg')
imgMirahT = cv2.cvtColor(imgMirahT, cv2.COLOR_BGR2RGB)
imgDayahT = face_recognition.load_image_file('Dataset/Hidayati/Dayah1.jpg')
imgDayahT = cv2.cvtColor(imgDayahT, cv2.COLOR_BGR2RGB)
imgSyaqT = face_recognition.load_image_file('Dataset/Syaqirah/Syaqirah1.jpg')
imgSyaqT = cv2.cvtColor(imgSyaqT, cv2.COLOR_BGR2RGB)

#get the face location and encoding of Hamizah
#faceLocMiza = face_recognition.face_locations(imgMiza)[0]
#encodeMiza = face_recognition.face_encodings(imgMiza)[0]
#cv2.rectangle(imgMiza, (faceLocMiza[3], faceLocMiza[0]), (faceLocMiza[1], faceLocMiza[2]), (255, 0, 255), 2)

#compare faces of Hamizah with others using face distance
#faceLocMizaT = face_recognition.face_locations(imgMizaT)[0]
#encodeMizaT = face_recognition.face_encodings(imgMizaT)[0]
#cv2.rectangle(imgMizaT, (faceLocMizaT[3], faceLocMizaT[0]), (faceLocMizaT[1], faceLocMizaT[2]), (255, 0, 255), 2)
#results = face_recognition.compare_faces([encodeMiza], encodeMizaT)
#faceDis = face_recognition.face_distance([encodeMiza], encodeMizaT)
#print(results, faceDis)
#cv2.putText(imgMizaT, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
#faceLocMirahT = face_recognition.face_locations(imgMirahT)[0]
#encodeMirahT = face_recognition.face_encodings(imgMirahT)[0]
#cv2.rectangle(imgMirahT, (faceLocMirahT[3], faceLocMirahT[0]), (faceLocMirahT[1], faceLocMirahT[2]), (255, 0, 255), 2)
#results = face_recognition.compare_faces([encodeMiza], encodeMirahT)
#faceDis = face_recognition.face_distance([encodeMiza], encodeMirahT)
#print(results, faceDis)
#cv2.putText(imgMirahT, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
#faceLocDayahT = face_recognition.face_locations(imgDayahT)[0]
#encodeDayahT = face_recognition.face_encodings(imgDayahT)[0]
#cv2.rectangle(imgDayahT, (faceLocDayahT[3], faceLocDayahT[0]), (faceLocDayahT[1], faceLocDayahT[2]), (255, 0, 255), 2)
#results = face_recognition.compare_faces([encodeMiza], encodeDayahT)
#faceDis = face_recognition.face_distance([encodeMiza], encodeDayahT)
#print(results, faceDis)
#cv2.putText(imgDayahT, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
#faceLocSyaqT = face_recognition.face_locations(imgSyaqT)[0]
#encodeSyaqT = face_recognition.face_encodings(imgSyaqT)[0]
#cv2.rectangle(imgSyaqT, (faceLocSyaqT[3], faceLocSyaqT[0]), (faceLocSyaqT[1], faceLocSyaqT[2]), (255, 0, 255), 2)
#results = face_recognition.compare_faces([encodeMiza], encodeSyaqT)
#faceDis = face_recognition.face_distance([encodeMiza], encodeSyaqT)
#print(results, faceDis)
#cv2.putText(imgSyaqT, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

#show the result of Hamizah's comparison
#cv2.imshow('Hamizah', imgMiza)
#cv2.imshow('HamizahTest', imgMizaT)
#cv2.imshow('AmirahTest', imgMirahT)
#cv2.imshow('HidayatiTest', imgDayahT)
#cv2.imshow('SyaqirahTest', imgSyaqT)

#get the face location and encoding of Amirah
#faceLocMirah = face_recognition.face_locations(imgMirah)[0]
#encodeMirah = face_recognition.face_encodings(imgMirah)[0]
#cv2.rectangle(imgMirah, (faceLocMirah[3], faceLocMirah[0]), (faceLocMirah[1], faceLocMirah[2]), (255, 0, 255), 2)

##compare faces of Amirah with others using face distance
#faceLocMizaT = face_recognition.face_locations(imgMizaT)[0]
#encodeMizaT = face_recognition.face_encodings(imgMizaT)[0]
#cv2.rectangle(imgMizaT, (faceLocMizaT[3], faceLocMizaT[0]), (faceLocMizaT[1], faceLocMizaT[2]), (255, 0, 255), 2)
#results = face_recognition.compare_faces([encodeMirah], encodeMizaT)
#faceDis = face_recognition.face_distance([encodeMirah], encodeMizaT)
#print(results, faceDis)
#cv2.putText(imgMizaT, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
#faceLocMirahT = face_recognition.face_locations(imgMirahT)[0]
#encodeMirahT = face_recognition.face_encodings(imgMirahT)[0]
#cv2.rectangle(imgMirahT, (faceLocMirahT[3], faceLocMirahT[0]), (faceLocMirahT[1], faceLocMirahT[2]), (255, 0, 255), 2)
#results = face_recognition.compare_faces([encodeMirah], encodeMirahT)
#faceDis = face_recognition.face_distance([encodeMirah], encodeMirahT)
#print(results, faceDis)
#cv2.putText(imgMirahT, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
#faceLocDayahT = face_recognition.face_locations(imgDayahT)[0]
#encodeDayahT = face_recognition.face_encodings(imgDayahT)[0]
#cv2.rectangle(imgDayahT, (faceLocDayahT[3], faceLocDayahT[0]), (faceLocDayahT[1], faceLocDayahT[2]), (255, 0, 255), 2)
#results = face_recognition.compare_faces([encodeMirah], encodeDayahT)
#faceDis = face_recognition.face_distance([encodeMirah], encodeDayahT)
#print(results, faceDis)
#cv2.putText(imgDayahT, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
#faceLocSyaqT = face_recognition.face_locations(imgSyaqT)[0]
#encodeSyaqT = face_recognition.face_encodings(imgSyaqT)[0]
#cv2.rectangle(imgSyaqT, (faceLocSyaqT[3], faceLocSyaqT[0]), (faceLocSyaqT[1], faceLocSyaqT[2]), (255, 0, 255), 2)
#results = face_recognition.compare_faces([encodeMirah], encodeSyaqT)
#faceDis = face_recognition.face_distance([encodeMirah], encodeSyaqT)
#print(results, faceDis)
#cv2.putText(imgSyaqT, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

##show the result of Amirah's comparison
#cv2.imshow('Amirah', imgMirah)
#cv2.imshow('HamizahTest', imgMizaT)
#cv2.imshow('AmirahTest', imgMirahT)
#cv2.imshow('HidayatiTest', imgDayahT)
#cv2.imshow('SyaqirahTest', imgSyaqT)

##get the face location and encoding of Hidayati
#faceLocDayah = face_recognition.face_locations(imgDayah)[0]
#encodeDayah = face_recognition.face_encodings(imgDayah)[0]
#cv2.rectangle(imgDayah, (faceLocDayah[3], faceLocDayah[0]), (faceLocDayah[1], faceLocDayah[2]), (255, 0, 255), 2)

##compare faces of Hidayati with others using face distance
#faceLocMizaT = face_recognition.face_locations(imgMizaT)[0]
#encodeMizaT = face_recognition.face_encodings(imgMizaT)[0]
#cv2.rectangle(imgMizaT, (faceLocMizaT[3], faceLocMizaT[0]), (faceLocMizaT[1], faceLocMizaT[2]), (255, 0, 255), 2)
#results = face_recognition.compare_faces([encodeDayah], encodeMizaT)
#faceDis = face_recognition.face_distance([encodeDayah], encodeMizaT)
#print(results, faceDis)
#cv2.putText(imgMizaT, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
#faceLocMirahT = face_recognition.face_locations(imgMirahT)[0]
#encodeMirahT = face_recognition.face_encodings(imgMirahT)[0]
#cv2.rectangle(imgMirahT, (faceLocMirahT[3], faceLocMirahT[0]), (faceLocMirahT[1], faceLocMirahT[2]), (255, 0, 255), 2)
#results = face_recognition.compare_faces([encodeDayah], encodeMirahT)
#faceDis = face_recognition.face_distance([encodeDayah], encodeMirahT)
#print(results, faceDis)
#cv2.putText(imgMirahT, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
#faceLocDayahT = face_recognition.face_locations(imgDayahT)[0]
#encodeDayahT = face_recognition.face_encodings(imgDayahT)[0]
#cv2.rectangle(imgDayahT, (faceLocDayahT[3], faceLocDayahT[0]), (faceLocDayahT[1], faceLocDayahT[2]), (255, 0, 255), 2)
#results = face_recognition.compare_faces([encodeDayah], encodeDayahT)
#faceDis = face_recognition.face_distance([encodeDayah], encodeDayahT)
#print(results, faceDis)
#cv2.putText(imgDayahT, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
#faceLocSyaqT = face_recognition.face_locations(imgSyaqT)[0]
#encodeSyaqT = face_recognition.face_encodings(imgSyaqT)[0]
#cv2.rectangle(imgSyaqT, (faceLocSyaqT[3], faceLocSyaqT[0]), (faceLocSyaqT[1], faceLocSyaqT[2]), (255, 0, 255), 2)
#results = face_recognition.compare_faces([encodeDayah], encodeSyaqT)
#faceDis = face_recognition.face_distance([encodeDayah], encodeSyaqT)
#print(results, faceDis)
#cv2.putText(imgSyaqT, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

##show the result of Hidayati's comparison
#cv2.imshow('Hidayati', imgDayah)
#cv2.imshow('HamizahTest', imgMizaT)
#cv2.imshow('AmirahTest', imgMirahT)
#cv2.imshow('HidayatiTest', imgDayahT)
#cv2.imshow('SyaqirahTest', imgSyaqT)

#get the face location and encoding of Syaqirah
faceLocSyaq = face_recognition.face_locations(imgSyaq)[0]
encodeSyaq = face_recognition.face_encodings(imgSyaq)[0]
cv2.rectangle(imgSyaq, (faceLocSyaq[3], faceLocSyaq[0]), (faceLocSyaq[1], faceLocSyaq[2]), (255, 0, 255), 2)

#compare faces of Syaqirah with others using face distance
faceLocMizaT = face_recognition.face_locations(imgMizaT)[0]
encodeMizaT = face_recognition.face_encodings(imgMizaT)[0]
cv2.rectangle(imgMizaT, (faceLocMizaT[3], faceLocMizaT[0]), (faceLocMizaT[1], faceLocMizaT[2]), (255, 0, 255), 2)
results = face_recognition.compare_faces([encodeSyaq], encodeMizaT)
faceDis = face_recognition.face_distance([encodeSyaq], encodeMizaT)
print(results, faceDis)
cv2.putText(imgMizaT, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
faceLocMirahT = face_recognition.face_locations(imgMirahT)[0]
encodeMirahT = face_recognition.face_encodings(imgMirahT)[0]
cv2.rectangle(imgMirahT, (faceLocMirahT[3], faceLocMirahT[0]), (faceLocMirahT[1], faceLocMirahT[2]), (255, 0, 255), 2)
results = face_recognition.compare_faces([encodeSyaq], encodeMirahT)
faceDis = face_recognition.face_distance([encodeSyaq], encodeMirahT)
print(results, faceDis)
cv2.putText(imgMirahT, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
faceLocDayahT = face_recognition.face_locations(imgDayahT)[0]
encodeDayahT = face_recognition.face_encodings(imgDayahT)[0]
cv2.rectangle(imgDayahT, (faceLocDayahT[3], faceLocDayahT[0]), (faceLocDayahT[1], faceLocDayahT[2]), (255, 0, 255), 2)
results = face_recognition.compare_faces([encodeSyaq], encodeDayahT)
faceDis = face_recognition.face_distance([encodeSyaq], encodeDayahT)
print(results, faceDis)
cv2.putText(imgDayahT, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
faceLocSyaqT = face_recognition.face_locations(imgSyaqT)[0]
encodeSyaqT = face_recognition.face_encodings(imgSyaqT)[0]
cv2.rectangle(imgSyaqT, (faceLocSyaqT[3], faceLocSyaqT[0]), (faceLocSyaqT[1], faceLocSyaqT[2]), (255, 0, 255), 2)
results = face_recognition.compare_faces([encodeSyaq], encodeSyaqT)
faceDis = face_recognition.face_distance([encodeSyaq], encodeSyaqT)
print(results, faceDis)
cv2.putText(imgSyaqT, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

#show the result of all the comparison
cv2.imshow('Syaqirah', imgSyaq)
cv2.imshow('HamizahTest', imgMizaT)
cv2.imshow('AmirahTest', imgMirahT)
cv2.imshow('HidayatiTest', imgDayahT)
cv2.imshow('SyaqirahTest', imgSyaqT)
cv2.waitKey(0)