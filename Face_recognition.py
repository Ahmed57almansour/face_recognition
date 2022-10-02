import cv2
import numpy as ny
import numpy
import face_recognition
imgelon1 = face_recognition.load_image_file('Image Basic/D')
imgelon = face_recognition.load_image_file('Image Basic/Elon')
imgelon = cv2.cvtColor(imgelon,cv2.COLOR_BGR2RGB)
imgelon1 = cv2.cvtColor(imgelon1,cv2.COLOR_BGR2RGB)
#cv2.imshow('Elon Mask0',imgelon1)
faceLoc=face_recognition.face_locations(imgelon1)[0]
encodeElon=face_recognition.face_encodings(imgelon1)[0]
cv2.rectangle(imgelon1,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(150,0,255),3)

faceLoc0=face_recognition.face_locations(imgelon)[0]
encodeElon0=face_recognition.face_encodings(imgelon)[0]
cv2.rectangle(imgelon,(faceLoc0[3],faceLoc0[0]),(faceLoc0[1],faceLoc0[2]),(150,0,255),3)
results=face_recognition.compare_faces([encodeElon],encodeElon0)
faceDis=face_recognition.face_distance([encodeElon],encodeElon0)
cv2.putText(imgelon,f'{results}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),5)

cv2.imshow('Elon Mask',imgelon1)
cv2.imshow('Elon Mask0',imgelon)
print(results,faceDis)



cv2.waitKey(0)
