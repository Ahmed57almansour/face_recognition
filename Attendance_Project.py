from importlib.resources import path
import cv2
from cv2 import VideoCapture
from cv2 import imshow
from cv2 import waitKey
import numpy as ny
import face_recognition
import os
from datetime import datetime
path = 'Attendance_Project'
images = []
classnames=[]
mylist = os.listdir(path)
print(mylist)
for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images .append(curImg)
    classnames.append(os.path.splitext(cl)[0])
    print(classnames)
    def findEncodings(images):
        encodeList=[]
        for img in images:
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            encode=face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        return encodeList
    

    def markAttendanc(name):
        with open('attendance.csv','r+') as f:
            MyDataList=f.readlines()
            nameList=[]
            for line in MyDataList:
                entry=line.split(',')
                nameList.append(entry[0])
            if name not in nameList:
                now=datetime.now()
                dtString=now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{dtString}')





encodeListKnown = findEncodings(images)
print(len(encodeListKnown))
print('Encode Complete')
cap=VideoCapture('rtsp://admin:Acc12345@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0')

while True:
    success ,img=cap.read()
    height, width, layers = img.shape
    newHeight = int(height/2)
    newWidth = int(width/2)
    imgs=cv2.resize(img,(newWidth,newHeight),None,0.25,0.25)
    img=cv2.resize(img,(newWidth,newHeight),None,0.25,0.25)
    imgs=cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)
    faceCurFrame=face_recognition.face_locations(imgs)
    encodeCurFrame=face_recognition.face_encodings(imgs,faceCurFrame)
    for encodeface,faceloc in zip(encodeCurFrame,faceCurFrame):
        matches=face_recognition.compare_faces(encodeListKnown,encodeface)
        faceDis=face_recognition.face_distance(encodeListKnown,encodeface)
        print (faceDis)
        matchIndex = ny.argmin(faceDis)

        if matches [matchIndex]:
            name = classnames[matchIndex].upper()
            print (name)
            y1,x2,y2,x1=faceloc
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),1)
            cv2.rectangle(img,(x1,y2-34),(x2,y2),(200,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendanc(name)

    cv2.imshow('Webcam',img)
    cv2.waitKey(1)
