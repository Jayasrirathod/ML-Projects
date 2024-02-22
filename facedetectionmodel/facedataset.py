import cv2
import os

cam=cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)


facedetector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


faceid=input('\n Enter user id  and  press enter')
print("\n [INFO] Initializing face Capturing .")

count=0

while(True):
    ret, img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=facedetector.detectMultiScale(gray,1.3,5)
    
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        count+=1
        
        cv2.imwrite("dataset/user." + str(faceid) + '-' + str(count) + ".jpg", gray[y:y+h, x:x+w])

        
        cv2.imshow('image',img)
        
        
        
    k = cv2.waitKey(10) & 0xff
    if k==20:
        break
    elif count >=25:
        break
    print("\n [INFO] exiting program")
    cam.release
    cv2.destroyAllWindows()

