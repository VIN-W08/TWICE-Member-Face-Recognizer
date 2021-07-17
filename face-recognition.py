import cv2
import os 
import numpy as np

face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")

face_list = []
label_list = []

# detect face and train face recognizer
for label_idx, person_name in enumerate(os.listdir("./dataset/train")):
    for idx, img_name in enumerate(os.listdir("./dataset/train/"+person_name)):
        img_gray = cv2.imread("./dataset/train/"+person_name+'/'+img_name, 0)

        face_detected = face_cascade.detectMultiScale(
            img_gray,
            scaleFactor = 1.7,
            minNeighbors = 5
        )
    
        for (x,y,w,h) in face_detected:
            face = img_gray[y:y+h, x:x+h]
            face_list.append(face)
            label_list.append(label_idx)

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(face_list, np.array(label_list))

# detect face and predict face
person_name = os.listdir("./dataset/train")  
for idx, img_name in enumerate(os.listdir("./dataset/test")):
    img = cv2.imread("./dataset/test/"+img_name)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    face_detected = face_cascade.detectMultiScale(
        img_gray,
        scaleFactor = 1.6,
        minNeighbors = 5
    )

    for (x,y,w,h) in face_detected:
        face_img = img_gray[y:y+h, x:x+w]
        result, confidence = face_recognizer.predict(face_img)
        text = person_name[result] + ' ' + str(np.ceil(confidence)) + '%'
        cv2.putText(img, text, (x-60,y-10), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        
    cv2.imshow(text,img)
    cv2.waitKey(0)

