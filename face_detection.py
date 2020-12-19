from imutils import paths
import face_recognition
import numpy as np
import dlib
import argparse
import pickle
import cv2
import os
import sqlite3

def face_extractor(img):
        faceRects = face_recognition.face_locations(img)
        if len(faceRects) == 0:
            return None
    
        else:
            for (top, right, bottom, left) in faceRects:
                x1 = left
                y1 = top
                x2 = right
                y2 = bottom    
                cropped_face = img[y1:y2, x1:x2]
            return cropped_face
			
def getFace(id): 
    os.mkdir('./faces/user/'+str(id))
    folder = './faces/user/'+str(id)
    cap = cv2.VideoCapture(0)
    count = 0
    while True:

        ret, frame = cap.read()
        if face_extractor(frame) is not None:
            count += 1
            face = cv2.resize(face_extractor(frame), (640, 480))
            
            # Save file in specified directory with unique name
            file_name_path = folder +"/"+ str(count) + '.jpg'
            cv2.imwrite(file_name_path, face)

            # Put count on images and display live count
            cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Face Cropper', face)
        
        else:
            print("Face not found")
            pass

        if cv2.waitKey(1) == 13 or count == 50: #13 is the Enter Key
            break
        
    cap.release()
    cv2.destroyAllWindows()      
    print("Collecting Samples Complete")
	
def RegisterFace(participantCount):
    participantCount += 1
    name = input("Enter name: ")
    print("Participant's ID is "+str(participantCount))
    conn=sqlite3.connect("FACE_DB.db")
    cmd  = "INSERT INTO Participants(ID, Name) Values("+str(participantCount)+","+"\""+str(name)+"\""+")"
    conn.execute(cmd)
    conn.commit()
    conn.close()
    getFace(participantCount)
    return participantCount