import face_recognition
import numpy as np
import dlib
import cv2

def StartMonitoring(knownNames, knownEncodings):
    videoCapture = cv2.VideoCapture(0)
    faceLocations = []
    faceEncodings = []
    faceNames = []

    while True:
        # Grab a single frame of video
        ret, frame = videoCapture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        smallFrame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgbSmallFrame = smallFrame[:, :, ::-1]
        # Find all the faces and face encodings in the current frame of video
        faceLocations = face_recognition.face_locations(rgbSmallFrame)
        faceEncodings = face_recognition.face_encodings(rgbSmallFrame, faceLocations)

        faceNames = []
        for faceEncoding in faceEncodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(knownEncodings, faceEncoding)
            name = "Unknown"

            # # If a match was found in known_faceEncodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_faceNames[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            faceDistances = face_recognition.face_distance(knownEncodings, faceEncoding)
            bestMatchIndex = np.argmin(faceDistances)
            if matches[bestMatchIndex]:
                name = knownNames[bestMatchIndex]

            faceNames.append(name)


        # Display the results
        for (top, right, bottom, left), name in zip(faceLocations, faceNames):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
    
            if name == 'Unknown':
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            else:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    videoCapture.release()
    cv2.destroyAllWindows()