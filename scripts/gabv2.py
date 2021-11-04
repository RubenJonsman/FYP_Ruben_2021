from scipy.spatial import distance
from imutils import face_utils
import cv2
import imutils
import dlib
from sound import App
import numpy as np

sound = "C:/Users/ruben/PycharmProjects/SOP/ree (2).mp3"
thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("C:/Users/ruben/PycharmProjects/SOP/shape_predictor_68_face_landmarks.dat")# Dat file is the crux of the code

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def mouth_aspect_ratio(mouth):
    mund = distance.euclidean(mouth[1], mouth[2])
    print(mund)
    return mund

(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
cap = cv2.VideoCapture(0)
flag = 0
yawns = 0
yawn_status = False




while True:

    prev_yawn_status = yawn_status
    ret, frame=cap.read()
    frame = imutils.resize(frame, width=450, height=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)  # converting to NumPy Array
        mouth = shape[mStart:mEnd]
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        mouthRatio = mouth_aspect_ratio(mouth)
        ear = (leftEAR + rightEAR) / 2.0
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1) #image, contours, contourIdx, color
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouthHull], -1, (0, 0, 255), 1)
        if ear < thresh:
            flag += 1
            #print (flag)
            if flag >= frame_check:
                cv2.putText(frame, "****************ALERT!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "****************ALERT!****************", (10,325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print ("Drowsy")
                App()
        else:
            flag = 0

        if mouthRatio < 8:
            yawn_status = True

            cv2.putText(frame, "Subject is Yawning", (50, 450),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

            output_text = " Yawn Count: " + str(yawns + 1)

            cv2.putText(frame, output_text, (50, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)

        else:
            yawn_status = False

        if prev_yawn_status == True and yawn_status == False:
            yawns += 1



    #cv2.imshow('Live Landmarks', image_landmarks)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()