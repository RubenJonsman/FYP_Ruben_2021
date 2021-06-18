from scipy.spatial import distance
from imutils import face_utils
import cv2
import imutils
import dlib
from sound import App
import numpy as np
from time import sleep


sound = "C:/Users/ruben/PycharmProjects/SOP/ree (2).mp3"
thresh = 0.25
frame_check = 7
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("C:/Users/ruben/PycharmProjects/SOP/shape_predictor_68_face_landmarks.dat")


def get_landmarks(im):
    rects = detect(im, 0)

    if len(rects) > 1:
        return "error"
    if len(rects) == 0:
        return "error"
    return np.matrix([[p.x, p.y] for p in predict(im, rects[0]).parts()])


def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im


def top_lip(landmarks):
    top_lip_pts = []
    for i in range(50, 53):
        top_lip_pts.append(landmarks[i])
    for i in range(61, 64):
        top_lip_pts.append(landmarks[i])
    top_lip_mean = np.mean(top_lip_pts, axis=0)
    return int(top_lip_mean[:, 1])


def bottom_lip(landmarks):
    bottom_lip_pts = []
    for i in range(65, 68):
        bottom_lip_pts.append(landmarks[i])
    for i in range(56, 59):
        bottom_lip_pts.append(landmarks[i])
    bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
    return int(bottom_lip_mean[:, 1])


def mouth_open(image):
    landmarks = get_landmarks(image)

    if landmarks == "error":
        return image, 0

    image_with_landmarks = annotate_landmarks(image, landmarks)
    top_lip_center = top_lip(landmarks)
    bottom_lip_center = bottom_lip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center)
    return image_with_landmarks, lip_distance

    # cv2.imshow('Result', image_with_landmarks)
    # cv2.imwrite('image_with_landmarks.jpg',image_with_landmarks)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def eye_aspect_ratio(eye):
    a = distance.euclidean(eye[1], eye[5])
    b = distance.euclidean(eye[2], eye[4])
    c = distance.euclidean(eye[0], eye[3])
    ear = (a + b) / (2.0 * c)
    return ear


(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mouthStart, mouthEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["inner_mouth"]
(ebLeftStart, ebLeftEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eyebrow"]
(ebRightStart, ebRightEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eyebrow"]
(noseStart, noseEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["nose"]
cap = cv2.VideoCapture(0)
flag = 0

yawns = 0
yawn_status = False

while True:
    ret, frame = cap.read()
    image_landmarks, lip_distance = mouth_open(frame)

    prev_yawn_status = yawn_status

    if lip_distance > 25:
        yawn_status = True

        cv2.putText(frame, " Subject is Yawning", (0, 400), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        output_text = " Yawn Count: " + str(yawns + 1)
        App()
        sleep(1)

        cv2.putText(frame, output_text, (0, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)

    else:
        yawn_status = False

    if prev_yawn_status is True and yawn_status is False:
        yawns += 1

    frame = imutils.resize(frame, width=450, height=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)  # converting to NumPy Array
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        # leftEyebrow = shape[ebLeftStart:ebLeftEnd]
        # rightEyebrow = shape[ebRightStart:ebRightEnd]
        # nose = shape[noseStart:noseEnd]
        # mouth = shape[mouthStart:mouthEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        # leftEyeHull = cv2.convexHull(leftEye)
        # rightEyeHull = cv2.convexHull(rightEye)
        # cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)  # image, contours, contourIdx, color
        # cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        # cv2.drawContours(frame, [mouth], -1, (0, 255, 0), 1)
        # cv2.drawContours(frame, [leftEyebrow], -1, (0, 255, 0), 1)
        # cv2.drawContours(frame, [rightEyebrow], -1, (0, 255, 0), 1)
        # cv2.drawContours(frame, [nose], -1, (0, 255, 0), 1)
        if ear < thresh:
            flag += 1
            # print(flag)
            if flag >= frame_check:
                cv2.putText(frame, "*****************ALERT!*****************", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "*****************ALERT!*****************", (5, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # print("Drowsy")
                App()

        else:
            flag = 0

   # cv2.imshow('Live Landmarks', image_landmarks)
   # cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
   # key = input()
    if key == 13:
        break
cv2.destroyAllWindows()
