import cv2 as cv
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
import mediapipe as mp
import numpy as np
import Rpi.GPIO as GPIO
import time
import threading

EACH_FRAME_DETECTION = False                 #Keep it True for first method and False for averaging the EARs method
NEED_CALIBRATION = True
EYE_CUTOFF = 26
MOUTH_CUTOFF = 50
AVG_LIST_LENGTH = 15
WINDOW_WIDTH,WINDOW_HEIGHT = 2560,1440
MIN_FRAME_FOR_BLINK_ALARM = 13
MIN_FRAME_FOR_YAWN_ALARM = 30
BLINK_THRESHOLD = 3
YAWN_THRESHOLD = 2
DOZE_THRESHOLD = 2
TIME_FRAME = 30
HEAD_POS_ANGLE = 10
LEFTEYE = [33,159,158,155,153,145]
RIGHTEYE = [463,385,386,263,374,380]
UPPERLIP = [72,11,302,82,13,312,61]
LOWERLIP = [87,14,317,85,16,315,306]

cap = cv.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)
[LOWERLIPCoord,UPPERLIPCoord,LEFTEYECoord,RIGHTEYECoord] = [[(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)],[(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)],[(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)],[(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]]
[yawnCounter,blinkCounter,dozeCounter,totalYawnCounter,totalBlinkCounter,totalDozeCounter,ratioList,mouthList,flagEye,flagMouth,flagDown,eyeCounter,mouthCounter,calibrated_eye,frame,seconds,start_time,stop_time] = [0,0,0,0,0,0,[],[],False,False,False,0,0,EYE_CUTOFF,0,0,0,0]

def buzzer(time,dur):
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(8,GPIO.OUT)
    for i in range(time):
        GPIO.output(8,True)
        time.sleep(dur)
        GPIO.output(8,False)
        time.sleep(dur)
    GPIO.cleanup()

def play_buzzer(duration, interval):
    buzzer_thread = threading.Thread(target=buzzer, args=(duration, interval))
    buzzer_thread.start()

mp_drawing = mp.solutions.drawing_utils
mp_facemesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence = 0.5,min_tracking_confidence=0.5)

def processOutput(results,im_h,im_w,x_calibrate,y_calibrate):
    text = "Not valid"
    [face_3d,face_2d] = [[],[]]
    if(results.multi_face_landmarks):
        for face_landmarks in results.multi_face_landmarks:
            for idx,lm in enumerate(face_landmarks.landmark):
                if(idx in [33,263,1,61,291,199]):
                    x,y = int(lm.x*im_w),int(lm.y*im_h)
                    face_2d.append([x,y])
                    face_3d.append([x,y,lm.z])
        face_2d = np.array(face_2d).astype("float32")
        face_3d = np.array(face_3d).astype("float32")
        focal_length = 1 * im_w
    
        cam_mat = np.array([[focal_length,0,im_h/2],[0,focal_length,im_w/2],[0,0,1]])
        dist_mat = np.zeros((4,1),dtype = np.float32)
    
        _,rot_v,_ = cv.solvePnP(face_3d,face_2d,cam_mat,dist_mat)
        rmat,_ = cv.Rodrigues(rot_v)
    
        angles,_,_,_,_,_ = cv.RQDecomp3x3(rmat)
        x = angles[0]*360
        y = angles[1]*360
        if (y>y_calibrate-HEAD_POS_ANGLE and y<y_calibrate + HEAD_POS_ANGLE):
            if(x>x_calibrate-HEAD_POS_ANGLE and x<x_calibrate + HEAD_POS_ANGLE):
                text = "Valid"
            elif(x<x_calibrate-HEAD_POS_ANGLE):
                text = "Down"
    else:
        [x,y] = 0,0
        text = "Not Valid"
    return (True,x,y,text) if text=="Valid" else (False,x,y,text)

def calcEAR(l):
    a = detector.findDistance(l[1],l[5])
    b = detector.findDistance(l[2],l[4])
    c = detector.findDistance(l[0],l[3])
    return ((a[0]+b[0])/(2*c[0]))*100

def calcYawn(u,l):
    [ux,uy,lx,ly] = [0,0,0,0]
    distH = detector.findDistance(u.pop(),l.pop())
    for i,j in zip(u,l):
        ux += i[0]
        uy += i[1]
        lx += j[0]
        ly += j[1]

    ux,uy,lx,ly = [i/6 for i in [ux,uy,lx,ly]]
    distV = detector.findDistance((ux,uy),(lx,ly))
    return (distV[0]/distH[0])*100

def calibrate(text):
    while cap.isOpened():
        _,img = cap.read()
        img,faces = detector.findFaceMesh(img,draw=False)
        if faces:
            face = faces[0]
            for i in range(6):
                LEFTEYECoord[i] = face[LEFTEYE[i]]
                RIGHTEYECoord[i] = face[RIGHTEYE[i]]

            leftEAR = calcEAR(LEFTEYECoord)
            rightEAR = calcEAR(RIGHTEYECoord)
            ear = (leftEAR+rightEAR)//2

            ratioList.append(ear)
            if len(ratioList) > AVG_LIST_LENGTH:
                ratioList.pop(0)
            ear = sum(ratioList)//len(ratioList)
            cvzone.putTextRect(img,"Calibrating now... press c to stop calibration",(50,50),scale=2,thickness=1,colorR=(0,0,0),colorT=(255,255,255))
            cvzone.putTextRect(img,text,(50,100),scale=2,thickness=1,colorR=(0,0,0),colorT=(255,255,255))
            cvzone.putTextRect(img,str(int(ear)),(50,150),scale=2,thickness=1,colorR=(0,0,0),colorT=(255,255,255))

            if cv.waitKey(10) & 0xFF == ord('c'):
                results = mp_facemesh.process(img)
                im_h,im_w,_ = img.shape
                _,x,y,_ = processOutput(results,im_h,im_w,0,0)
                cv.destroyAllWindows()
                return ear,int(x),int(y)
        cv.imshow("Calibrating",img)

def callCalibrate():
    calibrated_eye_close,x1,y1 = calibrate("Close your eyes")
    calibrated_eye_open,x2,y2 = calibrate("Open your eyes")
    [calibrated_x,calibrated_y] = [round(((x1+x2)/2),2),round(((y1+y2)/2),2)]
    calibrated_eye = (calibrated_eye_open+calibrated_eye_close)//2
    calibrated_eye_range = (calibrated_eye_open-calibrated_eye_close)//2
    print(f"Calibrated eye cutoff as {calibrated_eye} and range as {calibrated_eye_range} and angles as {calibrated_x,calibrated_y}")
    return calibrated_eye_close,calibrated_eye_open,calibrated_x,calibrated_y,calibrated_eye,calibrated_eye_range

[calibrated_eye_close,calibrated_eye_open,calibrated_x,calibrated_y,calibrated_eye,calibrated_eye_range] = callCalibrate() if NEED_CALIBRATION else [0,0,0,0,EYE_CUTOFF,4]

start_time = time.time()
while cap.isOpened():
    frame += 1
    suc,img = cap.read()
    results = mp_facemesh.process(img)
    im_h,im_w,_ = img.shape
    forward,_,_,text = processOutput(results,im_h,im_w,calibrated_x,calibrated_y)
    textDown = text == "Down"
    if(forward):
        flagDown = False
        img,faces = detector.findFaceMesh(img,draw=False)
        if faces:
            face = faces[0]
            for i in range(7):
                LOWERLIPCoord[i] = face[LOWERLIP[i]]
                UPPERLIPCoord[i] = face[UPPERLIP[i]]
                if i != 6:
                    LEFTEYECoord[i] = face[LEFTEYE[i]]
                    RIGHTEYECoord[i] = face[RIGHTEYE[i]]

            mouth = int(calcYawn(UPPERLIPCoord,LOWERLIPCoord))
            leftEAR = calcEAR(LEFTEYECoord)
            rightEAR = calcEAR(RIGHTEYECoord)
            ear = (leftEAR+rightEAR)//2

            eyeClosed = ear<=calibrated_eye
            mouthOpen = mouth>=MOUTH_CUTOFF

            if EACH_FRAME_DETECTION:
                if eyeClosed and not flagEye:
                    eyeCounter += 1
                    blinkCounter += 1                                       #increment here for actual blink count
                    flagEye = True
                elif (ear>= calibrated_eye+calibrated_eye_range and flagEye) or (eyeCounter>60):
                    eyeCounter = 0
                    flagEye = False
                elif eyeClosed and flagEye:
                    eyeCounter += 1
                    # if eyeCounter==MIN_FRAME_FOR_BLINK_ALARM:
                    #     blinkCounter += 1                                   #increment here for blinks longer than certain time

                if mouthOpen and not flagMouth:
                    flagMouth = True
                    mouthCounter += 1
                elif mouthOpen and flagMouth:
                    mouthCounter += 1
                    if mouthCounter == MIN_FRAME_FOR_YAWN_ALARM:
                        yawnCounter += 1
                elif not mouthOpen and flagMouth:
                    mouthCounter = 0
                    flagMouth = False
            else:
                ratioList.append(ear)
                mouthList.append(mouth)
                if len(ratioList) > AVG_LIST_LENGTH:
                    ratioList.pop(0)
                    if len(mouthList) > AVG_LIST_LENGTH*2 : mouthList.pop(0)
                ear = sum(ratioList)/len(ratioList)
                mouth = sum(mouthList)/len(mouthList)

                eyeClosed = ear<=calibrated_eye
                mouthOpen = mouth>=MOUTH_CUTOFF

                if eyeClosed and not flagEye:
                    blinkCounter += 1
                    flagEye = True
                elif ear >= calibrated_eye+3 and flagEye:
                    flagEye = False

                if mouthOpen and not flagMouth:
                    yawnCounter += 1
                    flagMouth = True
                elif not mouthOpen and flagMouth:
                    flagMouth = False

            cvzone.putTextRect(img,f"Blinks : {blinkCounter}",(50,50),2,colorT=(0,0,0),colorR=(255,255,255))
            cvzone.putTextRect(img,f"Yawn : {yawnCounter}",(50,100),2,colorT=(0,0,0),colorR=(255,255,255))
            cvzone.putTextRect(img,f"Dozed off : {dozeCounter}",(50,150),2,colorT=(0,0,0),colorR=(255,255,255))
            cvzone.putTextRect(img,f"Time : {round((time.time()-start_time),2)}",(50,200),2,colorT=(0,0,0),colorR=(255,255,255))
            cvzone.putTextRect(img,f"Blink for {MIN_FRAME_FOR_BLINK_ALARM} frames under {calibrated_eye}",(350,50),1,thickness=1,colorR=(0,0,0),colorT=(255,255,255))
            cvzone.putTextRect(img,f"Yawn for {MIN_FRAME_FOR_YAWN_ALARM} frames over {MOUTH_CUTOFF}",(350,75),1,thickness=1,colorR=(0,0,0),colorT=(255,255,255))

            if((((round(time.time() - start_time,2))//TIME_FRAME - seconds) == 1) and not flagEye and not flagMouth):     #one time frame complete
                seconds += 1
                factors = 0
                if(blinkCounter >= BLINK_THRESHOLD): factors += 1
                if(yawnCounter >= YAWN_THRESHOLD): factors += 1
                if(dozeCounter >= DOZE_THRESHOLD): factors += 2
                if factors>= 2 : play_buzzer(10,3)
                if blinkCounter>= BLINK_THRESHOLD and yawnCounter<YAWN_THRESHOLD and dozeCounter<DOZE_THRESHOLD : play_buzzer(3,1)   #small duration alarm if only eye is drowsy
                print(f"Blinked {blinkCounter} times, yawned {yawnCounter} times and dozed off {dozeCounter} times in minute {seconds}")
                totalBlinkCounter += blinkCounter
                totalYawnCounter += yawnCounter
                totalDozeCounter += dozeCounter
                [blinkCounter,yawnCounter,dozeCounter] = [0,0,0]
    elif(textDown and not flagDown and flagEye):
        flagDown = True
        dozeCounter += 1

    cv.imshow("Result",img)
    [LOWERLIPCoord,UPPERLIPCoord,LEFTEYECoord,RIGHTEYECoord] = [[(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)],[(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)],[(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)],[(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]]
    key = cv.waitKey(10)
    if key & 0xFF == ord('c'):
        cv.destroyAllWindows()
        [calibrated_eye_close,calibrated_eye_open,calibrated_x,calibrated_y,calibrated_eye,calibrated_eye_range] = callCalibrate()
    elif key & 0xFF == ord('q'):
        break
cv.destroyAllWindows()
cap.release()
stop_time = time.time()

total_time = round((stop_time-start_time),2)
print(f"Completed run for {int(total_time/60)} minutes and {round(total_time%60 , 2)} seconds")
