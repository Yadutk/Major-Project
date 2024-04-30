import cv2 as cv
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
# from gpiozero import Buzzer
import time

EACH_FRAME_DETECTION = True                 #Keep it True for first method and False for averaging the EARs method
NEED_CALIBRATION = False
EYE_CUTOFF = 26
MOUTH_CUTOFF = 50
AVG_LIST_LENGTH = 15
WINDOW_WIDTH,WINDOW_HEIGHT = 2560,1440
MIN_FRAME_FOR_BLINK_ALARM = 12
MIN_FRAME_FOR_YAWN_ALARM = 30
BLINK_THRESHOLD = 5
YAWN_THRESHOLD = 3
TIME_FRAME = 30
LEFTEYE = [33,159,158,155,153,145]
RIGHTEYE = [463,385,386,263,374,380]
UPPERLIP = [72,11,302,82,13,312,61]
LOWERLIP = [87,14,317,85,16,315,306]

cap = cv.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)
[LOWERLIPCoord,UPPERLIPCoord,LEFTEYECoord,RIGHTEYECoord] = [[(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)],[(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)],[(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)],[(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]]
[yawnCounter,blinkCounter,totalYawnCounter,totalBlinkCounter,flagEye,flagMouth,eyeCounter,mouthCounter,calibrated_eye,frame,seconds,start_time,stop_time] = [0,0,0,0,False,False,0,0,EYE_CUTOFF,0,0,0,0]
[ratioList,yawnList] = [[],[]]
# buzzer = Buzzer(17)

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
            ear = sum(ratioList)/len(ratioList)
            cvzone.putTextRect(img,"Calibrating now... press c to stop calibration",(50,50),scale=2,thickness=1,colorR=(0,0,0),colorT=(255,255,255))
            cvzone.putTextRect(img,text,(50,100),scale=2,thickness=1,colorR=(0,0,0),colorT=(255,255,255))
            cvzone.putTextRect(img,str(int(ear)),(50,150),scale=2,thickness=1,colorR=(0,0,0),colorT=(255,255,255))
            cv.imshow("Calibrating",img)

            if cv.waitKey(10) & 0xFF == ord('c'):
                print(f"Calibrated normal eye aspect ratio as {int(ear)}")
                cv.destroyAllWindows()
                return ear

calibrated_eye_open = calibrate("Open your eyes") if NEED_CALIBRATION else 0
calibrated_eye_close = calibrate("Close your eyes") if NEED_CALIBRATION else 0
calibrated_eye = (calibrated_eye_open+calibrated_eye_close)//2 if NEED_CALIBRATION else EYE_CUTOFF
print(f"Calibrated eye cutoff as {calibrated_eye}")

start_time = time.time()
while cap.isOpened():
    frame += 1
    suc,img = cap.read()
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
            elif (ear>= calibrated_eye+3 and flagEye) or (eyeCounter>30):
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
            yawnList.append(mouth)
            if len(ratioList) > AVG_LIST_LENGTH:
                ratioList.pop(0)
                if len(yawnList) > AVG_LIST_LENGTH*3 : yawnList.pop(0)
            ear = sum(ratioList)/len(ratioList)
            mouth = sum(yawnList)/len(yawnList)

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
        cvzone.putTextRect(img,f"Time : {round((time.time()-start_time),2)}",(50,150),2,colorT=(0,0,0),colorR=(255,255,255))
        cvzone.putTextRect(img,f"Blink for {MIN_FRAME_FOR_BLINK_ALARM} frames under {calibrated_eye}",(350,50),1,thickness=1,colorR=(0,0,0),colorT=(255,255,255))
        cvzone.putTextRect(img,f"Yawn for {MIN_FRAME_FOR_YAWN_ALARM} frames over {MOUTH_CUTOFF}",(350,75),1,thickness=1,colorR=(0,0,0),colorT=(255,255,255))

        if(((round(time.time() - start_time,2))//TIME_FRAME - seconds) == 1):     #one time frame complete
          seconds += 1
        #   if(blinkCounter >= BLINK_THRESHOLD or yawnCounter >= YAWN_THRESHOLD):
        #     buzzer.on()
        #     time.sleep(2)
        #     buzzer.off()
          totalBlinkCounter += blinkCounter
          totalYawnCounter += yawnCounter
          yawnCounter = 0
          blinkCounter = 0

    cv.imshow("Result",img)
    [LOWERLIPCoord,UPPERLIPCoord,LEFTEYECoord,RIGHTEYECoord] = [[(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)],[(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)],[(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)],[(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]]
    if cv.waitKey(10) & 0xFF == ord('q'):
        break
cv.destroyAllWindows()
cap.release()
stop_time = time.time()

print(f"You blinked {totalBlinkCounter} times and yawned {totalYawnCounter} times in {round(stop_time-start_time,2)} seconds")
