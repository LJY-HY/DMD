import cv2
import numpy as np
import pandas as pd

font                   = cv2.FONT_HERSHEY_DUPLEX 
bottomLeftCornerOfText = (10,500)
fontScale              = 0.7
fontBlack              = (0,0,0)
fontRed                = (255,0,0)
lineType               = 0

# read video
cap = cv2.VideoCapture('./dmd.mp4')
width = int(cap.get(3))
hetight = int(cap.get(4))

text_area = np.ones((720,400,3),dtype=np.uint8)*255

#fcc = cv2.VideoWriter_fourcc('D','I','V','X')

frame_count=1
label = 0

# show video
while (cap.isOpened()) :
    ret, frame = cap.read()
    cv2.namedWindow('results')
    img = cv2.hconcat([frame,text_area])

    # Text Results
    # item
    cv2.putText(img,'Ground Truth', (1410,50), font, fontScale, fontBlack, lineType)
    cv2.putText(img,'Baseline', (1400,150),font, fontScale, fontBlack, lineType)
    cv2.putText(img,'Ours', (1570, 150),font, fontScale, fontBlack, lineType)
    cv2.putText(img,'Prediction',(1280, 180),font,fontScale,fontBlack, lineType)
    cv2.putText(img,'ACC',(1280, 200),font,fontScale,fontBlack,lineType)
    cv2.putText(img,'Frames'+str(frame_count),(1280, 220),font,fontScale,fontBlack,lineType)

    # Ground Truth
    cv2.putText(img,str(label), (1500,80),font,fontScale, fontBlack, lineType)
    # Before Model
    cv2.putText(img,str(label),(1400,180),font,fontScale,fontBlack,lineType)
    # After Model
    cv2.putText(img,str(label),(1570,180),font,fontScale,fontBlack, lineType)
    # 
    
    # frame count
    if frame_count ==10:
        frame_count=0
        label +=1
    frame_count+=1

    cv2.imshow('results',img)

    if cv2. waitKey(10) == 27 : break

cap.release()
cv2.destroyAllWindows()
