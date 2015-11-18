import numpy as np
import cv2

dimm = (640,480)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, dimm[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, dimm[1])

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.createBackgroundSubtractorGMG()

while(1):
    ret, frame = cap.read()

    if ret == True:
        # rotate video upright
        M = cv2.getRotationMatrix2D((dimm[0]/2, dimm[1]/2), 270, 1.0)
        frame = cv2.warpAffine(frame, M, dimm)
            
	fgmask = fgbg.apply(frame)
	fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        cv2.imshow('PiGuard', fgmask)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

cap.release()
cv2.destroyAllWindows()
