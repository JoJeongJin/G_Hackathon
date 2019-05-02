import numpy as np
import cv2

def tracking():
    try:
        print('카메라를 구동합니다')
        cap = cv2.VideoCapture(1)
    except:
        print('카메라를 구동 실패')
        return
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (400, 400))

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_blue = np.array([110, 100, 100])
        upper_blue = np.array([130, 255, 255])

        lower_green = np.array([50, 100, 100])
        upper_green = np.array([70, 255, 255])

        lower_red = np.array([-10, 100, 100])
        upper_red = np.array([10, 255, 255])

        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_red = cv2.inRange(hsv, lower_red, upper_red)

        res1 = cv2.bitwise_and(frame, frame, mask=mask_blue)
        res2 = cv2.bitwise_and(frame, frame, mask=mask_green)
        res3 = cv2.bitwise_and(frame, frame, mask=mask_red)

        cv2.imshow('original', frame)
        cv2.imshow('Blue', res1)
        cv2.imshow('Green', res2)
        cv2.imshow('Red', res3)

        k = cv2.waitKey(1) & 0xFF
        if k==27:
            break

    cv2.destroyAllWindows()

if __name__=='__main__':
    tracking()

