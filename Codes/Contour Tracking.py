import cv2 as cv
import numpy as np
import serial


def nothing(b):
    pass


arduino = serial.Serial('COM5', '9600')
capture = cv.VideoCapture(1) # Select required Cam INPUT
x_center = 320
y_center = 240
lower_range = np.array([0, 0, 0])
upper_range = np.array([0, 0, 0])
text_frame = np.zeros((640, 480), dtype='uint8')
x, y, w, h = [0, 0, 0, 0]

cv.namedWindow("Trackbars")                                     # Creating Trackbars to find accurate HSV Ranges
cv.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

ok = True
while ok:
    ok, img = capture.read()
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    l_h = cv.getTrackbarPos("L - H", "Trackbars")   #Adjust the trackbars until you get a perfect mask. Hit ENTER or SPACEBAR when done.
    l_s = cv.getTrackbarPos("L - S", "Trackbars")
    l_v = cv.getTrackbarPos("L - V", "Trackbars")
    u_h = cv.getTrackbarPos("U - H", "Trackbars")
    u_s = cv.getTrackbarPos("U - S", "Trackbars")
    u_v = cv.getTrackbarPos("U - V", "Trackbars")
    lower_range = np.array([l_h, l_s, l_v])
    upper_range = np.array([u_h, u_s, u_v])
    mask = cv.inRange(hsv, lower_range, upper_range)
    cv.putText(img, "Adjust HSV Ranges to get a Perfect Mask.", (40, 400),
               cv.FONT_HERSHEY_TRIPLEX, 0.7, (255, 50, 100), 2)
    cv.putText(img, "Hit Spacebar or Enter to set Mask !", (90, 430),
               cv.FONT_HERSHEY_TRIPLEX, 0.7, (255, 50, 100), 2)
    cv.imshow("Trackbars", mask)
    cv.imshow('Image', img)
    if cv.waitKey(1) == 32 or cv.waitKey(1) == 10:
        print('Mask Set !')
        print(f"Lower Working Range : {lower_range}")
        print(f"Upper Working Range : {upper_range}")
        cv.destroyAllWindows()
        break

while True:
    ret, frame = capture.read()
    cv.resizeWindow('detect', 640, 480)
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv_frame, lower_range, upper_range)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)    #Finding Contours based on the mask
    contours = sorted(contours, key=lambda X: cv.contourArea(X), reverse=True)       #Getting the largest contour(Area) out of all the found contours.
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 500:                           # Noise Reduction
            (x, y, w, h) = cv.boundingRect(cnt)
            x_center = int((x + x + w) / 2)
            y_center = int((y + y + h) / 2)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv.line(frame, (x_center, 0), (x_center, 480), (0, 255, 0), 2)
            cv.line(frame, (0, y_center), (640, y_center), (0, 255, 0), 2)
            break
        else:
            cv.putText(frame, "No Object Detected !", (320, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    arduino.write(f'X{x_center}Y{y_center}Z'.encode())
    cv.imshow('detect', frame)
    cv.imshow('mask', mask)
    if cv.waitKey(1) == 27:     #Enter ESC to terminate.
        break
capture.release()
cv.destroyAllWindows()
