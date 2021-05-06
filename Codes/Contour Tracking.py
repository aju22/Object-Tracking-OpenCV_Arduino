import cv2 as cv
import numpy as np
import serial

x_center = 320
y_center = 240    # Based on Frame size of Video Output


class KalmanFilter(object):
    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas):
        """
        :param dt: sampling time (time for 1 cycle)
        :param u_x: acceleration in x-direction
        :param u_y: acceleration in y-direction
        :param std_acc: process noise magnitude
        :param x_std_meas: standard deviation of the measurement in x-direction
        :param y_std_meas: standard deviation of the measurement in y-direction
        """

        # Define sampling time
        self.dt = dt

        # Define the  control input variables
        self.u = np.matrix([[u_x], [u_y]])

        # Initial State
        self.x = np.matrix([[x_center], [y_center], [0], [0]])

        # Define the State Transition Matrix A
        self.A = np.matrix([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        # Define the Control Input Matrix B
        self.B = np.matrix([[(self.dt ** 2) / 2, 0],
                            [0, (self.dt ** 2) / 2],
                            [self.dt, 0],
                            [0, self.dt]])

        # Define Measurement Mapping Matrix
        self.H = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0]])

        # Initial Process Noise Covariance
        self.Q = np.matrix([[(self.dt ** 4) / 4, 0, (self.dt ** 3) / 2, 0],
                            [0, (self.dt ** 4) / 4, 0, (self.dt ** 3) / 2],
                            [(self.dt ** 3) / 2, 0, self.dt ** 2, 0],
                            [0, (self.dt ** 3) / 2, 0, self.dt ** 2]]) * std_acc ** 2

        # Initial Measurement Noise Covariance
        self.R = np.matrix([[x_std_meas ** 2, 0],
                            [0, y_std_meas ** 2]])

        # Initial Covariance Matrix
        self.P = np.eye(self.A.shape[1])

    def predict(self):
        # Update time state

        # x_k =Ax_(k-1) + Bu_(k-1)
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)

        # Calculate error covariance
        # P= A*P*A' + Q
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x[0], self.x[1]

    def update(self, z):
        # S = H*P*H'+R
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R

        # Calculate the Kalman Gain
        # K = P * H'* inv(H*P*H'+R)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        # New X = X + K(MEASURED - H*X)
        self.x = np.round(self.x + np.dot(K, (z - np.dot(self.H, self.x))))

        I = np.eye(self.H.shape[1])

        # Update error covariance matrix
        self.P = (I - (K * self.H)) * self.P
        return self.x[0, 0], self.x[1, 0]


def nothing(b):
    pass


arduino = serial.Serial('COM5', '9600')
capture = cv.VideoCapture(0)  # Select required Cam INPUT
centers = []
lower_range = np.array([0, 0, 0])
upper_range = np.array([0, 0, 0])
text_frame = np.zeros((640, 480), dtype='uint8')
x, y, w, h = [0, 0, 0, 0]

cv.namedWindow("Trackbars")  # Creating Trackbars to find accurate HSV Ranges
cv.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv.createTrackbar("L - S", "Trackbars", 101, 255, nothing)
cv.createTrackbar("L - V", "Trackbars", 162, 255, nothing)
cv.createTrackbar("U - H", "Trackbars", 56, 179, nothing)
cv.createTrackbar("U - S", "Trackbars", 213, 255, nothing)
cv.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

ok = True
while ok:
    ok, img = capture.read()
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    l_h = cv.getTrackbarPos("L - H",
                            "Trackbars")  # Adjust the trackbars until you get a perfect mask. Hit ENTER or SPACEBAR when done.
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

KF = KalmanFilter(0.1, 1, 1, 1, 0.1, 0.1)

while True:
    ret, frame = capture.read()

    cv.resizeWindow('detect', 640, 480)
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv_frame, lower_range, upper_range)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # Finding Contours based on the mask
    contours = sorted(contours, key=lambda X: cv.contourArea(X),
                      reverse=True)  # Getting the largest contour(Area) out of all the found contours.


    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 500:  # Noise Reduction
            (x, y, w, h) = cv.boundingRect(cnt)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            break
        else:
            cv.putText(frame, "No Object Detected !", (100, 100), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

    centers = np.array([[x], [y]])
    a, b = KF.predict()
    x1, y1 = KF.update(centers)
    cv.rectangle(frame, (int(x1), int(y1)), (int(x1) + w, int(y1) + h), (0, 255, 0), 2)
    cv.putText(frame, "Estimated Position", (int(x1) + 15, int(y1) + 10), 0, 0.5, (0, 0, 255), 2)
    x_center = int((x1 + x1 + w) / 2)
    y_center = int((y1 + y1 + h) / 2)
    cv.line(frame, (x_center, 0), (x_center, 480), (0, 255, 0), 2)
    cv.line(frame, (0, y_center), (640, y_center), (0, 255, 0), 2)


    arduino.write(f'X{x_center}Y{y_center}Z'.encode())


    cv.imshow('detect', frame)
    cv.imshow('mask', mask)
    if cv.waitKey(1) == 27:  # Enter ESC to terminate.
        break


capture.release()
cv.destroyAllWindows()
