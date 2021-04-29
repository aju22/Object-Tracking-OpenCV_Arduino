# Object-Tracking-OpenCV_Arduino


## Description
This project operates a pan-tilt-servo camera bracket by an Arduino to track any object. It uses OpenCV a real time computer vision library, to detect object and the information is relayed onto the Arduino, which control the Servo Motors connected to the mount, and tracks the target.
## Technology Used
    1. 2 S690 Servo Motors
    2. Pan-Tilt Assembly
    3. Arduino UNO
    4. WebCam
    5. Laptop/PC
![Tech Used](https://github.com/aju22/Object-Tracking-OpenCV_Arduino/blob/main/0b06aca20a6b3ea950088cf2372125d6.jpg?raw=true)
## Masking
A mask is created by the user, by adjusting the sliders that control the HSV ranges to effectively track the target.
Once mask is set, hit Enter or SpaceBar to lock the mask.
![Masking Values](https://github.com/aju22/Object-Tracking-OpenCV_Arduino/blob/main/Githubimg.jpg?raw=true)
## Tracking
OpenCV then tracks the target, by locating its contours and relays the positional information to Arduino for Servo Control.
![Tracking Target](https://github.com/aju22/Object-Tracking-OpenCV_Arduino/blob/main/git2.jpg?raw=true)
## Guidance
Camera is guided by the positional information relayed by the Arduino to the Servo Motors, keeping the target always at the centre of the frame.

![Movement and Guidance](https://github.com/aju22/Object-Tracking-OpenCV_Arduino/blob/main/20210429_163852.gif?raw=true)


