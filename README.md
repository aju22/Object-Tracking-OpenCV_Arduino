# Object-Tracking-OpenCV_Arduino


## Description
This project operates a pan-tilt-servo camera bracket by an Arduino to track any object. It uses OpenCV a real time computer vision library, to detect object and the information is relayed onto the Arduino, which control the Servo Motors connected to the mount, and tracks the target.
## Masking
A mask is created by the user, by adjusting the sliders that control the HSV ranges to effectively track the target.
Once mask is set, hit Enter or SpaceBar to lock the mask.
![Masking Values](https://github.com/aju22/Object-Tracking-OpenCV_Arduino/blob/main/Githubimg.jpg?raw=true)
## Tracking
OpenCV then tracks the target, by locating its contours and relays the positional information to Arduino for Servo Control.
![Tracking Target](https://github.com/aju22/Object-Tracking-OpenCV_Arduino/blob/main/git2.jpg?raw=true)
## Guidance


