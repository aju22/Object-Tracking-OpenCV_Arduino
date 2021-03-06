# Object-Tracking-OpenCV_Arduino


## Description
This project operates a pan-tilt-servo camera bracket by an Arduino to track any object. It uses OpenCV a real time computer vision library, to detect object and the information is relayed onto the Arduino, which control the Servo Motors connected to the mount, and tracks the target.

The Codes for the project are written in Python and Arduino Programming Language.
## Technology Used
    1. 2 S690 Servo Motors
    2. Pan-Tilt Assembly
    3. Arduino UNO
    4. WebCam
    5. Laptop/PC

![Tech Used](Gifs%20and%20Images/0b06aca20a6b3ea950088cf2372125d6.jpg)

## Masking

A mask is created by the user, by adjusting the sliders that control the HSV ranges to effectively track the target.
Once mask is set, hit Enter or SpaceBar to lock the mask.
![Masking Values](Gifs%20and%20Images/Githubimg.jpg)

## Tracking

OpenCV then tracks the target, by locating its contours and relays the positional information to Arduino for Servo Control.
![Tracking Target](Gifs%20and%20Images/git2.jpg)

## Guidance

Camera is guided by the positional information relayed by the Arduino to the Servo Motors, keeping the target always at the centre of the frame.

![Movement and Guidance](Gifs%20and%20Images/20210429_163852.gif)

## Future Scopes

Implementing faster and more accurate tracking algorithms.

