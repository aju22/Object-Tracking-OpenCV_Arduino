#include<Servo.h>
Servo servoVer; //Vertical Servo
Servo servoHor; //Horizontal Servo
int x;
int y;
float rate = 1;
int clearance = 60;
const int x_center = 320;
const int y_center = 240;
int posx = 90;
int posy = 70;
char ch;
char input;
void setup()
{
  Serial.begin(9600);
  servoVer.attach(9); 
  servoHor.attach(8); 
  servoVer.write(70);
  servoHor.write(90);
}

void loop()
{
  if(Serial.available() > 0)
  {
    if(Serial.read() == 'X')
    {
      x = Serial.parseInt();
      if (x < x_center - clearance){
          posx += rate;
         }
      else if (x > x_center + clearance){
          posx -= rate;
          }
      servoHor.write(posx);
      if(Serial.read() == 'Y')
        {
          y = Serial.parseInt();
          if (y < (y_center - clearance)){
          posy -= rate;
          }
          else if (y > (y_center + clearance)){
          posy += rate;
          }
          servoVer.write(posy);
        }
    }
    while(Serial.available() > 0)
    {
      Serial.read();
    }
  }
}
