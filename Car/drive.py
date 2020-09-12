import cv2
import serial
from serial import Serial
from time import sleep
import readchar
import threading
import paramiko

ser = serial.Serial('/dev/ttyUSB0',9600)

def dieuKhien():
    while True:
        drive = open('/home/pi/drive.txt','r')
        input1 = drive.read()
        drive.close()
        try:
            delayTime = input1.split("|")[1]
        except Exception:
            delayTime = "100"
        if 'p' in input1:
            exit()
        if 'w' in input1 and input1 != 'wq':
            str1 = "up|" + delayTime + "\r"
            ser.write(str1.encode())
            sleep(float(delayTime)/1000 + 0.2)
        if 's' in input1:
            str1 = "down|" + delayTime + "\r"
            ser.write(str1.encode())
            sleep(float(delayTime)/1000 + 0.2)
        if 'a' in input1:
            str1 = "left|" + delayTime + "\r"
            ser.write(str1.encode())
            sleep(float(delayTime)/1000 + 0.2)
        if 'd' in input1:
            str1 = "right|" + delayTime + "\r"
            ser.write(str1.encode())
            sleep(float(delayTime)/1000 + 0.2)
        if 'q' in input1:
            str1 = "quay|400\r"
            ser.write(str1.encode())
            sleep(0.31)
        if 'e' in input1:
            str1 = "dungQuay|0\r"
            ser.write(str1.encode())
        if(ser.in_waiting>0):
            print(ser.readline())

def cameraXe():
    s = paramiko.SSHClient()
    s.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    s.connect("192.168.43.108",22,username="ntdat", password="1", timeout=10)
    sftp = s.open_sftp()
    cam = cv2.VideoCapture(0)
    while True:
        #cam = cv2.VideoCapture(0)
        re, image = cam.read()
        image = cv2.flip(image, -1)
        image = cv2.resize(image, (256,256))
        cv2.imwrite("/home/pi/cameraxe.png", image)
        sftp.put("/home/pi/cameraxe.png", "/home/ntdat/Desktop/DoAn2/Camera.png")
        cv2.waitKey(200)
        #cam.release()
try:
   t1 = threading.Thread(target=dieuKhien)
   t2 = threading.Thread(target=cameraXe)
   t1.start()
   t2.start()
   t1.join()
   t2.join()
except:
    print("ERROR!!!!!")
