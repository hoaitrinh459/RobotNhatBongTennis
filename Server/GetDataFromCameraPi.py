import paramiko
import curses
import time
import cv2
import threading


def driver():
    s = paramiko.SSHClient()
    s.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    s.connect("192.168.43.18",22,username="pi", password="15042018", timeout=10)
    sftp = s.open_sftp()

    screen = curses.initscr()
    screen.keypad(1)

    while(1):
        drive = open('/home/ntdat/Desktop/DoAn2/drive.txt', 'w')
        drive.write("")
        drive.close()
        sftp.put("/home/ntdat/Desktop/DoAn2/drive.txt", "/home/pi/drive.txt", confirm=False)
        
        drive = open('/home/ntdat/Desktop/DoAn2/drive.txt', 'w')
        #Lấy ký tự char nhập từ bàn phím -> Đưa vô file drive.txt
        key = screen.getch()
        print(chr(key))
        drive.write(chr(key))
        
        screen.refresh()
        drive.close()
        sftp.put("/home/ntdat/Desktop/DoAn2/drive.txt", "/home/pi/drive.txt", confirm=False)
        # f = sftp.open("/home/pi/drive.txt", 'w')
        # f.write(chr(key))
        # f.close()
    curses.endwin()

def showCamera():
    dem = 1500
    while True:
        try:
            cam = cv2.imread("/home/ntdat/Desktop/DoAn2/Camera.png")
            cv2.imshow("Camera Xe",cam)
            cv2.imwrite("/home/ntdat/Desktop/DoAn2/Data 03-07-2020/Camera_" + str(dem) + ".png", cam)
            cv2.waitKey(100)
            dem = dem + 1
        except:
            print("Read Camera Error!!!")

try:
    t1 = threading.Thread(target=showCamera)
    t2 = threading.Thread(target=driver)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

except Exception as e:
    print(e)
