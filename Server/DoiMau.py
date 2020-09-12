import cv2
import os
import numpy as np

list_dir = os.listdir("/home/ntdat/Downloads/Data_Camera_SanTennis_Labeled/Labels/")

for img in list_dir:
  image = cv2.imread("/home/ntdat/Downloads/Data_Camera_SanTennis_Labeled/Labels/" + img)
  mask_background = cv2.inRange(image, np.array([0,0,128]), np.array([0,0,128]))
  mask_regionEnable = cv2.inRange(image, np.array([128,0,0]), np.array([128,0,0]))
  mask_tennisCourts = cv2.inRange(image, np.array([128,0,128]), np.array([128,0,128]))
  mask_line = cv2.inRange(image, np.array([0,128,128]), np.array([0,128,128]))
  mask_ball = cv2.inRange(image, np.array([0,128,0]), np.array([0,128,0]))
  # cv2.imshow("mask_line", mask_line)
  # cv2.waitKey()

  image[mask_background>0]=(0,0,0)
  image[mask_regionEnable>0]=(1,1,1)
  image[mask_tennisCourts>0]=(2,2,2)
  image[mask_line>0]=(3,3,3)
  image[mask_ball>0]=(4,4,4)

  # cv2.imshow("image Gray", cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
  # cv2.waitKey()

  cv2.imwrite("/home/ntdat/Downloads/Data_Camera_SanTennis_Labeled/Labels/" + img, image)
