# -*- coding: utf-8 -*-
import tensorflow as tf
import cv2
import numpy as np
import os
import time
import paramiko

BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
NUM_OF_CLASSES = 5
PATH_MODEL = "/home/ntdat/Desktop/DoAn2/Model 1-7-20202/model.ckpt"
PATH_CAMERA_IMAGE = "/home/ntdat/Desktop/DoAn2/Camera.png"

LABEL_COLOR = np.array([[0,0,0],[0,50,0], [100,0,0], [0,0,150], [255,255,255]], dtype=np.float32)

def one_hot_tensors_to_color_images(img_np):
    outputs = []
    output = np.zeros(shape=(IMG_HEIGHT, IMG_WIDTH, 3), dtype='uint8')
    img_np[np.where(img_np == img_np.max(axis=-1, keepdims=1))] = 1
    for i in range(NUM_OF_CLASSES):
        indices = np.where(img_np[0,:, :, i] == 1)
        output[indices] = LABEL_COLOR[i]
    outputs.append(output)
    return outputs

def convert_five_color2two_color(img_np, index):
    output = np.zeros(shape=(IMG_HEIGHT, IMG_WIDTH, 3), dtype='uint8')
    img_np[np.where(img_np == img_np.max(axis=-1, keepdims=1))] = 1
    indices = np.where(img_np[0, :, :, index] == 1)
    output[indices] = LABEL_COLOR[4]
    return output

# Tìm quả bóng nà
def find_entity(img_np, index, name):
    # predict_two_color = cv2.inRange(img_np, LABEL_COLOR[4], LABEL_COLOR[4])
    predict_two_color = convert_five_color2two_color(img_np, index)
    gray = cv2.cvtColor(predict_two_color, cv2.COLOR_BGR2GRAY)
    cv2.imshow(name, gray)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #Khởi tạo max
    maxx=0
    maxy=0
    maxw=0
    maxh=0
    maxS=0

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        s = cv2.contourArea(contour)
        #Diện tích hơn 200 mới lấy
        if s < 500:
            continue
        
        if s > maxS:
            if index == 4:
                crop_ball = predict[x:x+w,y:y+h]
                crop_ball = cv2.countNonZero(cv2.inRange(predict, np.array([100,0,0]), np.array([100,0,0])))
                print(crop_ball)
                if crop_ball < 200:
                    maxS = s
                    maxx = x
                    maxy = y
                    maxw = w
                    maxh = h
            else:
                maxS = s
                maxx = x
                maxy = y
                maxw = w
                maxh = h
                
    return maxS, maxx, maxy, maxw, maxh

def Unet_classify(input_image):
  '''
  :param input_image: Input shape = (N, 480, 480, 1)
  :return:
  '''
  # block1
  conv1_1 = tf.layers.conv2d(inputs=input_image, filters=64, kernel_size=(3, 3), strides=1, padding="same")
  conv1_2 = tf.layers.conv2d(inputs=conv1_1, filters=64, kernel_size=(3, 3), strides=1, padding="same")
  # Shape = (N, 480, 480, 64)

  # block2
  maxP2 = tf.layers.max_pooling2d(inputs=conv1_2, pool_size=(2, 2), strides=2, padding="same")
  conv2_1 = tf.layers.conv2d(inputs=maxP2, filters=128, kernel_size=(3, 3), strides=1, padding="same")
  conv2_2 = tf.layers.conv2d(inputs=conv2_1, filters=128, kernel_size=(3, 3), strides=1, padding="same")
  #Shape = (N, 240, 240, 128)

  # block3
  maxP3 = tf.layers.max_pooling2d(inputs=conv2_2, pool_size=(2, 2), strides=2, padding="same")
  conv3_1 = tf.layers.conv2d(inputs=maxP3, filters=256, kernel_size=(3, 3), strides=1, padding="same")
  conv3_2 = tf.layers.conv2d(inputs=conv3_1, filters=256, kernel_size=(3, 3), strides=1, padding="same")
  # Shape = (N, 120, 120, 256)

  # block4
  maxP4 = tf.layers.max_pooling2d(inputs=conv3_2, pool_size=(2, 2), strides=2, padding="same")
  conv4_1 = tf.layers.conv2d(inputs=maxP4, filters=512, kernel_size=(3, 3), strides=1, padding="same")
  conv4_2 = tf.layers.conv2d(inputs=conv4_1, filters=512, kernel_size=(3, 3), strides=1, padding="same")
  # Shape = (N, 60, 60, 512)

  # block5
  maxP5 = tf.layers.max_pooling2d(inputs=conv4_2, pool_size=(2, 2), strides=2, padding="same")
  conv5_1 = tf.layers.conv2d(inputs=maxP5, filters=1024, kernel_size=(3, 3), strides=1, padding="same")
  conv5_2 = tf.layers.conv2d(inputs=conv5_1, filters=1024, kernel_size=(3, 3), strides=1, padding="same")
  # Shape = (N, 30, 30, 1024)
  #================#    
  #  SEGMENTATION  #
  #================#
  #blockUP1
  up_conv1 = tf.layers.conv2d_transpose(inputs=conv5_2, filters=512, kernel_size= (2, 2), strides=2, padding="same")
  # Shape = (N, 60, 60, 512)
  concat1 = tf.concat([conv4_2, up_conv1], axis=3)
  # Shape = (N, 60, 60, 1024)
  convUP1_1 = tf.layers.conv2d(inputs=concat1, filters=512, kernel_size=(3, 3), strides=1, padding="same")
  convUP1_2 = tf.layers.conv2d(inputs=convUP1_1, filters=512, kernel_size=(3, 3), strides=1, padding="same")
  # Shape = (N, 60, 60, 512)

  #blockUP2
  up_conv2 = tf.layers.conv2d_transpose(inputs=convUP1_2, filters=256, kernel_size=(2, 2), strides=2, padding="same")
  # Shape = (N, 120, 120, 256)
  concat2 = tf.concat([conv3_2, up_conv2], axis=3)
  # Shape = (N, 120, 120, 512)
  convUP2_1 = tf.layers.conv2d(inputs=concat2, filters=256, kernel_size=(3, 3), strides=1, padding="same")
  convUP2_2 = tf.layers.conv2d(inputs=convUP2_1, filters=256, kernel_size=(3, 3), strides=1, padding="same")
  # Shape = (N, 120, 120, 256)

  #blockUP3
  up_conv3 = tf.layers.conv2d_transpose(inputs=convUP2_2, filters=128, kernel_size=(2, 2), strides=2, padding="same")
  # Shape = (N, 240, 240, 128)
  concat3 = tf.concat([conv2_2, up_conv3], axis=3)
  # Shape = (N, 240, 240, 256)
  convUP3_1 = tf.layers.conv2d(inputs=concat3, filters=128, kernel_size=(3, 3), strides=1, padding="same")
  convUP3_2 = tf.layers.conv2d(inputs=convUP3_1, filters=128, kernel_size=(3, 3), strides=1, padding="same")
  # Shape = (N, 240, 240, 128)

  #blockUP4
  up_conv4 = tf.layers.conv2d_transpose(inputs=convUP3_2, filters=64, kernel_size=(2, 2), strides=2, padding="same")
  # Shape = (N, 480, 480, 64)
  concat4 = tf.concat([conv1_2, up_conv4], axis=3)
  # Shape = (N, 480, 480, 128)
  convUP4_1 = tf.layers.conv2d(inputs=concat4, filters=64, kernel_size=(3, 3), strides=1, padding="same")
  convUP4_2 = tf.layers.conv2d(inputs=convUP4_1, filters=64, kernel_size=(3, 3), strides=1, padding="same")
  # Shape = (N, 480, 480, 64)

  middlemen = tf.layers.conv2d(inputs= convUP4_2, filters=32, kernel_size=(3, 3), strides=1, padding="same")

  output_segmentation = tf.layers.conv2d(inputs= middlemen, filters=NUM_OF_CLASSES, kernel_size=(3, 3), strides=1, padding="same")

  return tf.nn.softmax(output_segmentation)

input_tensor = tf.placeholder(dtype=tf.float32, shape=(1, IMG_HEIGHT, IMG_WIDTH, 3))

predict_tensor = Unet_classify(input_tensor)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth = True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

tf.train.Saver().restore(sess, PATH_MODEL)

#ssh qua PI
s = paramiko.SSHClient()
s.set_missing_host_key_policy(paramiko.AutoAddPolicy())
s.connect("192.168.43.18",22,username="pi", password="15042018", timeout=10)
sftp = s.open_sftp()
dem = 1500
while True:
    try:
        drive = open('/home/ntdat/Desktop/DoAn2/drive.txt', 'w')
        drive.write("")
        drive.close()
        sftp.put("/home/ntdat/Desktop/DoAn2/drive.txt", "/home/pi/drive.txt", confirm=False)
        cam = cv2.imread(PATH_CAMERA_IMAGE)
        cv2.imwrite("/home/ntdat/Desktop/DoAn2/Data 03-07-2020/Camera_" + str(dem) + ".png", cam)
        dem = dem + 1
        image = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        image = np.expand_dims(image, axis=0)
        image = image/255.
        result = sess.run(predict_tensor, feed_dict={input_tensor: image})
        predict = one_hot_tensors_to_color_images(result)[0]

        cv2.imshow("Predict", predict)
        cv2.imshow("RGB", cam)
        
        #Kiểm tra xem sắp bay ra ngoài chưa
        background_Area, BGx, BGy, BGw, BGh = find_entity(result, 0, "BG")
        mask_san = cv2.inRange(predict, np.array([100,0,0]), np.array([100,0,0]))
        san_Area = cv2.countNonZero(mask_san)
        mask_enableRegion = cv2.inRange(predict, np.array([0,50,0]), np.array([0,50,0]))
        enableRegion_Area = cv2.countNonZero(mask_enableRegion)
        if enableRegion_Area > 30000:
            maxS, maxx, maxy, maxw, maxh = find_entity(result, 4, "Ball")
            s = ""
            if maxS != 0:
                if (maxx + maxw/2 <255/3 and maxy + maxh/2 < 255*2/3 and maxx!=0 and maxy!=0):
                    print("Rẽ trái!!!")
                    s = "a"
                else:
                    if maxx + maxw/2 > 255*2/3 and maxy + maxh/2 < 255*2/3:
                        print("Rẽ phải!!!")
                        s = "d"
                    else:
                        print("Đi thẳng")
                        s = "w"

                if maxS > 2000:
                    s+="q"
                else:
                    s+="e"
            else:
                s = "we"
            drive = open('/home/ntdat/Desktop/DoAn2/drive.txt', 'w')
            drive.write(s)
            drive.close()
            sftp.put("/home/ntdat/Desktop/DoAn2/drive.txt", "/home/pi/drive.txt", confirm=False)
            continue
        if (background_Area > 60000 or san_Area > 40000):
            drive = open('/home/ntdat/Desktop/DoAn2/drive.txt', 'w')
            drive.write("s")
            drive.close()
            print("Đi lùi nhá " + str(background_Area))
        else:
            maxS, _, _, _, _ = find_entity(result, 3, "Line")
            if(maxS > 7000):
                drive = open('/home/ntdat/Desktop/DoAn2/drive.txt', 'w')
                drive.write("d")
                drive.close()
            else:
                if background_Area > 15000 and BGx+BGw/2 > 255*1/3:
                    drive = open('/home/ntdat/Desktop/DoAn2/drive.txt', 'w')
                    drive.write("a")
                    drive.close()
                
        sftp.put("/home/ntdat/Desktop/DoAn2/drive.txt", "/home/pi/drive.txt", confirm=False)
    except Exception as e:
        cv2.waitKey(50)
