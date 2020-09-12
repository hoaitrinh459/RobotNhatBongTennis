import tensorflow as tf
import cv2
import numpy as np
import os
import time

BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
PATH = "/home/ntdat/Downloads/Data_Camera_SanTennis_Labeled/"
NUM_CLASSES = 5
PATH_MODEL = "/home/ntdat/Desktop/DoAn2/Model 5 Class/model.ckpt"

LABEL_COLOR = np.array([[0,0,0],[142,0,0], [128,64,128], [0,220,220], [180,130,70]], dtype=np.float32)

def one_hot_tensors_to_color_images(img_np):
    outputs = []
    output = np.zeros(shape=(IMG_HEIGHT, IMG_WIDTH, 3), dtype='uint8')
    img_np[np.where(img_np == img_np.max(axis=-1, keepdims=1))] = 1
    for i in range(NUM_CLASSES):
        # if i==1:
        #     continue
        indices = np.where(img_np[0,:, :, i] == 1)
        output[indices] = LABEL_COLOR[i]
    outputs.append(output)
    return outputs

# def one_hot_tensors_to_color_images(output_id_classes):
    # output_batch = tf.reshape(tf.matmul(
    #         tf.cast(tf.reshape(tf.one_hot(tf.argmax(output_id_classes, -1), NUM_CLASSES), [-1, NUM_CLASSES]),dtype=tf.float32), LABEL_COLOR),
    #         [-1, IMG_HEIGHT, IMG_WIDTH, 3])
    # return output_batch

def one_hot(label_gray):
    label_channel_list = []
    for class_id in range(NUM_CLASSES):
        equal_map = tf.equal(label_gray, class_id)
        binary_map = tf.to_int32(equal_map)
        label_channel_list.append(binary_map)
    label_xin = tf.stack(label_channel_list, axis=3)
    label_xin = tf.squeeze(label_xin, axis=-1)
    return label_xin

def Unet(input_image):
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

  output_segmentation = tf.layers.conv2d(inputs= middlemen, filters=NUM_CLASSES, kernel_size=(3, 3), strides=1, padding="same")

  return tf.nn.softmax(output_segmentation)

# def Unet_classify(input_image):
#   '''
#   :param input_image: Input shape = (N, 480, 480, 1)
#   :return:
#   '''
#   # block1
#   conv1_1 = tf.layers.conv2d(inputs=input_image, filters=64, kernel_size=(3, 3), strides=1, padding="same")
#   conv1_2 = tf.layers.conv2d(inputs=conv1_1, filters=64, kernel_size=(3, 3), strides=1, padding="same")
#   # Shape = (N, 480, 480, 64)
#
#   # block2
#   maxP2 = tf.layers.max_pooling2d(inputs=conv1_2, pool_size=(2, 2), strides=2, padding="same")
#   conv2_1 = tf.layers.conv2d(inputs=maxP2, filters=128, kernel_size=(3, 3), strides=1, padding="same")
#   conv2_2 = tf.layers.conv2d(inputs=conv2_1, filters=128, kernel_size=(3, 3), strides=1, padding="same")
#   #Shape = (N, 240, 240, 128)
#
#   # block3
#   maxP3 = tf.layers.max_pooling2d(inputs=conv2_2, pool_size=(2, 2), strides=2, padding="same")
#   conv3_1 = tf.layers.conv2d(inputs=maxP3, filters=256, kernel_size=(3, 3), strides=1, padding="same")
#   conv3_2 = tf.layers.conv2d(inputs=conv3_1, filters=256, kernel_size=(3, 3), strides=1, padding="same")
#   # Shape = (N, 120, 120, 256)
#
#   # block4
#   maxP4 = tf.layers.max_pooling2d(inputs=conv3_2, pool_size=(2, 2), strides=2, padding="same")
#   conv4_1 = tf.layers.conv2d(inputs=maxP4, filters=512, kernel_size=(3, 3), strides=1, padding="same")
#   conv4_2 = tf.layers.conv2d(inputs=conv4_1, filters=512, kernel_size=(3, 3), strides=1, padding="same")
#   # Shape = (N, 60, 60, 512)
#
#   # block5
#   maxP5 = tf.layers.max_pooling2d(inputs=conv4_2, pool_size=(2, 2), strides=2, padding="same")
#   conv5_1 = tf.layers.conv2d(inputs=maxP5, filters=1024, kernel_size=(3, 3), strides=1, padding="same")
#   conv5_2 = tf.layers.conv2d(inputs=conv5_1, filters=1024, kernel_size=(3, 3), strides=1, padding="same")
#   # Shape = (N, 30, 30, 1024)
#   #================#
#   #  SEGMENTATION  #
#   #================#
#   #blockUP1
#   up_conv1 = tf.layers.conv2d_transpose(inputs=conv5_2, filters=512, kernel_size= (2, 2), strides=2, padding="same")
#   # Shape = (N, 60, 60, 512)
#   concat1 = tf.concat([conv4_2, up_conv1], axis=3)
#   # Shape = (N, 60, 60, 1024)
#   convUP1_1 = tf.layers.conv2d(inputs=concat1, filters=512, kernel_size=(3, 3), strides=1, padding="same")
#   convUP1_2 = tf.layers.conv2d(inputs=convUP1_1, filters=512, kernel_size=(3, 3), strides=1, padding="same")
#   # Shape = (N, 60, 60, 512)
#
#   #blockUP2
#   up_conv2 = tf.layers.conv2d_transpose(inputs=convUP1_2, filters=256, kernel_size=(2, 2), strides=2, padding="same")
#   # Shape = (N, 120, 120, 256)
#   concat2 = tf.concat([conv3_2, up_conv2], axis=3)
#   # Shape = (N, 120, 120, 512)
#   convUP2_1 = tf.layers.conv2d(inputs=concat2, filters=256, kernel_size=(3, 3), strides=1, padding="same")
#   convUP2_2 = tf.layers.conv2d(inputs=convUP2_1, filters=256, kernel_size=(3, 3), strides=1, padding="same")
#   # Shape = (N, 120, 120, 256)
#
#   #blockUP3
#   up_conv3 = tf.layers.conv2d_transpose(inputs=convUP2_2, filters=128, kernel_size=(2, 2), strides=2, padding="same")
#   # Shape = (N, 240, 240, 128)
#   concat3 = tf.concat([conv2_2, up_conv3], axis=3)
#   # Shape = (N, 240, 240, 256)
#   convUP3_1 = tf.layers.conv2d(inputs=concat3, filters=128, kernel_size=(3, 3), strides=1, padding="same")
#   convUP3_2 = tf.layers.conv2d(inputs=convUP3_1, filters=128, kernel_size=(3, 3), strides=1, padding="same")
#   # Shape = (N, 240, 240, 128)
#
#   #blockUP4
#   up_conv4 = tf.layers.conv2d_transpose(inputs=convUP3_2, filters=64, kernel_size=(2, 2), strides=2, padding="same")
#   # Shape = (N, 480, 480, 64)
#   concat4 = tf.concat([conv1_2, up_conv4], axis=3)
#   # Shape = (N, 480, 480, 128)
#   convUP4_1 = tf.layers.conv2d(inputs=concat4, filters=64, kernel_size=(3, 3), strides=1, padding="same")
#   convUP4_2 = tf.layers.conv2d(inputs=convUP4_1, filters=64, kernel_size=(3, 3), strides=1, padding="same")
#   # Shape = (N, 480, 480, 64)
#
#   middlemen = tf.layers.conv2d(inputs=convUP4_2, filters=32, kernel_size=(3, 3), strides=1, padding="same")
#
#   output_segmentation = tf.layers.conv2d(inputs=middlemen, filters=2, kernel_size=(3, 3), strides=1, padding="same")
#
#   return tf.nn.sigmoid(output_segmentation)

# tf.reset_default_graph()

input_tensor = tf.placeholder(dtype=tf.float32, shape=(1, IMG_HEIGHT, IMG_WIDTH, 3))
label_gray_tensor = tf.placeholder(dtype=tf.float32, shape=(1, IMG_HEIGHT, IMG_WIDTH, 1))
label_tensor = one_hot(label_gray_tensor)

predict_tensor = Unet(input_tensor)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth = True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# sess.run(tf.global_variables_initializer())
# sess.run(tf.local_variables_initializer())

tf.train.Saver().restore(sess, PATH_MODEL)

list_label = os.listdir(PATH + 'RGBs/')
for i in list_label:
    label = cv2.imread(PATH + "Labels/" + i)
    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    label = cv2.resize(label, (256,256))
    label = np.expand_dims(label, axis=0)
    label = np.expand_dims(label, axis=-1)

    BGR = cv2.imread(PATH + "RGBs/" + i)
    BGR = cv2.cvtColor(BGR, cv2.COLOR_BGR2RGB)
    image = cv2.resize(BGR, (256, 256))
    image = np.expand_dims(image, axis=0)

    image = image/255.

    t= time.time()
    result, label_val = sess.run([predict_tensor, label_tensor], feed_dict={input_tensor: image, label_gray_tensor: label})
    # result = sess.run([predict_tensor], feed_dict={input_tensor: image})

    print(label.shape)
    print(image.shape)
    print(result.shape)
    print(label_val.shape)

    print(time.time()-t)
    cv2.imshow("Predict", one_hot_tensors_to_color_images(result)[0])
    cv2.imshow("Label", one_hot_tensors_to_color_images(label_val)[0])
    cv2.imshow("RGB", cv2.resize(BGR,(256,256)))
    cv2.waitKey()

cv2.destroyAllWindows()


# while True:
#     camera = cv2.VideoCapture(0)
#     _, BGR = camera.read()
#     cv2.imshow("RGB1", BGR)
#     BGR = cv2.resize(BGR, (256, 256))
#     image = cv2.cvtColor(BGR, cv2.COLOR_BGR2GRAY)
#     image = np.expand_dims(image, axis=0)
#     image = np.expand_dims(image, axis=-1)
#
#     image = image/255.
#     t= time.time()
#
#     result = sess.run(predict_tensor, feed_dict={input_tensor: image})
#
#     print(time.time()-t)
#
#     cv2.imshow("Predict", one_hot_tensors_to_color_images(result)[0])
#     cv2.imshow("RGB", BGR)
#     cv2.waitKey(1)
#     camera.release()
