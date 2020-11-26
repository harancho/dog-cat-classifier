import os
import tensorflow as tf
from tensorflow import keras
import pickle
import matplotlib.pyplot as plt
import numpy as np
import cv2

DIRECTORY = '/home/harsh/Documents/Projects/image-dog-classifier'
DIRECTORY_2 = '/home/harsh/Documents/Projects/image-dog-classifier/testing_images_folder'

IMG_SIZE = 100

new_model = tf.keras.models.load_model(DIRECTORY + '/model_keras.h5')
new_model.summary()

x = pickle.load(open('x.pkl','rb'))  #again getting back the saved arrays
y = pickle.load(open('y.pkl','rb'))

# x_test = pickle.load(open('x_test.pkl','rb'))  #again getting back the saved arrays
# y_test = pickle.load(open('y_test.pkl','rb'))

# x_valid = pickle.load(open('x_valid.pkl','rb'))

x = x/255
# x_test = x_test/255
# x_valid = x_valid/255

# loss, acc = new_model.evaluate(x, y, verbose=2 , batch_size = 32)
# print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

# loss, acc = new_model.evaluate(x_test, y_test, verbose=2)
# print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

# print(y_test)
# output = new_model.predict_classes(x_test , verbose = 2)
# print(output)

# wrong = 0
# for item in output:
# 	if y_test[item] != output[item]:
# 		wrong = wrong + 1
# print(wrong)
img_array =[]

for image in os.listdir(DIRECTORY_2):
	img_path = os.path.join(DIRECTORY_2,image)
	data = cv2.imread(img_path)
	data = cv2.resize(data,(IMG_SIZE,IMG_SIZE))
	img_array.append(data)
	img_array = np.array(img_array)
	print(img_array.shape)

img_array = img_array/255

output = new_model.predict_classes(img_array,verbose = 2)
if output[0] == 0:
	print('cat')
else:
	print('dog')
