import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
import pickle

DIRECTIORY = "/home/harsh/Documents/Projects/image-dog-classifier/dataset"
# DIRECTIORY_2 ="/home/harsh/Downloads/dataset-dogcat-classifier/valid"
# DIRECTIORY_3 = "/home/harsh/Downloads/dataset-dogcat-classifier/test1"

CATEGORIES = ['cats','dogs']

IMG_SIZE = 100

data =[]
# data_2 = []
# data_3 = []

for category in CATEGORIES:
	folder = os.path.join(DIRECTIORY,category)
	label = CATEGORIES.index(category)
	for image in os.listdir(folder):
		img_path = os.path.join(folder,image)
		img_array = cv2.imread(img_path)                #reading the RGB of image through cv2
		img_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))     #converting all images to same size so that we can process them
		data.append([img_array,label])

		
print(len(data))
random.shuffle(data)

x = []
y = []

for features,labels in data:
	x.append(features)        #adding all RGB to x list
	y.append(labels)          #adding all lables to y list

x = np.array(x)         #converting list to array through numpy
y = np.array(y)

pickle.dump(x,open('x.pkl','wb'))    #saving arrays in a file in pc
pickle.dump(y,open('y.pkl','wb'))

######################################################################

# for category in CATEGORIES:
# 	folder = os.path.join(DIRECTIORY_2,category)
# 	label = CATEGORIES.index(category)
# 	for image in os.listdir(folder):
# 		img_path = os.path.join(folder,image)
# 		img_array = cv2.imread(img_path)                #reading the RGB of image through cv2
# 		img_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))     #converting all images to same size so that we can process them
# 		data_2.append([img_array,label])

# print(len(data_2))
# random.shuffle(data_2)

# x_test = []
# y_test = []

# for features,labels in data_2:
# 	x_test.append(features)        #adding all RGB to x list
# 	y_test.append(labels)          #adding all lables to y list

# x_test = np.array(x_test)         #converting list to array through numpy
# y_test = np.array(y_test)

# pickle.dump(x_test,open('x_test.pkl','wb'))    #saving arrays in a file in pc
# pickle.dump(y_test,open('y_test.pkl','wb'))

# ##################################################################
# x_valid = []

# for image in os.listdir(DIRECTIORY_3):
# 	img_path = os.path.join(DIRECTIORY_3,image)
# 	img_array = cv2.imread(img_path)                #reading the RGB of image through cv2
# 	img_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))     #converting all images to same size so that we can process them
# 	x_valid.append(img_array)

# print(len(x_valid))

# x_valid = np.array(x_valid)

# pickle.dump(x_valid,open('x_valid.pkl','wb')) 

# print('\n')
# print(x.shape)
# print(x_test.shape)
# print(x_valid.shape)