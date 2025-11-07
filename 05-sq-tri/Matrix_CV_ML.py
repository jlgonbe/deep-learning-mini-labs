# Python class for transforming images into 0,1 matrices from a directory
# Method build_ML_matrix() will loop thru the files in the folder, read them
# using OpenCV and then transform them into 0,1 matrices. These matrices will
# be then flattened into a row, and will be stacked together vertically ( one
# row per sample, and one column per pixel. Total columns equals = width*len
# specified by the user.
# It is assumed that the aspect ratio of the images is similar
# Outputs: global_matrix: Holds the 0,1 matrix of features (pixels)
#         labels       : Holds the labels 0,1,2,k according to each image class
# Inputs: It is assumed that the file names are like this "_". For example:
# [NAME]_[number].csv where [NAME] can be any name, and [number] is the label type
# For example, we can use 0 for triangles, 1 for circles, etc. It is important to
# note that this character must be a number.
from os import listdir
from os.path import isfile, join

import cv2
import numpy as np


class Matrix_CV_ML:

    def __init__(self, path, height, width):
        if not path.endswith('/'):
            path = path + "/"
        self.path = path
        self.height = height
        self.width = width
        self.type = type
        self.onlyfiles = [f for f in listdir(self.path) if isfile(join(self.path, f))]

    def prepare_matrix(self, pathx):
        path = self.path + pathx
        img = cv2.imread(path)

        img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
        binary_matrix = np.empty([img.shape[0], img.shape[1]], dtype="int32")

        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                if(img[x, y][0] == 255 and img[x, y][1] == 255 and img[x, y][2] == 255):
                    binary_matrix[x, y] = 1
                else:
                    binary_matrix[x, y] = 0
        transformed_matrix = (binary_matrix.flatten("F")).reshape(1, self.width*self.height)
        return(transformed_matrix)

    def build_ML_matrix(self):
        counter = 0
        self.labels = np.empty([len(self.onlyfiles)], dtype="int32")
        for file in self.onlyfiles:
            q = self.prepare_matrix(file)
            if (counter == 0):
                self.global_matrix = q
            else:
                self.global_matrix = np.append(self.global_matrix, q, axis=0)
            dash = file.rfind("_")
            dot = file.rfind(".")
            classtype = file[dash+1:dot]
            self.labels[counter] = classtype
            counter = counter + 1
