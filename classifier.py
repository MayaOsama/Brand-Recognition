

import os
import glob
import pandas as pd
import tensorflow as tf
import csv
from PIL import Image
import cv2

import numpy as np
import PIL

class Classifier(object):
    def __init__(self):
        PATH_TO_MODEL = './Adidas_model/frozen_inference_graph.pb'
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            # Works up to here.
            with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')
        self.sess = tf.Session(graph=self.detection_graph)
    

    
    def get_classification(self, img):

        # Bounding Box Detection.

        with self.detection_graph.as_default():

            # Expand dimension since the model expects image to have shape [1, None, None, 3].

            img_expanded = np.expand_dims(img, axis=0)  

            (boxes, scores, classes, num) = self.sess.run(

                [self.d_boxes, self.d_scores, self.d_classes, self.num_d],

                feed_dict={self.image_tensor: img_expanded})

        return boxes, scores, classes, num

    def draw_boxes(image_name):
        selected_value = full_labels[full_labels.filename == image_name]
        img = cv2.imread('train/all_images/{}'.format(image_name))
        for index, row in selected_value.iterrows():
            img = cv2.rectangle(img, (row['XMin'], row['YMin']), (row['XMax'], row['YMax']), (0, 255, 0), 3)
        return img


def main(_):
    test_dir= './test_dir/'
    # os.chdir(test_dir)

    extension = 'jpg'
    # TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR,'/*.{}' .format(extension)) ]
    all_images = [i for i in glob.glob(test_dir+'/*.{}'.format(extension))]

    # print ("enter")
    # print("all images" , all_images)
    # for img in all_images:
    #     print(img)
    classifier = Classifier()
    image = cv2.imread('{}'.format('./test_dir/image2.jpg'))
    boxes, scores, classes, num=classifier.get_classification(img=image)
    test = zip(boxes, scores, classes)
    for values in test:
        zipzipped = zip(values[0],values[1],values[2])
        max_ =  max(values[1])
        index = np.where(values[1] == max_)
        print("max",max_)
        if values[2][index]==1:
            class_="Adidas"
        elif values[2][index]==2:
            class_= "Apple"
        elif values[2][index]==3:
            class_= "BMW"
        elif values[2][index]==4:
            class_= "Citroen"
        elif values[2][index]==5:
            class_= "Cocacola"
        elif values[2][index]==6:
            class_= "DHL"
        elif values[2][index]==7:
            class_= "Fedex"
        elif values[2][index]==8:
            class_= "Ford"
        elif values[2][index]==9:
            class_= "Google"
        elif values[2][index]==10:
            class_= "HP"
        elif values[2][index]==11:
            class_= "Intel"
        elif values[2][index]==12:
            class_= "McDonalds"
        elif values[2][index]==13:
            class_= "Mini"
        elif values[2][index]==14:
            class_= "NBC"
        elif values[2][index]==15:
            class_= "Pepsi"
        elif values[2][index]==16:
            class_= "Porsche"
        elif values[2][index]==17:
            class_= "Puma"
        elif values[2][index]==18:
            class_= "RedBull"
        elif values[2][index]==19:
            class_= "Sprite"
        elif values[2][index]==20:
            class_= "Starbucks"
        elif values[2][index]==21:
            class_= "Texaco"
        elif values[2][index]==22:
            class_= "Unicef"
        elif values[2][index]==23:
            class_= "Vodafone"
        elif values[2][index]==24:
            class_= "Yahoo"
        elif values[2][index]==25:
            class_= "MK"
        elif values[2][index]==26:
            class_= "channel"
        elif values[2][index]==27:
            class_= "Gucci"
        elif values[2][index]==28:
            class_= "HH"
        elif values[2][index]==29:
            class_= "lacoste"
        elif values[2][index]==30:
            class_= "supreme"
        elif values[2][index]==31:
            class_= "Nike"
        elif values[2][index]==32:
            class_= " Heineken"
        elif values[2][index]==33:
            class_= "Ferrari"
        else:
            class_="NONE"
        print("box",values[0][index],"MAX score ",max_," class ",class_)
        index=0
        for zipp in zipzipped:
            # index = np.where(zipzipped == zipp)
            if values[2][index]==1:
                class_="Adidas"
            elif values[2][index]==2:
                class_= "Apple"
            elif values[2][index]==3:
                class_= "BMW"
            elif values[2][index]==4:
                class_= "Citroen"
            elif values[2][index]==5:
                class_= "Cocacola"
            elif values[2][index]==6:
                class_= "DHL"
            elif values[2][index]==7:
                class_= "Fedex"
            elif values[2][index]==8:
                class_= "Ford"
            elif values[2][index]==9:
                class_= "Google"
            elif values[2][index]==10:
                class_= "HP"
            elif values[2][index]==11:
                class_= "Intel"
            elif values[2][index]==12:
                class_= "McDonalds"
            elif values[2][index]==13:
                class_= "Mini"
            elif values[2][index]==14:
                class_= "NBC"
            elif values[2][index]==15:
                class_= "Pepsi"
            elif values[2][index]==16:
                class_= "Porsche"
            elif values[2][index]==17:
                class_= "Puma"
            elif values[2][index]==18:
                class_= "RedBull"
            elif values[2][index]==19:
                class_= "Sprite"
            elif values[2][index]==20:
                class_= "Starbucks"
            elif values[2][index]==21:
                class_= "Texaco"
            elif values[2][index]==22:
                class_= "Unicef"
            elif values[2][index]==23:
                class_= "Vodafone"
            elif values[2][index]==24:
                class_= "Yahoo"
            elif values[2][index]==25:
                class_= "MK"
            elif values[2][index]==26:
                class_= "channel"
            elif values[2][index]==27:
                class_= "Gucci"
            elif values[2][index]==28:
                class_= "HH"
            elif values[2][index]==29:
                class_= "lacoste"
            elif values[2][index]==30:
                class_= "supreme"
            elif values[2][index]==31:
                class_= "Nike"
            elif values[2][index]==32:
                class_= " Heineken"
            elif values[2][index]==33:
                class_= "Ferrari"
            else:
                class_="NONE"
            print("box",values[0][index]," score ",values[1][index]," class ",class_)
            index+=1

                # print(zipp)

       


    
if __name__ == '__main__':
    tf.app.run()





