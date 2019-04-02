from keras import backend as K
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
import PIL
import keras

model = model = load_model("saved_models/logos_trained_model3.h5")

# test_data_dir="test"
# test_datagen = ImageDataGenerator(rescale=1. / 255)
# train_generator = train_datagen.flow_from_directory(
#     test_data_dir,
#     target_size=(150, 150),
#     batch_size=10,
#     class_mode="categorical")


# img = load_img(image_path, target_size=(img_nrows, img_ncols))
test_list ={"jpg/gucci/Gucci_0_1541.jpg","jpg/adidas/adidas_0_1575.jpg",
"jpg/chanel/chanel_0_1898.jpg","jpg/mk/mk_0_1707.jpg"}
for i in test_list:
    print(i)
    test_img=load_img(i, target_size=(150, 150))
    x = img_to_array(test_img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)
    class_predicted = model.predict(x)
    print(class_predicted[0])

