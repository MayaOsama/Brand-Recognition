import numpy as np
from keras.datasets import mnist,fashion_mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD,adam
from keras.utils import np_utils
from keras import backend as K
import os
import PIL



np.random.seed(1617)  #for reproducibility



#network and training

NB_EPOCH =65
# NUM_EPOCHS = 25
INIT_LR = 1e-2
BARCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10 #number of outputs = nuber of digits
# OPTIMIZER =SGD()
OPTIMIZER =adam(lr=0.0001, decay=1e-6)

N_HIDDEN =128
VALIDATION_SPLIT = 0.2 #how much tain is reserved for VALIDATION
DROPOUT =0.3
labelNames = ["top", "trouser", "pullover", "dress", "coat",
	"sandal", "shirt", "sneaker", "bag", "ankle boot"]

#data: shuffled and split between train and test sets

(x_train, y_train),(x_test,y_test) = fashion_mnist.load_data()
#x_train is  60000 rows of 28*28 values 
#reshaped in 60000 *784

RESHAPED = 784

x_train = x_train.reshape(60000,RESHAPED)
x_test = x_test.reshape(10000,RESHAPED)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


#normalize

x_train/=255
x_test/=255

print(x_train.shape[0],'tarin samples')
print(x_test.shape[0],'test samples')


#convert class vectors to binary class matrcies

y_train =np_utils.to_categorical(y_train,NB_CLASSES)
y_test  =np_utils.to_categorical(y_test,NB_CLASSES)



# M O D E L
model= Sequential()
model.add(Dense(N_HIDDEN,input_shape=(RESHAPED,)))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))

model.add(Dense(N_HIDDEN))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))

model.add(Dense(N_HIDDEN))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))


model.add(Dense(N_HIDDEN))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))

model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))


model.summary()
model.compile(loss='categorical_crossentropy',optimizer=OPTIMIZER,
                metrics=['accuracy'])

history = model.fit(x_train,y_train,
                    batch_size=BARCH_SIZE, epochs=NB_EPOCH,
                    verbose=VERBOSE,validation_split=VALIDATION_SPLIT)

score = model.evaluate(x_test,y_test,verbose=VERBOSE)


print('Test Score: ',score[0])
print('Test accurcy: ',score[1])
model.save_weights('fashoin_mnist2.h5')
save_dir = os.path.join(os.getcwd(), 'saved_models')

model_name="fashoin_mnist2.h5"

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)

print('Saved trained model at %s ' % model_path)
