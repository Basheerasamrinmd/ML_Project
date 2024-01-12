# ML_Project
#Flower Classification using CNN
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model

daisy_dir=os.path.join(r'C:\Users\DELL\Documents\ml\flowers\daisy')

dandelion_dir=os.path.join(r'C:\Users\DELL\Documents\ml\flowers\dandelion')

rose_dir=os.path.join(r'C:\Users\DELL\Documents\ml\flowers\rose')

sunflower_dir=os.path.join(r'C:\Users\DELL\Documents\ml\flowers\sunflower')

tulip_dir=os.path.join(r'C:\Users\DELL\Documents\ml\flowers\tulip')


train_tulip_names=os.listdir(tulip_dir)
print(train_tulip_names[:5])
train_sf_names=os.listdir(sunflower_dir)
print(train_sf_names[:5])
batch_size=16
train_datagen=ImageDataGenerator(rescale=1./255)
train_generator=train_datagen.flow_from_directory(r'C:\Users\DELL\Documents\ml\flowers',
        target_size=(350,350),batch_size=batch_size,color_mode='grayscale',
        classes=['daisy','dandelion','rose','sunflower','tulip'],class_mode="categorical")
target_size=(350,350)
model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3),activation='relu',input_shape=(350,350,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(5,activation='softmax')])
model.summary()
model.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(lr=0.001),
    metrics=['acc'])
total_sample=train_generator.n
num_epochs=5
model.fit_generator(train_generator,steps_per_epoch=int(total_sample/batch_size),
                    epochs=num_epochs,verbose=1)

model_json=model.to_json()
with open("modelGG.json",'w')as json_file:
    json_file.write(model_json)
model.save_weights("model1GG.h5")
print("saved model to disk")


import warnings
warnings.filterwarnings("ignore")
from keras.models import load_model
from time import sleep
from tensorflow.keras.utils import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
from keras.models import model_from_json
import tensorflow as tf

json_file=open(r'C:\Users\DELL\modelGG.json','r')
loaded_model_json=json_file.read()
json_file.close()

from tensorflow.keras.models import model_from_json
loaded_model=model_from_json(loaded_model_json)

loaded_model.load_weights(r'C:\Users\DELL\model1GG.h5')
print("loaded model from disk")

class_labels = ['daisy','dandelion','rose','sunflower','tulip']

import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.image as plt
test_image=image.load_img(r"C:\Users\DELL\Pictures\SUN.jfif", target_size=(200,200, 1))

test_image=np.array(test_image)
gray=cv2.cvtColor(test_image,cv2.COLOR_BGR2GRAY)
test_image=image.img_to_array(gray)
test_image=np.expand_dims(test_image,axis=0)
result=loaded_model.predict(test_image)
print(result)

a=list(result[0]).index(max(list(result[0])))
r=class_labels[a]
print(r)
