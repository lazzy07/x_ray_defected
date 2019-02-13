from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import os
import sys


model = models.Sequential()
model.add(layers.Conv2D(100, (3, 3), activation='relu', input_shape=(300, 300, 3)))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(50, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(25, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(10, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(20, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

train_dir = './chest_xray/train/'
test_dir = './chest_xray/test/'

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2,
                                   height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2,
                                  height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(300, 300), batch_size=20, class_mode='binary')
validation_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(300, 300), batch_size=20, class_mode='binary')

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=0.001), metrics=['acc'])

history = model.fit_generator(train_generator, steps_per_epoch=1000,
                              epochs=3, validation_data=validation_generator, validation_steps=50)

model.save('model_02.h5')
