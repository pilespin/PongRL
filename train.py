
import tensorflow as tf
from tfHelper import tfHelper
import numpy as np
import os

import model
import predict
import common

path = './new/'

k = tf.keras
c = common

k.initializers.Ones()
# k.initializers.RandomUniform(minval=0.95, maxval=1.05, seed=None)
tfHelper.log_level_decrease()
tfHelper.numpy_show_entire_array(28)


# model = tfHelper.load_model("model")
model = model.model()

tensorBoard = k.callbacks.TensorBoard()

learning_rate_reduction = k.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                        patience=1, 
                                                        verbose=1, 
                                                        factor=0.5, 
                                                        min_lr=1e-09)

# datagen = k.preprocessing.image.ImageDataGenerator( 
                                                    # rotation_range=20,
                                                    # width_shift_range=0.1,
                                                    # height_shift_range=0.1,
                                                    # shear_range=0.2,
                                                    # zoom_range=0.1,
                                                    # horizontal_flip=True,
                                                    # fill_mode='nearest')
# datagen.fit(x_train)



print ("Load data ...")
x_train = []
y_train = []
for folder in os.listdir(path):
    if folder[0] != '.':
        print ("Load folder: " + folder)
        (x, y) = tfHelper.get_dataset_with_folder(path+folder + '/', c.convertColor)
        x = c.normalize(x)
        for i in x:
            x_train.append(i)
        for i in y:
            y_train.append(i)

x_train = np.array(x_train)
y_train = np.array(y_train)



for i in range(c.epochs):
    print("Epoch " + str(i) + '/' + str(c.epochs))
    # model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
    model.fit(x_train, y_train,
            batch_size=128,
            epochs=1,
            validation_data=(x_train, y_train),
            # validation_data=(x_test, y_test),
            shuffle=True,
            verbose=1,
            callbacks=[learning_rate_reduction, tensorBoard]
            )

    tfHelper.save_model(model, "model")
