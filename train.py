
import tensorflow as tf
from tfHelper import tfHelper
# import numpy as np

import model
import predict
import common
# import os
# import data
# import matplotlib

# from PIL import Image

path = './new/5/'

k = tf.keras
c = common

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
(x_train, y_train) = tfHelper.get_dataset_with_folder(path, c.convertColor)
x_train = c.normalize(x_train)
# print('x_train shape:', x_train.shape)



for i in range(c.epochs):
    print("Epoch " + str(i) + '/' + str(c.epochs))
    # model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
    model.fit(x_train, y_train,
            batch_size=128,
            epochs=50,
            validation_data=(x_train, y_train),
            # validation_data=(x_test, y_test),
            shuffle=True,
            verbose=1,
            callbacks=[learning_rate_reduction, tensorBoard]
            )

    tfHelper.save_model(model, "model")
