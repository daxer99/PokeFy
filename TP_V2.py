import pandas as pd
from keras.models import *
from keras.layers import *
import tensorflow as tf

directory = '.../PokemonData'

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(directory,
                                            seed = 2059,
                                            image_size = (120, 120),
                                            batch_size = 32,
                                            validation_split = 0.2,
                                            subset = 'training')


valid_dataset = tf.keras.preprocessing.image_dataset_from_directory(directory,
                                            seed = 2059,
                                            image_size = (120, 120),
                                            batch_size = 32,
                                            validation_split = 0.2,
                                            subset= 'validation')

labels = train_dataset.class_names


model = Sequential()

model.add(Conv2D(32, kernel_size = (3, 3), input_shape = (120, 120, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(128, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(256, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(512, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(128, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
model.add(MaxPooling2D(pool_size = (2, 2), padding = 'same'))
model.add(BatchNormalization())

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(256, activation = 'relu'))

model.add(Dropout(0.2))

model.add(Dense(len(labels), activation = 'softmax'))

model.summary()

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01),
             loss = 'sparse_categorical_crossentropy',
             metrics = ['accuracy'])

history = model.fit(x = train_dataset,
                   epochs = 60,
                   validation_data = valid_dataset)

history_df = pd.DataFrame(history.history)
history_df.to_csv('.../history_poke_v2.csv')

model.save('.../poke_v2_model.h5')