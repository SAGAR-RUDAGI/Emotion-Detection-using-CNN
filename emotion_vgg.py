import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os

NUM_CLASSES = 5 # As per dataset
IMG_ROWS, IMG_COLS = 48, 48
BATCH_SIZE = 32

#dataset link : https://drive.google.com/drive/folders/1E66iZdNz021aUZGsZjtc3EUu3NqAaIq3

train_dir = "C:\\Users\\Dell\\Desktop\\Projects\\Emotion Detection\\train"
validation_dir = "C:\\Users\\Dell\\Desktop\\Projects\\Emotion Detection\\validation"

train_datagen = ImageDataGenerator(
					rescale=1./255,
					rotation_range=30,
					shear_range=0.3,
					zoom_range=0.3,
					width_shift_range=0.4, 
					height_shift_range=0.4,
					horizontal_flip=True, 
					vertical_flip=True,
					fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
											color_mode='grayscale', 
											target_size=(IMG_ROWS,IMG_COLS), 
											batch_size=BATCH_SIZE, 
											class_mode='categorical', 
											shuffle=True)

validation_generator = validation_datagen.flow_from_directory(validation_dir,
											color_mode='grayscale', 
											target_size=(IMG_ROWS,IMG_COLS), 
											batch_size=BATCH_SIZE, 
											class_mode='categorical', 
											shuffle=True)

model = Sequential()

model.add(Conv2D(32, (3,3), padding='same', kernel_initializer='he_normal', input_shape=(IMG_ROWS,IMG_COLS,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3), padding='same', kernel_initializer='he_normal', input_shape=(IMG_ROWS,IMG_COLS,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))


model.add(Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))


model.add(Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal', input_shape=(IMG_ROWS,IMG_COLS,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal', input_shape=(IMG_ROWS,IMG_COLS,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))


model.add(Conv2D(256, (3,3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3,3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))


model.add(Flatten())
model.add(Dense(64,kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))


model.add(Dense(64,kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))


model.add(Dense(NUM_CLASSES, kernel_initializer='he_normal'))
model.add(Activation('softmax'))

print(model.summary())


checkpoint = ModelCheckpoint('emotion_vgg.h5', 
								monitor='val_loss', 
								mode='min',
								save_best_only=True,
								verbose=1)

earlystop = EarlyStopping(monitor='val_loss',
							min_delta=0,
							patience=6, 
							verbose=1, 
							restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
								factor=0.2,
								patience=3,
								verbose=1,
								min_delta=0.0001)

callbacks=[earlystop,checkpoint,reduce_lr]

model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])

train_samples = 24176
validation_samples = 3006
EPOCHS = 25

history = model.fit_generator(train_generator,
								steps_per_epoch=train_samples//BATCH_SIZE,
								epochs=EPOCHS,
								callbacks=callbacks,
								validation_data=validation_generator,
								validation_steps=validation_samples//BATCH_SIZE)
