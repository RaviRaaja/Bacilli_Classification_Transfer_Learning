import numpy as np
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten,Dense, GlobalAveragePooling2D,Input,BatchNormalization,Activation
from keras import backend as K
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau, LearningRateScheduler, TensorBoard, EarlyStopping, CSVLogger
from keras.layers.convolutional import Convolution2D,Conv2D, MaxPooling2D, ZeroPadding2D
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold


# In[14]:


batch_size = 32
epoch = 500

k = 10
img_width, img_height =224,224

def load_data_kfold(k):
    X_train = np.load('dataset.npy')
    temp = np.load('labels.npy')
    y_train = temp.reshape(temp.shape[0],)
    folds = list(StratifiedKFold(n_splits=k,
                                 shuffle=True,
                                 random_state=1).split(X_train,y_train))
    return folds, X_train, y_train

folds, X_train, y_train = load_data_kfold(k)
print (X_train.shape, y_train.shape)

def get_model():

	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=(224,224,3)))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	#model.add(Dropout(0.25))


	model.add(Conv2D(32, (3, 3)))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	#model.add(Dropout(0.25))


	model.add(Conv2D(64, (3, 3)))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	#model.add(Dropout(0.25))


	model.add(Conv2D(64, (3, 3)))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))


	model.add(Conv2D(64, (3, 3)))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))


	model.add(Conv2D(128, (3, 3)))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))




	model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
	model.add(Dense(128, activation='relu')) 
	#model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))

	sgd = optimizers.SGD(lr=0.01)

	model.compile(loss='binary_crossentropy',
             	optimizer=sgd,
             	metrics=['accuracy'])
	return model

train_data_gen = ImageDataGenerator(featurewise_center = True,
                                    samplewise_center = False,
                                    featurewise_std_normalization = True,
                                    samplewise_std_normalization = False,
                                    zca_whitening = False,
                                    rotation_range = 20,
                                    zoom_range = 0.2,
                                    width_shift_range = 0.15,
                                    height_shift_range = 0.15,
                                    horizontal_flip = True,
                                    vertical_flip = True)


def get_callbacks(name_weights, patience_lr, csvlog):
    mcp_save = ModelCheckpoint(name_weights, save_best_only=True,
                                monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, 
                                       patience = patience_lr,
                                       verbose=1,epsilon=1e-4,
                                       mode='min')
    csvlog = CSVLogger(csvlog, separator=',', append=False)

    return [mcp_save, reduce_lr_loss, csvlog]



for j, (train_idx, val_idx) in enumerate(folds):
    
    print('\nFold ',j)
    X_train_cv = X_train[train_idx]
    y_train_cv = y_train[train_idx]
    X_valid_cv = X_train[val_idx]
    y_valid_cv= y_train[val_idx]
    
    name_weights = "final_model_fold" + str(j) + "_weights.hdf5"
    csvlog = "fold_log_" + str(j) + ".csv"
    callbacks = get_callbacks(name_weights = name_weights, patience_lr=10, csvlog=csvlog)
    train_data_gen.fit(X_train_cv)
    train_generator = train_data_gen.flow(X_train_cv,y_train_cv, batch_size=batch_size)
    model = get_model()
    model.fit_generator(train_generator,
                        steps_per_epoch = len(X_train_cv)/batch_size,
                        epochs = epoch,
                        verbose = 1,
                        validation_data = (X_valid_cv,y_valid_cv),
                        callbacks = callbacks)

    print(model.evaluate(X_valid_cv,y_valid_cv))

