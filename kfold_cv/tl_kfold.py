import numpy as np
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten,Dense, GlobalAveragePooling2D,Input
from keras import backend as K
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau, LearningRateScheduler, TensorBoard, EarlyStopping, CSVLogger
from keras.layers.convolutional import Convolution2D,Conv2D, MaxPooling2D, ZeroPadding2D
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold


# In[14]:


batch_size = 64
epoch = 200

k = 5
#folds, X_train, y_train = load_data_kfold(k)
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

# In[30]:
def VGG_16(path,input_shape):
    img_input = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    
    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x) 
    inputs = img_input
    model = Model(inputs, x, name='vgg16')
    model.load_weights(path)
    return model
    


# In[ ]:

def get_model():
    path = '/home/raviraja/chestxray/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    base_model= VGG_16(path,(img_width, img_height,3))


    # In[23]:


    #base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_width,img_height,3))


    for layer in base_model.layers[:5]:
        layer.trainable = False

    #Adding custom Layers 
    x = base_model.output
    avg_pool1 = GlobalAveragePooling2D()(x)
    dense1 = Dense(1024, activation='relu')(avg_pool1)
    dense2 = Dense(128, activation='relu')(dense1)
    predictions = Dense(1, activation='sigmoid')(dense2)

    model = Model(inputs = base_model.input,
                  outputs = predictions)

    SGD = optimizers.SGD(lr=0.0001,momentum = 0.9)
    opt_adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=SGD, loss='binary_crossentropy', metrics=["accuracy"])

    print(model.summary())
    return model



# In[24]:


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

val_data_gen = ImageDataGenerator(featurewise_center = True,
				  featurewise_std_normalization = True)

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
    val_data_gen.fit(X_valid_cv)
    
    train_generator = train_data_gen.flow(X_train_cv,y_train_cv, batch_size=batch_size)
    validation_generator = val_data_gen.flow(X_valid_cv,y_valid_cv,batch_size=batch_size)
    
    model = get_model()
    
    model.fit_generator(train_generator,
                        steps_per_epoch = len(X_train_cv)/batch_size,
                        epochs = epoch,
                        verbose = 1,
                        validation_data = validation_generator,
                        callbacks = callbacks)

    print(model.evaluate(X_valid_cv,y_valid_cv))


