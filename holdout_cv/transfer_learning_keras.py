
# coding: utf-8

# In[29]:


from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten,Dense, GlobalAveragePooling2D,Input
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.layers.convolutional import Convolution2D,Conv2D, MaxPooling2D, ZeroPadding2D
import tensorflow as tf


# In[5]:


img_width, img_height = 224, 224
train_data_dir = "data/train"
validation_data_dir = "data/validation"


# In[14]:


batch_size = 32
epochs = 50


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


path = '/home/raviraja/chestxray/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
base_model= VGG_16(path,(img_width, img_height,3))


# In[23]:


#base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_width,img_height,3))

for layer in base_model.layers:
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
model.compile(optimizer=SGD, loss='binary_crossentropy', metrics=["accuracy"])

print(model.summary())


# In[24]:


train_data_gen = ImageDataGenerator(featurewise_center = False,
                                    samplewise_center = False,
                                    featurewise_std_normalization = False,
                                    samplewise_std_normalization = False,
                                    zca_whitening = True,
                                    rotation_range = 10,
                                    zoom_range = 0.2,
                                    width_shift_range = 0.15,
                                    height_shift_range = 0.15,
                                    horizontal_flip = True,
                                    vertical_flip = True)

val_data_gen = ImageDataGenerator(featurewise_center = False,
                                    featurewise_std_normalization = False)


# In[25]:


train_generator = train_data_gen.flow_from_directory(train_data_dir,
                                                     batch_size = batch_size,
                                                     target_size=(img_width, img_height),
                                                     class_mode='binary')

validation_generator = train_data_gen.flow_from_directory(validation_data_dir,
                                                     batch_size = batch_size,
                                                     target_size=(img_width, img_height),
                                                     class_mode='binary')


# In[27]:


checkpointer = ModelCheckpoint(filepath='tl_vgg16.hdf5',verbose=1, save_best_only=True)

history = model.fit_generator(
        train_generator,
        steps_per_epoch = 40,
        epochs = epochs,
        validation_data=validation_generator,
        validation_steps= 17,
        callbacks=[checkpointer])

