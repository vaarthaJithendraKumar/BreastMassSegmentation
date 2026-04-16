#from sgd import *
#schedule = SGDRScheduler(min_lr=1e-6,
#                             max_lr=1e-2,
#                             steps_per_epoch=np.ceil(epochs/batch_size),
#                             lr_decay=0.9,
#                             cycle_length=5,
#                             mult_factor=1.5)


import os
from skimage.io import imsave
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, Add, MaxPooling2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD, Adadelta, Adagrad
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from data import load_train_data, load_test_data
from sklearn.model_selection import train_test_split
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

# GPU memory growth configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU detected:", gpus[0].name)
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected — running on CPU")

img_rows = 256
img_cols = 256
smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def iou_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
  union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou

def loss(y_true, y_pred):
    return -(0.4*dice_coef(y_true, y_pred)+0.6*iou_coef(y_true, y_pred))


def get_unet():
    inputs = Input((img_rows, img_cols, 3))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    
    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    
    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)
    
    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)
        
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    model.compile(optimizer=Adam(), loss=[loss], metrics=[dice_coef, iou_coef])
    return model


name = os.getenv('DATASET_EXPERIMENT_NAME', 'cbis_ddsm')
train_dataset_names = [ds.strip() for ds in os.getenv('TRAIN_DATASETS', 'cbis_ddsm').split(',') if ds.strip()]
test_dataset_names = [ds.strip() for ds in os.getenv('TEST_DATASETS', 'cbis_ddsm').split(',') if ds.strip()]

if not train_dataset_names:
    raise ValueError('TRAIN_DATASETS is empty. Provide at least one dataset name.')
if not test_dataset_names:
    raise ValueError('TEST_DATASETS is empty. Provide at least one dataset name.')

train_sets = [load_train_data(dataset_name) for dataset_name in train_dataset_names]
imgs_train = np.concatenate([dataset_data[0] for dataset_data in train_sets], axis=0)
imgs_mask_train = np.concatenate([dataset_data[1] for dataset_data in train_sets], axis=0)

#imgs_train, imgs_mask_train = load_train_data(name)

fname = 'basic_unet_'+name+'_weights.h5'
pred_dir = fname[:-11]

imgs_train = imgs_train.astype('float32')

mean = np.mean(imgs_train)  # mean for data centering
std = np.std(imgs_train)  # std for data normalization

imgs_train -= mean
imgs_train /= std
imgs_mask_train = imgs_mask_train.astype('float32')

imgs_mask_train /= 255.  # scale masks to [0, 1]
imgs_mask_train = imgs_mask_train[..., np.newaxis]

imgs_train, imgs_val, imgs_mask_train, imgs_mask_val = train_test_split(imgs_train, imgs_mask_train, test_size=0.2, random_state=42)
print('-'*30)
print('Creating and compiling model...')
print('-'*30)

model = get_unet()

model_checkpoint = ModelCheckpoint(fname, monitor='val_loss', save_best_only=True)

print('-'*30)
print('Fitting model...')
print('-'*30)

history = model.fit(imgs_train, imgs_mask_train,
                    batch_size=16, epochs=100, verbose=1, shuffle=True,
                    validation_data=(imgs_val, imgs_mask_val),
                    callbacks=[model_checkpoint])

print('-'*30)
print('Loading and preprocessing test data...')
print('-'*30)

test_sets = [load_test_data(dataset_name) for dataset_name in test_dataset_names]
imgs_test = np.concatenate([dataset_data[0] for dataset_data in test_sets], axis=0)
imgs_id_test = np.concatenate([dataset_data[1] for dataset_data in test_sets], axis=0)

#imgs_test, imgs_id_test = load_test_data(name)

imgs_test = imgs_test.astype('float32')
imgs_test -= mean
imgs_test /= std

print('-'*30)
print('Loading saved weights...')
print('-'*30)
model.load_weights(fname)

print('-'*30)
print('Predicting masks on test data...')
print('-'*30)
imgs_mask_test = model.predict(imgs_test, verbose=1)
np.save('imgs_mask_test_'+name+'_basic_unet.npy', imgs_mask_test)


print('-' * 30)
print('Saving predicted masks to files...')
print('-' * 30)

if not os.path.exists(pred_dir):
    os.mkdir(pred_dir)

for idx, image in enumerate(imgs_mask_test):
    image = (image[:, :, 0] * 255.).astype(np.uint8)
    output_name = 'sample_{:04d}_pred.png'.format(idx)
    imsave(os.path.join(pred_dir, output_name), image)

imgs_id_test = imgs_id_test.astype('float32')
imgs_id_test = imgs_id_test[..., np.newaxis]
imgs_id_test = imgs_id_test // 255

ev = model.evaluate(imgs_test, imgs_id_test)
dice, iou = ev[1], ev[2]

print("dice score:", dice)
print("iou score:", iou)

   
plt.plot(history.history['dice_coef'])
plt.plot(history.history['val_dice_coef'])
plt.title('model dice coef')
plt.ylabel('dice coef')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['iou_coef'])
plt.plot(history.history['val_iou_coef'])
plt.title('model iou coef')
plt.ylabel('iou coef')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



