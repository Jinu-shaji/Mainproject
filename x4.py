import numpy as np

from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img,img_to_array
import tensorflow as tf
import cv2




imagePaths=["dset/1.png","dset/2.png","dset/3.png","dset/4.png","dset/5.png","dset/6.png","dset/7.png","dset/8.png"]
dd=[1,2,3,4,5,6,7,8]
lenn=8
img_size=160
img_sizemm=img_size*img_size
color_img = []
noiseimg=[]
p=0


def Plot(x, p, labels = False):
    plt.figure(figsize = (20, 2))
    for i in range(2):
        plt.subplot(1, 10, i + 1)
        plt.imshow(x[i].reshape(img_size,img_size), cmap = 'viridis')
        plt.xticks([])
        plt.yticks([])
        if labels:
            plt.xlabel(np.argmax(p[i]))
    plt.show()
for i in imagePaths:
    try:
        img_arr = cv2.imread(i,1)#convert BGR to RGB format
        gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        resized_arr = cv2.resize(gray, (img_size, img_size)) # Reshaping images to preferred size
        color_img.append([resized_arr, p])
        
        p+=1
    except Exception as e:
         print(e)

traindata=np.array(color_img)

x_train = []
y_train = []
x_val = []
y_val = []

for feature, label in traindata:
  x_train.append(feature)
  x_val.append(feature)
  y_train.append(label)
  y_val.append(label)

#normalize
x_train = np.array(x_train).astype('float')/255.
x_val  = np.array(x_val).astype('float')/255.
y_train=np.array(y_train)
y_test=np.array(y_val)

x_train = np.reshape(x_train, (lenn, img_sizemm))
x_test  = np.reshape(x_val, (lenn, img_sizemm))
# start adding noise
x_train_noisy = x_train + np.random.rand(lenn, img_sizemm)*0.2
x_test_noisy  = x_test + np.random.rand(lenn, img_sizemm)*0.2

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy  = np.clip(x_test_noisy, 0., 1.)
#end adding noise

Plot(x_train, None)
#Plot(x_train_noisy, None)

classifier = Sequential([
    Dense(256, activation = 'relu', input_shape = (img_sizemm,)),
    Dense(128, activation = 'relu'),
    Dense(lenn, activation = 'softmax')
])

classifier.compile(optimizer = 'adam',
                  loss = 'sparse_categorical_crossentropy',
                  metrics = ['accuracy'])

classifier.fit(x_train, y_train, batch_size = 512, epochs = 3)
loss, acc = classifier.evaluate(x_test, y_test)

input_image = Input(shape = (img_sizemm,))
encoded = Dense(64, activation = 'relu')(input_image)
decoded = Dense(img_sizemm, activation = 'sigmoid')(encoded)

autoencoder = Model(input_image, decoded)
autoencoder.compile(loss = 'binary_crossentropy', optimizer = 'adam')

autoencoder.fit(
    x_train_noisy, 
    x_train,
    epochs = 100,
    batch_size = 512,
    validation_split = 0.2,
    verbose = False,
    callbacks = [
        EarlyStopping(monitor = 'val_loss', patience = 5),
        LambdaCallback(on_epoch_end = lambda e,l: print('{:.3f}'.format(l['val_loss']), end = ' _ '))
    ]
)

print(' _ ')
print('Training is complete!')

input_image=Input(shape=(img_sizemm,))
x=autoencoder(input_image)
y=classifier(x)

denoise_and_classfiy = Model(input_image, y)

predictions=denoise_and_classfiy.predict(x_test_noisy)
#Plot(x_test_noisy, predictions, True)
preds = autoencoder.predict(x_test_noisy)
print(type(preds))
#Plot(x_test_noisy, None)

Plot(preds, None)















