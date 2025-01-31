import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from sklearn.metrics import classification_report, log_loss, accuracy_score
from sklearn.model_selection import train_test_split

data_path = r"D:\Python\GestureRegognition\train_data2\data"

model_name_to_save = "gest_recogn_data2.h5"

Name=[]
for file in os.listdir(data_path):
    if file[-4:]!='pt.m' and file[-4:]!='.txt':
        Name+=[file]
print(Name)
print(len(Name))

N = []
for i in range(len(Name)):
    N += [i]

normal_mapping = dict(zip(Name, N))
reverse_mapping = dict(zip(N, Name))


def mapper(value):
    return reverse_mapping[value]

File=[]
for file in os.listdir(data_path):
    File+=[file]
    print(file)




dataset=[]
testset=[]
count=0
for file in File:
    path=os.path.join(data_path,file)
    t=0
    for im in os.listdir(path):
        if im[-4:]!='pt.m' and im[-4:]!='.txt':
            image=load_img(os.path.join(path,im), color_mode='rgb', target_size=(128,128))
            image=img_to_array(image)
            image=image/255.0
            if t<400:
                dataset.append([image,count])
            else:
                testset.append([image,count])
            t+=1
    count=count+1


plt.figure(figsize=(10, 10))
for i, category in enumerate(set(reverse_mapping.values())):
    plt.subplot(3, 4, i+1)
    category_images = [img for img, lbl in dataset if reverse_mapping[lbl] == category]
    if category_images:
        plt.imshow(category_images[0], cmap='hot')
        plt.xticks([])
        plt.yticks([])
        plt.title(category)
plt.tight_layout()
plt.show()

data,labels0=zip(*dataset)
test,tlabels0=zip(*testset)
labels1=to_categorical(labels0)
data=np.array(data)
labels=np.array(labels1)


tlabels1=to_categorical(tlabels0)
test=np.array(test)
tlabels=np.array(tlabels1)

X_train,X_test,y_train,y_test =train_test_split(data,labels,test_size=0.2,random_state=44)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(8, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    epochs=11,
                    batch_size=32,
                    validation_data=(X_test, y_test))

model.save(model_name_to_save)

plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()

plt.plot(model.history.history['accuracy'])
plt.plot(model.history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()

#calculate loss and accuracy on test data

test_loss, test_accuracy = model.evaluate(X_test, y_test)

print('Test accuracy: {:2.2f}%'.format(test_accuracy*100))

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

print(classification_report(y_true, y_pred_classes, target_names=list(reverse_mapping.values())))

misclassified_indices = np.where(y_pred_classes != y_true)[0]
plt.figure(figsize=(20, 4))
for i, idx in enumerate(misclassified_indices[:5]):
    plt.subplot(1, 5, i+1)
    plt.imshow(X_test[idx].reshape(128, 128, 3))
    plt.title(f"True: {mapper(y_true[idx])}\nPred: {mapper(y_pred_classes[idx])}")
    plt.axis('off')
plt.tight_layout()
plt.show()

















