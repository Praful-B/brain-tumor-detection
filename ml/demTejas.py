import tensorflow as tf
import os

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus :
    tf.config.experimental.set_memory_growth(gpu, True)   #Avoids the overuse of VRAM and prevents the Out Of Memory [OOM] Error !

import cv2
import imghdr
import numpy as np
from matplotlib import pyplot as plt

print("INITIALIZING THE PROCESS OF IMAGE VERIFICATION AND CORRPUPTED IMAGE DELETION")

# =========DATA CLEANING [I]=========

data = 'Training'
imgExts = ['jpeg', 'jpg', 'png', 'bmp', 'tiff']

for imgCls in os.listdir(data) :  # This will return the list of the files inside the TRAINING folder, stored under the data var !
    for img in os.listdir(os.path.join(data, imgCls)) :  # This will return the list of files inside each class folder !
        imgPath = os.path.join(data,imgCls,img)
        try :
            imgType = imghdr.what(imgPath)   # This will return the image type if it's a valid image, else it returns None !
            if imgType not in imgExts :      # If the image type is not in the list of valid image extensions !
                print('Deleting : ', imgPath)
                os.remove(imgPath)           # Delete the invalid image !
        except Exception as e :
            print('Error : ', e)


print('Image Verification and Deletion of Corrupted Images Completed !')

# ========DATA PREPARATION [II]=========
dataS = tf.keras.utils.image_dataset_from_directory('Training')

dataS_iterate = dataS.as_numpy_iterator()
batch = dataS_iterate.next()

fig, AX = plt.subplots(nrows=3, ncols=3, figsize=(5,5))
for i in range(3) :
    for j in range(3) :
        AX[i,j].imshow(batch[0][i*3 + j].astype(int))
        AX[i,j].axis('off')
plt.show()

dataS = dataS.map(lambda x,y: (x/255,y)) # Normalizing the images to the range [0,1]
dataS = dataS.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
print(dataS)
dataS_iterate = dataS.as_numpy_iterator()
batch = dataS_iterate.next()
print(batch[0].max())  # Should print 1.0 after normalization
print(batch[0].min())  # Should print 0.0 after normalization
print(batch[1])      # Labels of the batch
print('Data Preparation Completed !')
print('Image Verification and Deletion of Corrupted Images Started !')
print('Data Preparation Started !')

print('Data Preparation Completed !')

# ========DATA SPLITTING [III]=========

trainSize = int(len(dataS)*.7)
valSize = int(len(dataS)*.2)
testSize = int(len(dataS)*.1)

print('Train Size : ', trainSize)
print('Validation Size : ', valSize)
print('Test Size : ', testSize)

train = dataS.take(trainSize)
val = dataS.skip(trainSize).take(valSize)
test = dataS.skip(trainSize + valSize).take(testSize)

print("Data Splitting Completed !")

# ========MODEL BUILDING [IV]=========

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()

# First Convolutional Block

model.add(Conv2D(16, (3,3),1,activation = 'relu',padding = 'same',input_shape = (256,256,3)))
model.add(MaxPooling2D())

# Second Convolutional Block
model.add(Conv2D(32, (3, 3), 1, activation='relu', padding='same'))
model.add(MaxPooling2D())

# Third Convolutional Block
model.add(Conv2D(16, (3, 3), 1, activation='relu', padding='same'))
model.add(MaxPooling2D())

# Flatten and Dense Layers
model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.5))   
model.add(Dense(1, activation = 'sigmoid'))   # Signifies a YES/NO classification !

# Model Compilation
model.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])
model.summary()

print("Model Construction Completed !")

# ========MODEL TRAINING [V]=========

from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

longDir = 'logs'
tensorboardCallBack = TensorBoard(log_dir = longDir)
earlyStopping = EarlyStopping(monitor = 'val_loss',patience = 3,restore_best_weights = True)

print("Initialized Model Training ")

history = model.fit(train, epochs = 9, validation_data = val, callbacks = [tensorboardCallBack, earlyStopping])

print("Model Training Completed !")

# ========MODEL EVALUATION [VI]=========

loss, accuracy = model.evaluate(test)
print(f"Test Loss - {loss}\nTest Accuracy - {accuracy}")

# Plotting training history
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plotting Training and Validation Accuracy and Loss
axes[0].plot(history.history['accuracy'], label='Training Accuracy')
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()

# Plot loss
axes[1].plot(history.history['loss'], label='Training Loss')
axes[1].plot(history.history['val_loss'], label='Validation Loss')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()

plt.tight_layout()
plt.show()

# Saving the model
model.save('deMLon_model.h5')

#=========PREDICTION [VII]=========

def predictTumor(imgPath, model):
    """Predict if an image has tumor and return results"""
    img = cv2.imread(imgPath)
    if img is None: 
        print("Error: Unable to read image at", imgPath)
        return None, 0.0
    
    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0   # Image Normalization
    img = np.expand_dims(img, axis=0)  # Expanding dimensions
    
    prediction = model.predict(img, verbose=0)  # Fixed typo
    confidence = float(prediction[0][0])
    
    if confidence > 0.5:
        result = "TUMOR DETECTED"
        conf_display = confidence * 100
    else:
        result = "NO TUMOR"
        conf_display = (1 - confidence) * 100
    
    return result, conf_display, confidence

def visualizePrediction(imgPath, model):  # Added model parameter
    """Visualize prediction results"""
    result, conf_display, confidence = predictTumor(imgPath, model)
    
    if result is None:  # Handle error case
        return None, None
    
    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title(f"{result}\nConfidence: {conf_display:.2f}%", 
              fontsize=14, fontweight='bold')  # Fixed typo
    plt.axis('off')
    plt.show()
    
    return result, conf_display

#=========PREDICTION TESTING [VIII]=========

testImgPath = input("Enter the image path: ")
if os.path.exists(testImgPath): 
    print("Analyzing:", testImgPath)
    
    result, conf_display = visualizePrediction(testImgPath, model)  # Added model
    
    if result:  # Check if result is valid
        print(f"Prediction: {result} with confidence of {conf_display:.2f}%")
else:
    print("The provided image path does not exist. Please check and try again.")