#import trained Haar Cascade models
import cv2
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from sklearn.preprocessing import Normalizer
from scipy.spatial.distance import cosine
import csv
in_encoder = Normalizer('l2')

# set parameter
path = "/Users/austin/Desktop/NUS/Project/"
probability_minimum = 0.5
threshold = 0.3
database = {}

# Deefine facenet loss function
def triplet_loss(y_true, y_pred, alpha = 0.2):

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    # Step 1: Compute the (encoding) distance between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))

    return loss

#Load model
model_Face = load_model(path+"facenet_keras.h5",custom_objects={ 'loss': triplet_loss })
#model2 = load_model("/Users/austin/Desktop/NUS/Project/mask_recog_ver2.h5",custom_objects={ 'loss': triplet_loss })
#to read the classifier
faceCascade = cv2.CascadeClassifier(path+'haarcascade_frontalface_alt2.xml')
#to load yolov4
with open('classes.names') as f:
    # Getting labels reading every line
    # and putting them into the list
    labels = [line.strip() for line in f]

network = cv2.dnn.readNetFromDarknet(path+'yolov4_custom_train.cfg',
                                     path+'yolov4_custom_train_last.weights')
layers_names_all = network.getLayerNames()

layers_names_output = \
    [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]


#get face embedding and perform face recognition
def get_embedding(image):
    # scale pixel values
    face = image.astype('float32')
    # standardization
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    face = cv2.resize(face,(160,160))
    face = np.expand_dims(face, axis=0)
    encode = model_Face.predict(face)[0]
    return encode

def find_person(encoding):
    min_dist = float("inf")
    encoding = in_encoder.transform(np.expand_dims(encoding, axis=0))[0]
    for (name, db_enc) in database.items():
        dist = cosine(db_enc,encoding)
        if dist < 0.5 and dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > 0.5:
        return "None"
    else:
        return identity
    return "None"

#load facenet pre-train database
reader = csv.reader(open(path+'dict.csv'), delimiter='\n')
for row in reader:
    data = row[0].split(",", 1)
    encode = data[1].replace('"','')
    encode = encode.replace('[','')
    encode = encode.replace(']','')
    encode = np.fromstring(encode, dtype=float, sep=',')
    database[data[0]] = encode


#use openCV camera
video_capture = cv2.VideoCapture(0)

while True:
  # Capture frame-by-frame
  ret, frame = video_capture.read()

  #convert to greyscale
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  #detect face
  faces = faceCascade.detectMultiScale(gray,
                                       scaleFactor=1.1,
                                       minNeighbors=5,
                                       minSize=(60, 60),
                                       flags=cv2.CASCADE_SCALE_IMAGE)

  faces_list=[]
  encodes=[]
  name = ""

  #to draw rectangle
  for (x, y, w, h) in faces:

    face_frame = frame[y:y+h,x:x+w]
    face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
    face_frame = img_to_array(face_frame)
    face_frame = cv2.resize(face_frame,(160, 160))
    face = np.expand_dims(face_frame, axis=0)
    face =  preprocess_input(face)

    faces_list.append(face)
    if len(faces_list)>0:
        encode = get_embedding(face_frame)
        encodes.append(encode)
        #preds = model.predict(faces_list)
    for pred in encodes:
        #mask = model2.predict(pred)
        name = find_person(pred)
        #(mask, withoutMask) = pred
    if name == "None":
        label = "Not found"
    else:
        label = name
    #label = "Not found" if name == false else label = name
    color = (0, 0, 255)  if label == "Not found" else (0, 255, 0)
    #label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
    #label = "{}: {:.2f}%".format(label)
    cv2.putText(frame, label, (x, y- 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

    cv2.rectangle(frame, (x, y), (x + w, y + h),color, 2)

  cv2.imshow('Video', frame)
  #quit when received "Q" key
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

#to clean up
video_capture.release()
cv2.destroyAllWindows()
