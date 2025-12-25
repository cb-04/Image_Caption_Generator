from data import load_doc,all_image_captions,save_descriptions,cleaning_text,text_vocabulary
from utils import download_with_retry
from keras.applications.xception import Xception, preprocess_input
import os
import tqdm
from PIL import Image
import numpy as np

dataset_text = "Flickr8k_text"
dataset_images = "Flicker8k_Dataset"

filename = dataset_text + "/Flickr8k.token.txt"
descriptions = all_image_captions(filename)
print('Length of descriptions =',len(descriptions))

clean_descriptions = cleaning_text(descriptions)
vocabulary = text_vocabulary(clean_descriptions)
print('Length of vocabulary =',len(vocabulary))

save_descriptions(clean_descriptions,"descriptions.txt")

weights_url = "https://storage.googleapis.com/tensorflow/keras-applications/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5"
weights_path = download_with_retry(weights_url,'xception_weights.h5')
model = Xception(include_top=False, pooling='avg', weights=weights_path)

def extract_features(directory):
    features = {}
    valid_images = ['.jpg','.jpeg','.png']
    for img in tqdm(os.listdir(directory)):
        ext = os.path.splitext(img)[1].lower()
        if ext not in valid_images:
            continue
        filename = directory+"/"+img
        image = Image.open(filename)
        image = image.resize((299,299))
        image = np.expand_dims(image, axis=0)
        image = image/127.0
        image = image - 1.0

        feature = model.predict(image)
        features[img] = feature
    return features
