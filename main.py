from data import load_doc,all_image_captions,save_descriptions,cleaning_text,text_vocabulary,load_photos,load_clean_descriptions,load_features,dict_to_list,create_tokenizer
from pickle import load,dump
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from keras.layers import Input,Dense,LSTM,Embedding,Dropout
from keras.layers import add
from keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

dataset_text = "Flickr8k_text"
dataset_images = "Flicker8k_Dataset"

filename = dataset_text + "/Flickr8k.token.txt"
descriptions = all_image_captions(filename)
print('Length of descriptions =',len(descriptions))

clean_descriptions = cleaning_text(descriptions)
vocabulary = text_vocabulary(clean_descriptions)
print('Length of vocabulary =',len(vocabulary))

save_descriptions(clean_descriptions,"descriptions.txt")

features = load(open("features/xception_features.p","rb"))

filename = dataset_text + "/" + "Flickr_8k.trainImages.txt"

train_imgs = load_photos(filename)
train_descriptions = load_clean_descriptions("descriptions.txt",train_imgs)
train_features = load_features(train_imgs)

tokenizer = create_tokenizer(train_descriptions)
#dump(tokenizer,open("artifacts/tokenizer.p","wb"))

vocab_size = len(tokenizer.word_index) + 1

def max_length(descriptions):
    desc_list = dict_to_list(descriptions)
    return max(len(d.split()) for d in desc_list)

max_length = max_length(train_descriptions)
print(max_length)

def create_sequences(tokenizer, max_length, desc_list, feature):
    X1, X2, y = list(), list(), list()
    for desc in desc_list:
        seq = tokenizer.texts_to_sequences([desc])[0]
        for i in range(len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            out_seq = to_categorical([out_seq],num_classes=vocab_size)[0]
            X1.append(feature)
            X2.append(out_seq)
    return np.array(X1),np.array(X2),np.array(y)


def data_generator(descriptions,features,tokenizer,max_length):
    def generator():
        while True:
            for key,description in descriptions.items():
                feature = features[key][0]
                input_image, input_sequence, output_word = create_sequences(tokenizer, max_length, descriptions, feature)
                for i in range(len(input_image)):
                    yield {'input_1':input_image[i], 'input_2':input_sequence[i]}, output_word[i]
    output_signature = (
        {
            'input_1': tf.TensorSpec(shape=(2048,),dtype=(tf.float32)),
            'input_2': tf.TensorSpec(shape=(max_length,),dtype=(tf.int32))
        },
        tf.TensorSpec(shape=(vocab_size,),dtype=(tf.float32)) 
    )

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    )
    return dataset.batch(32)

dataset = data_generator(train_descriptions, features, tokenizer, max_length)
for (a,b) in dataset.take(1):
    print(a['input_1'].shape, a['input_2'].shape, b.shape)
    break

def define_model(vocab_size, max_length):
    #CNN model from 2048 nodes to 256 nodes
    inputs1 = Input(shape=(2048,),name='input_1')
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    #LSTM sequence model
    inputs2 = Input(shape=(max_length,),name='input_2')
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    decoder1 = add([fe2, se3])
    decoder2 = Dense(256,activation='relu')(decoder1)
    outputs = Dense(vocab_size,activation='softmax')(decoder2)
    model = Model(inputs=[inputs1,inputs2], outputs=outputs)

    model.compile(loss='categorical_cross_entropy', optimizer='adam')
    print(model.summary())
    return model

