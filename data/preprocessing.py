from pickle import load
import os

dataset_images = "Flicker8k_Dataset"

def load_doc(filename):
    file = open(filename,'r')
    text = file.read()
    file.close()
    return text

def all_image_captions(filename):
    file = load_doc(filename)
    captions = file.split('\n')
    descriptions = {}
    for caption in captions[:-1]:
        img, caption = caption.split('\t')
        if img[:-2] not in descriptions:
            descriptions[img[:-2]] = [caption]
        else:
            descriptions[img[:-2]].append(caption)
    return descriptions

def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + '\t' + desc)
    data = '\n'.join(lines)
    file = open(filename,'w')
    file.write(data)
    file.close()

def load_photos(filename):
    file = load_doc(filename)
    photos = file.split("\n")[:-1]
    photos_present = [photo for photo in photos if os.path.exists(os.path.join(dataset_images,photo))]
    return photos_present

def load_clean_descriptions(filename,photos):
    file = load_doc(filename)
    descriptions = {}
    for line in file.split("\n"):
        words = line.split()
        if len(words)<1:
            continue

        image, image_caption = words[0],words[1:]
        if image in photos:
            if image not in descriptions:
                descriptions[image] = []
            desc = '<start> '+" ".join(image_caption)+' <end>'
            descriptions[image].append(desc)
    return descriptions

def load_features(photos):
    all_features = load(open("features/xception_features.p","rb"))
    features = {k:all_features[k] for k in photos}
    #print(features)
    return features

def dict_to_list(descriptions):
    all_desc = []
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc