from data import load_doc,all_image_captions,save_descriptions,cleaning_text,text_vocabulary

dataset_text = "Flickr8k_text"
dataset_images = "Flicker8k_Dataset"

filename = dataset_text + "/Flickr8k.token.txt"
descriptions = all_image_captions(filename)
print('Length of descriptions =',len(descriptions))

clean_descriptions = cleaning_text(descriptions)
vocabulary = text_vocabulary(clean_descriptions)
print('Length of vocabulary =',len(vocabulary))

save_descriptions(clean_descriptions,"descriptions.txt")

