import json
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns

from IPython.display import display
from numpy.linalg import norm
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from umap import UMAP
from utils import bt_embedding_from_prediction_guard as bt_embeddings
from utils import encode_image
from utils import prepare_dataset_for_umap_visualization as data_prep
# from datasets import load_dataset


## ------------------------------------------------------ ##
url1 = 'http://farm3.staticflickr.com/2519/4126738647_cc436c111b_z.jpg'
cap1 = 'A motorcycle sits parked across from a herd of livestock'

url2 = 'http://farm3.staticflickr.com/2046/2003879022_1b4b466d1d_z.jpg'
cap2 = 'Motorcycle on platform to be worked on in garage'

url3 = 'http://farm1.staticflickr.com/133/356148800_9bf03b6116_z.jpg'
cap3 = 'a cat laying down stretched out near a laptop'

img1 = {'flickr_url' : url1, 'caption' : cap1, 'image_path' : './shared_data/motorcycle_1.jpg'}
img2 = {'flickr_url' : url2, 'caption' : cap2, 'image_path' : './shared_data/motorcycle_2.jpg'}
img3 = {'flickr_url' : url3, 'caption' : cap3, 'image_path' : './shared_data/cat_1.jpg'}

imgs = [img1, img2, img3]

for img in imgs:
    data = requests.get(img['flickr_url']).content

    with open(img['image_path'], 'wb') as f:
        f.write(data)

## ------------------------------------------------------ ##
for img in [img1, img2, img3]:
    image = Image.open(img['image_path'])
    caption = img['caption']
    display(image)
    display(caption)
    print()

## ------------------------------------------------------ ##
embeddings = []

for img in [img1, img2, img3]:
    img_path = img['image_path']
    caption = img['caption']
    base64_img = encode_image(img_path)
    embedding = bt_embeddings(caption, base64_img)
    embeddings.append(embedding)

## ------------------------------------------------------ ##
print(len(embeddings[0]))

## ------------------------------------------------------ ##
def cosine_similarity(vec1, vec2):
    similarity = np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

    return similarity

## ------------------------------------------------------ ##
ex1_embed = np.array(embeddings[0])
ex2_embed = np.array(embeddings[1])
ex3_embed = np.array(embeddings[2])

sim_ex1_ex2 = cosine_similarity(ex1_embed, ex2_embed)
sim_ex1_ex3 = cosine_similarity(ex1_embed, ex3_embed)

## ------------------------------------------------------ ##
print("Cosine similarity between ex1_embeded and ex2_embeded is: ")
display(sim_ex1_ex2)

print("Cosine similarity between ex1_embeded and ex3_embeded is: ")
display(sim_ex1_ex3)

## ------------------------------------------------------ ##
dist_ex1_ex2 = cv2.norm(ex1_embed, ex2_embed, cv2.NORM_L2)
dist_ex1_ex3 = cv2.norm(ex1_embed, ex3_embed, cv2.NORM_L2)

print("Euclidean distance between ex1_embeded and ex2_embeded is: ")
display(dist_ex1_ex2)

print("Euclidean distance between ex1_embeded and ex3_embeded is: ")
display(dist_ex1_ex3)

## ------------------------------------------------------ ##
# dataset_cats = load_dataset("yashikota/cat-image-dataset")
# dataset_cars = load_dataset("tanganke/stanford_cars")

# train_set_cats = dataset_cats["train"]
# train_set_cars = dataset_cars["train"]

cat_img_txt_pairs = data_prep("yashikota/cat-image-dataset", "cat", test_size = 50)

car_img_txt_pairs = data_prep("tanganke/stanford_cars", "car", test_size = 50)

## ------------------------------------------------------ ##
display(cat_img_txt_pairs[0]['caption'])
display(cat_img_txt_pairs[0]['pil_img'])

display(car_img_txt_pairs[0]['caption'])
display(car_img_txt_pairs[0]['pil_img'])

## ------------------------------------------------------ ##
cat_embeddings = []

for img_txt_pair in tqdm(cat_img_txt_pairs, total = len(cat_img_txt_pairs)):
    pil_img = img_txt_pair['pil_img']
    caption = img_txt_pair['caption']
    base64_img = encode_image(pil_img)
    embedding = bt_embeddings(caption, base64_img)
    cat_embeddings.append(embedding)

car_embeddings = []

for img_txt_pair in tqdm(car_img_txt_pairs, total = len(car_img_txt_pairs)):
    pil_img = img_txt_pair['pil_img']
    caption = img_txt_pair['caption']
    base64_img = encode_image(pil_img)
    embedding = bt_embeddings(caption, base64_img)
    car_embeddings.append(embedding)

## ------------------------------------------------------ ##
def dimensionality_reduction(embed_arr, label):
    X_scaled = MinMaxScaler().fit_transform(embed_arr)
    print(X_scaled)

    mapper = UMAP(n_components = 2, metric = "cosine").fit(X_scaled)
    df_emb = pd.DataFrame(mapper.embedding_, columns = ["X", "Y"])
    df_emb["label"] = label
    print(df_emb)

    return df_emb

## ------------------------------------------------------ ##
all_embeddings = np.concatenate([cat_embeddings, car_embeddings])

labels = ['cat'] * len(cat_embeddings) + ['car'] * len(car_embeddings)

reduced_dim_emb = dimensionality_reduction(all_embeddings, labels)

## ------------------------------------------------------ ##
fig, ax = plt.subplots(figsize = (8, 6))

sns.set_style("whitegrid", {'axes.grid' : False})
sns.scatterplot(data = reduced_dim_emb, x = reduced_dim_emb['X'], y = reduced_dim_emb['Y'],
                hue = 'label', palette = 'bright')
sns.move_legend(ax, "upper left", bbox_to_anchor = (1, 1))
plt.title('Scatter plot of images of cats and cars using UMAP')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
