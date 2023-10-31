#!/usr/bin/env python
# coding: utf-8

# In[17]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"


# In[18]:


# !git clone https://github.com/salesforce/LAVIS.git
# %cd LAVIS
# !pip install -e .


# In[19]:


# pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu102


# In[20]:


import torch


# In[21]:


torch.__version__


# In[28]:


import torch
from PIL import Image
# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
# load sample image
raw_image = Image.open("25.jpg").convert("RGB")
display(raw_image.resize((596, 437)))


# In[ ]:


import os
from PIL import Image
from matplotlib import pyplot as plt


# In[ ]:


# Install latest version of the library
get_ipython().system('pip install -q datasets==2.13.1')


# In[ ]:


pip install transformers


# In[ ]:


pip install albumentations


# In[ ]:


# Import important libraries
import pandas as pd
from datasets import load_dataset
import transformers
from transformers import BlipProcessor, BlipForImageTextRetrieval,BlipForConditionalGeneration, AutoProcessor
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize
import os

import gc
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
import albumentations as A
import cv2
import shutil
import json
from PIL import Image
import requests
from matplotlib import pyplot as plt


# In[ ]:


# Read CSV dataset from Pandas
df_train = pd.read_csv('ADR.csv') #, nrows = nRowsRead
df_train.dataframeName = 'ADR.csv'
nRow, nCol = df_train.shape
print(f'There are {nRow} rows and {nCol} columns')


# In[ ]:


df_train.head()


# In[ ]:


filtered_df=df_train


# In[ ]:


# filtered_df['image'] = "ADR image" +"/"+ filtered_df['Image']


# In[ ]:


filtered_df


# In[ ]:


# Create new directory for training images
folder_path = "train"
if not os.path.exists(folder_path):
    os.mkdir(folder_path)


# In[ ]:


# Iterate through the DataFrame and move the files to the destination folder
for index, row in filtered_df.iterrows():
    source_file = row["image"]
    file_name = os.path.basename(source_file)
    destination_file = os.path.join(folder_path, file_name)

    # Use shutil.move() to move the file
    shutil.copy(source_file, folder_path)


# In[ ]:


# Delete extra column from the dataframe
# filtered_df = filtered_df.drop(columns=["image"])


# In[ ]:


filtered_df.head()


# In[ ]:


# Convert dataframe to json format
captions = filtered_df.apply(lambda row: {"file_name": row["Image"], "text": row["Text"]}, axis=1).tolist()


# In[ ]:


# Save data to json file
with open(folder_path + "/metadata.jsonl", 'w') as f:
    for item in captions:
        f.write(json.dumps(item))


# In[ ]:


# Load dataset for training
dataset = load_dataset("imagefolder", data_dir=folder_path, split="train")
dataset


# In[ ]:


# Create class for training

class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor, image_size=(224, 224)):
        self.dataset = dataset
        self.processor = processor
        self.image_size = image_size
        self.resize_transform = Resize(image_size)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(images=item["image"], text=item["text"], padding="max_length", return_tensors="pt")
        # remove batch dimension
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        return encoding


# In[ ]:


from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
import requests


# In[ ]:


model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")


# In[ ]:


# from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
# import torch
# from PIL import Image
# import requests

# model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl")
# processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")


# In[ ]:


# Create dataset for training
image_size = (224, 224)
train_dataset = ImageCaptioningDataset(dataset, processor, image_size)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2)
type(train_dataloader)


# In[ ]:


# initialize the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.train()


# In[ ]:



# Start training
for epoch in range():
  print("Epoch:", epoch)
  for idx, batch in enumerate(train_dataloader):
    input_ids = batch.pop("input_ids").to(device)
    pixel_values = batch.pop("pixel_values").to(device)

    outputs = model(input_ids=input_ids,
                    pixel_values=pixel_values,
                    labels=input_ids)

    loss = outputs.loss

    #print("Loss:", loss.item())

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

  print("Loss:", loss.item())


# In[ ]:


#### Create save model path and save the trained model
saved_folder_path = "/content/saved_model"
if not os.path.exists(saved_folder_path):
    os.mkdir(saved_folder_path)

model.save_pretrained(saved_folder_path)
processor.save_pretrained(saved_folder_path)


# In[ ]:


### Load the trained model
load_model = AutoModelForCausalLM.from_pretrained(saved_folder_path)
load_processor = AutoProcessor.from_pretrained(saved_folder_path)


# In[ ]:


import os
from PIL import Image
import csv

# Folder containing your images
folder_path = "/content/drive/MyDrive/ADR image"

# Initialize an empty list to store image captions
captions = []

# Iterate through each image in the folder
for filename in os.listdir(folder_path):
    if filename.endswith((".jpg", ".jpeg")):
        # Open the image
        image_path = os.path.join(folder_path, filename)
        image = Image.open(image_path)

        # Prepare the image for the model
        inputs = processor(images=image, return_tensors="pt").to(device)
        pixel_values = inputs.pixel_values

        # Generate a caption
        generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Append the filename and caption to the list
        captions.append((filename, generated_caption))

# Save the captions to a CSV file with UTF-8 encoding
output_csv = "image_captions_gitft.csv"
with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    caption_writer = csv.writer(csvfile)
    caption_writer.writerow(["Image Filename", "Caption"])
    caption_writer.writerows(captions)

print(f"Captions saved to {output_csv}")


# In[ ]:


#### Create save model path and save the trained model
saved_folder_path = "/content/saved_model"
if not os.path.exists(saved_folder_path):
    os.mkdir(saved_folder_path)

model.save_pretrained(saved_folder_path)
processor.save_pretrained(saved_folder_path)


# In[ ]:


### Load the trained model
load_model = AutoModelForCausalLM.from_pretrained(saved_folder_path)
load_processor = AutoProcessor.from_pretrained(saved_folder_path)


# In[ ]:


import os
from PIL import Image
import csv

# Folder containing your images
folder_path = "/content/drive/MyDrive/ADR_test"

# Initialize an empty list to store image captions
captions = []

# Iterate through each image in the folder
for filename in os.listdir(folder_path):
    if filename.endswith((".jpg", ".jpeg")):
        # Open the image
        image_path = os.path.join(folder_path, filename)
        image = Image.open(image_path)

        # Prepare the image for the model
        inputs = processor(images=image, return_tensors="pt").to(device)
        pixel_values = inputs.pixel_values

        # Generate a caption
        generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Append the filename and caption to the list
        captions.append((filename, generated_caption))

# Save the captions to a CSV file with UTF-8 encoding
output_csv = "image_captions.csv"
with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    caption_writer = csv.writer(csvfile)
    caption_writer.writerow(["Image Filename", "Caption"])
    caption_writer.writerows(captions)

print(f"Captions saved to {output_csv}")

