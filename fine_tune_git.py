# -*- coding: utf-8 -*-
"""Fine_Tune_GIT.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KJYFFqiQUxsy960tYKA7VLkfcDm_tvdu
"""

# Install latest version of the library
!pip install -q datasets==2.13.1

pip install transformers

import pandas as pd
from datasets import load_dataset
import transformers
from transformers import AutoProcessor
from transformers import AutoModelForCausalLM
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

# Read CSV dataset from Pandas
df_train = pd.read_csv('/content/adr_train.csv') #, nrows = nRowsRead
df_train.dataframeName = 'adr_train.csv'
nRow, nCol = df_train.shape
print(f'There are {nRow} rows and {nCol} columns')

df_train.dataframeName

df_train.head()

filtered_df=df_train

filtered_df['image'] = "/content/drive/MyDrive/ADR_train" +"/"+ filtered_df['Image']
# filtered_df=filtered_df.drop(['images'], axis=1)

filtered_df.head()

# Create new directory for training images
folder_path = "/content/train"
if not os.path.exists(folder_path):
    os.mkdir(folder_path)

# Iterate through the DataFrame and move the files to the destination folder
for index, row in filtered_df.iterrows():
    source_file = row["image"]
    file_name = os.path.basename(source_file)
    destination_file = os.path.join(folder_path, file_name)

    # Use shutil.move() to move the file
    shutil.copy(source_file, folder_path)

# Delete extra column from the dataframe
filtered_df = filtered_df.drop(columns=["image"])

filtered_df.head()

# Convert dataframe to json format
captions = filtered_df.apply(lambda row: {"file_name": row["Image"], "text": row["Text"]}, axis=1).tolist()

# Save data to json file
with open(folder_path + "/metadata.jsonl", 'w') as f:
    for item in captions:
        f.write(json.dumps(item))

# Load dataset for training
dataset = load_dataset("imagefolder", data_dir=folder_path, split="train")
dataset

from torch.utils.data import Dataset

class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        encoding = self.processor(images=item["image"], text=item["text"], padding="max_length", return_tensors="pt")

        # remove batch dimension
        encoding = {k:v.squeeze() for k,v in encoding.items()}

        return encoding

from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("microsoft/git-base")

train_dataset = ImageCaptioningDataset(dataset, processor)

item = train_dataset[0]
for k,v in item.items():
  print(k,v.shape)

from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2)

batch = next(iter(train_dataloader))
for k,v in batch.items():
  print(k,v.shape)

processor.decode(batch["input_ids"][0])

from PIL import Image
import numpy as np

MEAN = np.array([123.675, 116.280, 103.530]) / 255
STD = np.array([58.395, 57.120, 57.375]) / 255

unnormalized_image = (batch["pixel_values"][0].numpy() * np.array(STD)[:, None, None]) + np.array(MEAN)[:, None, None]
unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
Image.fromarray(unnormalized_image)

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")

outputs = model(input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                labels=batch["input_ids"])
outputs.loss

import torch

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

model.train()

for epoch in range(20):
  print("Epoch:", epoch)
  for idx, batch in enumerate(train_dataloader):
    input_ids = batch.pop("input_ids").to(device)
    pixel_values = batch.pop("pixel_values").to(device)

    outputs = model(input_ids=input_ids,
                    pixel_values=pixel_values,
                    labels=input_ids)

    loss = outputs.loss

    # print("Loss:", loss.item())

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()
  print("Loss:", loss.item())

import os

#### Create save model path and save the trained model
saved_folder_path = "/content/saved_model"
if not os.path.exists(saved_folder_path):
    os.mkdir(saved_folder_path)

model.save_pretrained(saved_folder_path)
processor.save_pretrained(saved_folder_path)

### Load the trained model
load_model = AutoModelForCausalLM.from_pretrained(saved_folder_path)
load_processor = AutoProcessor.from_pretrained(saved_folder_path)

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
output_csv = "image_captions_gitft.csv"
with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    caption_writer = csv.writer(csvfile)
    caption_writer.writerow(["Image Filename", "Caption"])
    caption_writer.writerows(captions)

print(f"Captions saved to {output_csv}")

