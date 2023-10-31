#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"


# In[6]:


import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# unconditional image captioning
inputs = processor(raw_image, return_tensors="pt").to("cuda")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))


# In[8]:


import requests
from PIL import Image
import os
import csv
from transformers import BlipProcessor, BlipForConditionalGeneration

# Initialize the BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")

# Folder containing your images
folder_path = "ADR image"

# Initialize an empty list to store image filenames and captions
image_captions = []

# Iterate through each image in the folder
for filename in os.listdir(folder_path):
    if filename.endswith((".jpg", ".jpeg")):
        # Load and process the image
        image_path = os.path.join(folder_path, filename)
        raw_image = Image.open(image_path).convert('RGB')
        inputs = processor(raw_image, return_tensors="pt").to("cuda")

        # Generate a caption for the image
        out = model.generate(**inputs)
        generated_caption = processor.decode(out[0], skip_special_tokens=True)

        # Append the filename and caption to the list
        image_captions.append((filename, generated_caption))

# Save the captions to a CSV file
output_csv = "blip_image_captions.csv"
with open(output_csv, 'w', newline='') as csvfile:
    caption_writer = csv.writer(csvfile)
    caption_writer.writerow(["Image Filename", "Caption"])
    caption_writer.writerows(image_captions)

print(f"Captions saved to {output_csv}")


# In[ ]:




