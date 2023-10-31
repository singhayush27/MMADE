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


# In[29]:


from lavis.models import load_model_and_preprocess
# loads InstructBLIP model
model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5_instruct", model_type="flant5xl", is_eval=True, device=device)
# prepare the image
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)


# In[30]:


model.generate({"image": image, "prompt": "What is unusual about this image?"})


# In[31]:


model.generate({"image": image, "prompt": "Write a short description for the image."})


# In[ ]:


# pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116


# In[ ]:


from transformers import AutoTokenizer
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
if processor.tokenizer.eos_token is None:
    processor.tokenizer.eos_token = '<|eos|>'
model = BlipForConditionalGeneration.from_pretrained("prasanna2003/Instruct-blip-v2")


# In[ ]:


image_paths=[]
raw_images=[]
for img_path in image_paths:
    raw_image = Image.open(img_path).convert("RGB")
    raw_images.append(raw_image)
    display(raw_image.resize((596, 437)))


# In[ ]:


import os
from PIL import Image

# Get the path to the folder containing the image files.
folder_path = "ADR_test"

# List the files in the folder.
files = os.listdir(folder_path)


image_files = [f for f in files]

# Add the full path to each image file to the image_paths list.
image_paths = []
for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    image_paths.append(image_path)

# Open each image file and append it to the raw_images list.
raw_images = []
for image_path in image_paths:
    raw_image = Image.open(image_path).convert("RGB")
    raw_images.append(raw_image)

# Display each image in the raw_images list.
for raw_image in raw_images:
    display(raw_image.resize((596, 437)))


# In[ ]:


image = Image.open('image/39.jpeg').convert('RGB')
for image in raw_images:
    prompt = f"""
    Input: Describe this image.
    """

    inputs = processor(image, prompt, return_tensors="pt")
    
    output = model.generate(**inputs, max_length=100)
    print(processor.tokenizer.decode(output[0]))


# In[ ]:


import csv

# Open a new CSV file for writing.
with open("output.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)

    # Write the header row to the CSV file.
    writer.writerow(["Output"])

    # Iterate over the raw_images list and write the output of the model to the CSV file.
    for image in raw_images:
        prompt = f"""Instruction: Answer the following input according to the image.
        Input: Describe this image.
        output: """

        inputs = processor(image, prompt, return_tensors="pt")

        output = model.generate(**inputs, max_length=100)

        # Write the output of the model to the CSV file.
        writer.writerow([processor.tokenizer.decode(output[0])])

# Close the CSV file.
csvfile.close()


# In[ ]:


# display(raw_image.resize((596, 437)))

print(processor.tokenizer.decode(output[0]).split('output :')[-1].split('.')[0])


# In[ ]:


print(processor.tokenizer.decode(output[0]).split('output :')[-1])


# In[ ]:


from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
if processor.tokenizer.eos_token is None:
    processor.tokenizer.eos_token = '<|eos|>'
model = BlipForConditionalGeneration.from_pretrained("prasanna2003/Instruct-blip-v2")

image = Image.open('image/39.jpeg').convert('RGB')

prompt = """Instruction: Answer the following input according to the image.
Input: Describe this image.
output: """

inputs = processor(image, prompt, return_tensors="pt")

output = model.generate(**inputs, max_length=100)
print(processor.tokenizer.decode(output[0]))


# In[ ]:




