import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
device = "cuda:4"
sd = 'runwayml/stable-diffusion-v1-5'

leditspp 需要改成 resize 到 512x512