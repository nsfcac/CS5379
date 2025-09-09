import torch
from diffusers import StableDiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"


pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

prompt = "a photo of a gold fish flying in the sky"
image = pipe(prompt).images[0]  

image.save("gold_fish_flying_in_sky.png")

