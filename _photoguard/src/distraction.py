import torch
from torch import nn
from torch.autograd import Variable
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionInpaintPipeline, UNet2DConditionModel, AutoencoderKL, PNDMScheduler
from PIL import Image
import numpy as np
from tqdm import trange
from attention_utils import *

cross_attn_init()
# Load the components of Stable Diffusion
# TODO: pipe_inpaint need 5 channels (4: masked images, 1: mask)
# pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
#     "runwayml/stable-diffusion-inpainting",
#     revision="fp16",
#     torch_dtype=torch.float16,
# )
# vae = pipe_inpaint.vae.to("cuda")
# unet = pipe_inpaint.unet.to("cuda")
# text_encoder = pipe_inpaint.text_encoder.to("cuda")
# tokenizer = pipe_inpaint.tokenizer
# scheduler = pipe_inpaint.scheduler
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", torch_dtype=torch.float16).to("cuda")
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", torch_dtype=torch.float16).to("cuda")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
scheduler = PNDMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

# Load and preprocess the input image
init_image = Image.open('../assets/trevor_5.jpg').convert('RGB').resize((512, 512))
width, height = init_image.size
init_image = np.array(init_image).astype(np.float32) / 255.0
init_image = torch.tensor(init_image).permute(2, 0, 1).unsqueeze(0).to("cuda")

content_mask = None  # Provide the content mask M

# Encode the prompt
### seed
prompt = "two men in front of the building" # 1. original caption 2. editing caption (w/ ca) 3. only ca
ca = "men"  # Provide the focal content ca
text_input = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
text_embeddings = text_encoder(text_input.input_ids.to("cuda"))[0].to(torch.float16)

# Parameters
N = 100
# T = torch.linspace(0, scheduler.config.num_train_timesteps - 1, steps=10).long()
T = torch.randint(low=0, high=scheduler.config.num_train_timesteps - 1, size=(10,)).long()
# TODO: not uniform sampling => transfer to weighted (annealing-like) sampling 
epsilon = 0.06
alpha = 0.01
threshold = 0.5

adv_image = init_image.clone().detach().to(torch.float16)

unet = set_layer_with_name_and_path(unet)
unet = register_cross_attention_hook(unet)

for n in trange(N):
    all_grad = torch.zeros_like(adv_image).to("cuda")
    all_loss = 0
    for t in T:
        # TODO: experiment to mask image first and make attention map
        ### adv_image = content_mask * adv_image 

        adv_image = adv_image.to(torch.float16)
        adv_image.requires_grad = True
        
        # Forward pass
        latents = vae.encode(adv_image).latent_dist.sample()
        noise = torch.randn_like(latents).to("cuda")
        latents = scheduler.add_noise(latents, noise, t)
        latents = latents * vae.config.scaling_factor

        noise_pred = unet(latents, t, encoder_hidden_states=text_embeddings)["sample"]
        if (n == 0 or n == N-1):
            save_by_timesteps(tokenizer, prompt, height, width, f'attention_{n}')
        attn_maps = get_average_attn_map(tokenizer, prompt, ca, height, width)

        if content_mask == None:
            content_mask = (attn_maps > threshold).float()
        #     all_mask = (attn_maps > 0).float()
        # print(content_mask.sum(), all_mask.sum())
        
        # Calculate the attention suppressing loss
        masked_attn_maps = attn_maps * content_mask
        loss = torch.norm(masked_attn_maps, p=1)
        grad = torch.autograd.grad(
                loss, adv_image, retain_graph=False, create_graph=False
            )[0]
        
        all_loss += loss
        all_grad += grad
        adv_image = adv_image.detach()

    print(all_loss)
    # Update adv_image
    all_grad /= len(T)
    adv_image = adv_image.detach() - alpha * all_grad.sign()
    delta = torch.clamp(adv_image - init_image, min=-epsilon, max=epsilon)
    adv_image = torch.clamp(init_image + delta, min=0, max=1).detach()

immunized_image = adv_image.detach().cpu().permute(0, 2, 3, 1).numpy()[0] * 255.0
immunized_image = Image.fromarray(immunized_image.astype(np.uint8))
immunized_image.save("immunized_image.png")
