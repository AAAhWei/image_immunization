# image_immunization
This repo is forked from [photoguard](https://github.com/MadryLab/photoguard) and uses the code of [attention-map](https://github.com/wooyeolBaek/attention-map/tree/main).

Reproducing "Distraction":\
The main file is `_photoguard/src/distraction.py`.

## Some problems to be fixed
1. The current code runs, but according to the original work, the victim model needs to be changed to inpaint/img2img. However, using the code from lines 13 ~ 22 causes issues (due to different input shapes https://huggingface.co/runwayml/stable-diffusion-inpainting).

2. Weighted Timestep Sampling

3. Try to change the order of mask and attn_map

4. As a global variable, does attn_maps retain data from the previous loop? And it need to be updated?