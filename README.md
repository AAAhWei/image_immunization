# image_immunization
This repo is forked from [photoguard](https://github.com/MadryLab/photoguard) and uses the code of [attention-map](https://github.com/wooyeolBaek/attention-map/tree/main).

Reproducing "Distraction":\
The main file is `_photoguard/src/distraction.py`.

## Some problems to be fixed
1. The current code runs, but according to the original work, the victim model needs to be changed to inpaint/img2img. However, using the code from lines 13 ~ 22 causes issues (possibly due to different input shapes or architecture differences?).

2. Is the timestep too large? It has been observed that after around timestep 400, the attention map can no longer be referenced to original image.

3. The current loss is not converging. It might be necessary to consider whether the content mask (related to threshold) is correctly mapped to the appropriate locations or to check the algorithm.

4. As a global variable, does attn_maps retain data from the previous loop? And it need to be updated?