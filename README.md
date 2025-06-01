# Improving-Temporal-Consistency-of-VDMs-in-the-Domain-of-Sport-Exercise
# Introduction

This repository contains code and resources for improvement of temporal consistency in Video Diffusion Models (VDMs) within the sport and exercise domain. The project assesses motion learning, noise initialization, and prompt engineering for enhancement of quality and temporal consistency in sports video generation. The repository provides code for motion model training, video generation for assessment of these techniques, and calculation of metrics for evaluation of temporal consistency and video quality.

This README is structured into three main sections:
- **Motion Model Training**: Instructions for training of VDMs with motion learning techniques.
- **Inference (Video Generation)**: Instructions for video generation to assess motion learning, noise initialization, and prompt engineering pipelines.
- **Evaluation Metrics**: Instructions for calculation of metrics to assess the impact of these techniques on temporal consistency and video quality.

This work provides a framework for creation and evaluation of sports videos with improved temporal consistency, with all code and documentation for experimentation and replication.

To clone this repository and set up the environment, run the following commands:

```bash
# Clone the repository
git clone https://github.com/RonZatuchny/Improving-Temporal-Consistency-of-VDMs-in-the-Domain-of-Sport-Exercise.git
```
## Train MotionLoRAs
Below are the steps for fine tuning motion LoRAs for the following exercises
1. Clone the stable difussion v1-5 into the models/StableDiffusion directory
```bash
!apt-get update
!apt-get install git-lfs
!git lfs install
!git clone https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5 models/StableDiffusion
```
2. Get v3_sd15_mm motion module
```bash
!wget -O models/Motion_Module/v3_sd15_mm.ckpt https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_mm.ckpt
```
3. Install libraries from requirements.txt
```bash
!pip install -r requirements.txt
```
4. Run train.py with selected config
```bash
!python train.py --config ./configs/training/motion_director/decline_bench_press.yaml
```


## Inference
Below are steps for running inference for running the inference esperiments:
1. Install ComfyUI: https://github.com/comfyanonymous/ComfyUI
2. Install the following ComfyUI custom nodes:
    * ComfyUI video helper suite: https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
    * comfyui-manager: https://github.com/Comfy-Org/ComfyUI-Manager
    * comfyui-inspire-pack: https://github.com/ltdrdata/ComfyUI-Inspire-Pack
    * ComfyUI-AnimateDiff-Evolved: https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved

    Note: It is recomended to install those from the GUI, one can drop any of the provided workflow files and one would be prompted with installing the required nodes.

3. Get the following chckpoint files 
    * Put  vae-ft-mse-840000-ema-pruned.safetensors from https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors into ComfyUI/models/vae
    * Put v1-5-pruned-emaonly-fp16.safetensors from https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive/blob/main/v1-5-pruned-emaonly-fp16.safetensors into ComfyUI/models/checkpoints
    * Put v3_sd15_mm.ckpt from https://huggingface.co/guoyww/animatediff/blob/main/v3_sd15_mm.ckpt into ComfyUI/models/animatediff_models 
    * Put selected fine tuned motion LoRA safetensor files from models/MotionLoRA into  ComfyUI/models/animatediff_motion_lora
4. Save the .json workflow files from Workflows/ into ComfyUI/user/default/workflows
5. Save prompt files (example provided in Prompts/ ) into ComfyUI/custom_nodes/comfyui-inspire-pack/prompts
6. Start comfyUI
7. Select the workflow (Workflows for freeInit with with/ without and with/without FreeInit are provided)
8. Select appropriate values for the workflow:
    * Select the prompt file
    * Select the motion LoRA file
    * Select Prefix name in the workflow
9. Click to run the workflow

## Evaluation
Below are the steps for running the evaluation metrics on the output videos
1. Calculate FVD for videos
    * Install required libraries 
    ```bash
    !pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
    !pip install "numpy<2.0"
    !pip install opencv-python tqdm einops
    ```
    * Run the process_fvd.py file with the generated videos folder and a folder with reference videos, FVD scores for each subfolder will be printed in the output
    ```bash
    !python process_fvd.py "results_full/MotionLoRA/DefaultInit/PositivePrompts" "ExerciseVideos_Training"
    ```
2. Calcuate DINO and RAFT scores for videos
    * Install required libraries
    ```bash
    !pip install tensorflow==2.15 tensorflow-hub tensorflow-probability torch torchvision transformers numpy scipy Pillow opencv-python
    ```
    * Run the dino_score.py with the ganarated videos folder as an input, the avrage DINO score for each for each subfolder of videos would be computed and printed
    ```bash
    !python dino_score.py "output/MotionLoRA/DefaultInit/NeutralPrompts"
    ```

    * Run theraft_score.py with the ganarated videos folder as an input, the avrage DINO score for each for each subfolder of videos would be computed and printed
    ```bash
    !python raft_score.py "output/MotionLoRA/DefaultInit/NeutralPrompts"
    ```