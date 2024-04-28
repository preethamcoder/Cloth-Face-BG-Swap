from flask import Flask, render_template, request, redirect, url_for
import os
import torch
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, to_tensor

import PIL, cv2
from PIL import Image

from io import BytesIO
import base64, json, requests
from matplotlib import pyplot as plt

import numpy as np
import copy

from numpy import asarray

import sys

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

from diffusers import StableDiffusionInpaintPipeline, EulerDiscreteScheduler

model_dir="stabilityai/stable-diffusion-2-inpainting"

scheduler = EulerDiscreteScheduler.from_pretrained(model_dir, subfolder="scheduler")

pipe = StableDiffusionInpaintPipeline.from_pretrained(model_dir, 
                                                   scheduler=scheduler,
                                                   revision="fp16",
                                                   torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.enable_xformers_memory_efficient_attention()

target_width, target_height = 512,512

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(enumerate(anns), key=(lambda x: x[1]['area']), reverse=True)
    ax = plt.gca()

    ax.set_autoscale_on(False)

    for original_idx, ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))

        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]

        ax.imshow(np.dstack((img, m*0.35)))

        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = contours[0]
            M = cv2.moments(cnt)

            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                ax.text(cx, cy, str(original_idx), color='white', fontsize=16, ha='center', va='center', fontweight='bold')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
    if file:
        s_m = Image.open(file)
        width, height = s_m.size
        s_m = s_m.crop((0, height-width , width , height))
        s_m = s_m.resize((target_width, target_height), Image.LANCZOS )
        filepath = os.path.join('static', file.filename)
        s_m.save(filepath)
        seg = asarray(s_m)
        masks = mask_generator.generate(seg)
        plt.figure(figsize=(5,5))
        plt.imshow(s_m)
        show_anns(masks)  # use masks[:-x] to only display the first masks of the list
        plt.axis('off')
        filepath = os.path.join('static', "masks.jpg")
        plt.savefig(filepath)
        return redirect(url_for('segment', filename=file.filename))
    return redirect(request.url)

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)


mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.999, 
    stability_score_thresh=0.9,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
)

@app.route('/segment/<filename>', methods=['GET', 'POST'])
def segment(filename):
    # print(request.files)
    # if 'image' not in request.files:
    #     return redirect(request.url)
    # file = request.files['image']
    # if file.filename == '':
    #     return redirect(request.url)
    # if file:
    #     s_m = Image.open(file)
    #     seg = asarray(s_m)
    #     masks = mask_generator.generate(seg)
    #     plt.figure(figsize=(20,20))
    #     plt.imshow(s_m)
    #     show_anns(masks)  # use masks[:-x] to only display the first masks of the list
    #     plt.axis('off')
    #     filepath = os.path.join('static', "masks.jpg")
    #     plt.savefig(filepath)
    #     print("SAVED THE NIGGA")
    return render_template('segment.html', filename=filename)



@app.route('/results/<filename>')
def results(filename):
    return render_template('results.html', filename=filename)


if __name__ == '__main__':
    app.run(debug=True)