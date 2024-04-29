from flask import Flask, render_template, request, redirect, url_for
import os
from helper import *
from PIL import Image
from numpy import asarray
from matplotlib import pyplot as plt


target_width, target_height = 512,512
app = Flask(__name__)

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
model_type = "vit_h"

sam_checkpoint = "sam_vit_h_4b8939.pth"

device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
global fn

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.999, 
    stability_score_thresh=0.9,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
)


def create_image_grid(original_image, images, names, rows, columns):
    names = copy.copy(names)  # Create a copy of the names list to avoid modifying the external variable
    images = copy.copy(images)  # Create a copy of the images list to avoid modifying the external variable

    # Check if images is a tensor
    if torch.is_tensor(images):
        # Check if the number of tensor images and names is equal
        assert images.size(0) == len(names), "Number of images and names should be equal"

        # Check if there are enough images for the specified grid size
        assert images.size(0) >= (rows * columns) - 1 - 1, "Not enough images for the specified grid size"

        # Convert tensor images to PIL images and apply sigmoid normalization
        images = [to_pil_image(torch.sigmoid(img)) for img in images]
    else:
        # Check if the number of PIL images and names is equal
        assert len(images) == len(names), "Number of images and names should be equal"

    # Check if there are enough images for the specified grid size
    assert len(images) >= (rows * columns) - 1 - 1, "Not enough images for the specified grid size"

    # Add the original image to the beginning of the images list
    images.insert(0, original_image)

    # Add an empty name for the original image to the beginning of the names list
    names.insert(0, 'Original')

    # Create a figure with specified rows and columns
    fig, axes = plt.subplots(rows, columns, figsize=(15, 15))

    # Iterate through the images and names
    for idx, (img, name) in enumerate(zip(images, names)):
        # Calculate the row and column index for the current image
        row, col = divmod(idx, columns)

        # Add the image to the grid
        axes[row, col].imshow(img, cmap='gray' if idx > 0 and torch.is_tensor(images) else None)

        # Set the title (name) for the subplot
        axes[row, col].set_title(name)

        # Turn off axes for the subplot
        axes[row, col].axis('off')

    # Iterate through unused grid cells
    for idx in range(len(images), rows * columns):
        # Calculate the row and column index for the current cell
        row, col = divmod(idx, columns)

        # Turn off axes for the unused grid cell
        axes[row, col].axis('off')

    # Adjust the subplot positions to eliminate overlaps
    plt.tight_layout()
    plt.savefig('static/res.jpg')

@app.route('/')
def home():
    return render_template('home.html')

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
        filepath = os.path.join('static', 'me.jpg')
        s_m.save(filepath)
        seg = asarray(s_m)
        masks = mask_generator.generate(seg)
        plt.figure(figsize=(8,8))
        plt.imshow(s_m)
        show_anns(masks)
        plt.axis('off')
        filepath = os.path.join('static', "masks.jpg")
        plt.savefig(filepath)
        return redirect(url_for('segment', filename='masks.jpg'))
    return redirect(request.url)


@app.route('/segment/<filename>', methods=['GET', 'POST'])
def segment(filename):
    fn = filename
    return render_template('segment.html', filename=fn)

@app.route('/submit-your-info', methods=['GET', 'POST'])
def results():
    mask_index = request.form['num']
    changes = request.form['name'].split(',')
    length = len(changes)
    source_image = Image.open('static/me.jpg')
    seg = asarray(source_image)
    masks = mask_generator.generate(seg)
    segmentation_mask=masks[int(mask_index)]['segmentation']
    stable_diffusion_mask=PIL.Image.fromarray(segmentation_mask)
    generator = torch.Generator(device="cuda").manual_seed(77)
    encoded_images = []
    for ind in range(length):
        image = pipe(prompt=changes[ind], guidance_scale=7.5, num_inference_steps=60, generator=generator, image=source_image, mask_image=stable_diffusion_mask).images[0]
        encoded_images.append(image)
    r = 2
    c = 2
    if length == 1:
        r = 1
    elif length == 2:
        r = 2
    elif length <= 5:
        c = 3
    create_image_grid(source_image, encoded_images, changes, r, c)
    return render_template('rems.html', filename='static/res.jpg')


if __name__ == '__main__':
    app.run(debug=False)
