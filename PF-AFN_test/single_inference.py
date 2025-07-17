import torch
import torch.nn as nn
import os
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image

# --- Import your project's model classes and functions ---
from options.test_options import TestOptions
from models.afwm import AFWM
from models.networks import load_checkpoint

# --- Helper Functions for Preprocessing ---

def preprocess_rgb_image(image_path, height, width):
    """Loads, resizes, and normalizes a standard 3-channel RGB image."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LANCZOS4)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    return transform(img).unsqueeze(0)

def preprocess_p_mode_agnostic(agnostic_path, height, width):
    """
    Correctly loads a 'P' mode agnostic map, extracts its integer labels,
    resizes it, and converts it to the required model tensor format.
    """
    with Image.open(agnostic_path) as img:
        if img.mode != 'P':
            raise TypeError(f"Agnostic map at '{agnostic_path}' must be a 'P' mode image.")
            
        # This is the key: np.array() on a 'P' mode image gives the raw label data.
        label_array = np.array(img)
        
    # Resize using NEAREST interpolation to preserve exact label values.
    resized_labels = cv2.resize(label_array, (width, height), interpolation=cv2.INTER_NEAREST)
    
    # Convert to a FloatTensor and add batch and channel dimensions for the model.
    agnostic_tensor = torch.from_numpy(resized_labels).unsqueeze(0).unsqueeze(0).float()
    return agnostic_tensor

# --- Main Inference Logic ---

# 1. Setup Models and Options
print("--- Initializing ---")
opt = TestOptions().parse()
device = torch.device(f'cuda:{opt.gpu_ids[0]}')

IMG_HEIGHT = 256
IMG_WIDTH = 192

warp_model = AFWM(opt, 4)
warp_model.eval()
warp_model.to(device)
load_checkpoint(warp_model, opt.warp_checkpoint)
print(f"--- Model loaded from {opt.warp_checkpoint} ---")

# 2. Define Input File Paths
# >>>>>>>>>>>> IMPORTANT: CHANGE THESE PATHS TO YOUR FILES <<<<<<<<<<<<
cloth_image_path    = 'data/test/cloth/000001_1.jpg'
densepose_image_path  = 'data/test/image-densepose/000001_0.jpg'
# This path should point to the 'P' mode agnostic map you created
agnostic_map_path   = 'single_inference_data/test/image-parse-agnostic-v3.2/person1.png'
person_image_path   = 'data/test/image/000001_0.jpg'

# 3. Load and Preprocess All Inputs
print(f"--- Processing Input Images ---")
clothing_image  = preprocess_rgb_image(cloth_image_path, IMG_HEIGHT, IMG_WIDTH).to(device)
densepose_image = preprocess_rgb_image(densepose_image_path, IMG_HEIGHT, IMG_WIDTH).to(device)
agnostic_map    = preprocess_p_mode_agnostic(agnostic_map_path, IMG_HEIGHT, IMG_WIDTH).to(device)

# 4. Create Final Model Inputs
person_representation = torch.cat([agnostic_map, densepose_image], 1)

print("--- Running Inference on a Single Image ---")
# 5. Run the Model Inference
with torch.no_grad():
    flow_out = warp_model(person_representation, clothing_image)
    warped_cloth, last_flow = flow_out

# 6. Save the Result
print("--- Saving Result ---")
output_dir = os.path.join('results', opt.name, 'warping_module_output')
os.makedirs(output_dir, exist_ok=True)

warped_cloth_np = (warped_cloth.squeeze().permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0
warped_cloth_rgb = (warped_cloth_np * 255).astype(np.uint8)
warped_cloth_bgr = cv2.cvtColor(warped_cloth_rgb, cv2.COLOR_RGB2BGR)

output_filename = f"final_warped_result.png"
cv2.imwrite(os.path.join(output_dir, output_filename), warped_cloth_bgr)
print(f"--- Success! Warped cloth saved to: {os.path.join(output_dir, output_filename)} ---")