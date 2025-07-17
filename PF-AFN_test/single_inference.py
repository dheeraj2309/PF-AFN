import torch
import torch.nn as nn
import os
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
import argparse
import torch.nn.functional as F

# --- Import your project's model classes and functions ---
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

def preprocess_and_one_hot_agnostic(agnostic_path, height, width, num_classes):
    """
    Loads a 'P' mode agnostic map, extracts labels, resizes, and converts
    it to the required one-hot tensor format IN-MEMORY.
    """
    with Image.open(agnostic_path) as img:
        if img.mode != 'P':
            raise TypeError(f"Agnostic map at '{agnostic_path}' must be a 'P' mode image.")
        # This extracts the raw integer labels (H, W)
        label_array = np.array(img)
        
    # Resize with NEAREST to preserve label integrity
    resized_labels = cv2.resize(label_array, (width, height), interpolation=cv2.INTER_NEAREST)
    
    # Convert to a LongTensor for one-hot encoding
    label_tensor = torch.from_numpy(resized_labels).long().unsqueeze(0) # Shape: [1, H, W]
    
    # >>>>> THIS IS THE KEY CONVERSION STEP <<<<<
    # Create the one-hot tensor. The output will have shape [1, H, W, num_classes]
    one_hot_tensor = F.one_hot(label_tensor, num_classes=num_classes)
    
    # Permute the dimensions to be [Batch, Channels, Height, Width] for the model
    one_hot_tensor = one_hot_tensor.permute(0, 3, 1, 2).float()
    
    return one_hot_tensor

# --- Argument Parser ---
def get_parser():
    parser = argparse.ArgumentParser(description="Run single-image inference for the Warping Module.")
    parser.add_argument('-c', '--cloth', type=str, required=True, help="Path to the clothing image.")
    parser.add_argument('-d', '--densepose', type=str, required=True, help="Path to the person's DensePose map.")
    parser.add_argument('-a', '--agnostic', type=str, required=True, help="Path to the 'P' mode cloth-agnostic parse map.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the pre-trained warping model checkpoint.")
    parser.add_argument('-o', '--output', type=str, required=True, help="Path to save the final warped cloth image.")
    parser.add_argument('--gpu_id', type=int, default=0, help="GPU ID to use for inference.")
    return parser

# --- Main Inference Logic ---
def main(opt):
    print("--- Initializing ---")
    device = torch.device(f'cuda:{opt.gpu_id}')
    
    MODEL_HEIGHT = 256
    MODEL_WIDTH = 192
    
    # This must match the number of segmentation classes the model was trained with.
    NUM_SEMANTIC_CLASSES = 20
    
    class MockOptions:
        def __init__(self):
            self.label_nc = NUM_SEMANTIC_CLASSES
            self.grid_size = 5
            self.semantic_nc = NUM_SEMANTIC_CLASSES
    model_opt = MockOptions()

    print(f"Loading model from checkpoint: {opt.checkpoint}")
    
    # Initialize the model to expect (3 + NUM_SEMANTIC_CLASSES) input channels
    warp_model = AFWM(model_opt, 3 + NUM_SEMANTIC_CLASSES)
    warp_model.eval()
    warp_model.to(device)
    load_checkpoint(warp_model, opt.checkpoint)
    
    print("Processing input images...")
    
    # Load and process all files
    agnostic_one_hot_tensor = preprocess_and_one_hot_agnostic(opt.agnostic, MODEL_HEIGHT, MODEL_WIDTH, NUM_SEMANTIC_CLASSES).to(device)
    densepose_tensor = preprocess_rgb_image(opt.densepose, MODEL_HEIGHT, MODEL_WIDTH).to(device)
    clothing_tensor = preprocess_rgb_image(opt.cloth, MODEL_HEIGHT, MODEL_WIDTH).to(device)

    # Concatenate to create the final 16-channel input
    person_representation = torch.cat([agnostic_one_hot_tensor, densepose_tensor], 1)
    
    print(f"Constructed person representation with shape: {person_representation.shape}") # Should be [1, 16, 256, 192]

    print("Running inference...")
    with torch.no_grad():
        flow_out = warp_model(person_representation, clothing_tensor)
        warped_cloth, _ = flow_out
        
    print("Saving result...")
    output_dir = os.path.dirname(opt.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    warped_cloth_np = (warped_cloth.squeeze().permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0
    warped_cloth_rgb = (warped_cloth_np * 255).astype(np.uint8)
    warped_cloth_bgr = cv2.cvtColor(warped_cloth_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(opt.output, warped_cloth_bgr)
    
    print(f"\n--- Success! Warped cloth saved to: {opt.output}")

if __name__ == '__main__':
    parser = get_parser()
    opt = parser.parse_args()
    main(opt)