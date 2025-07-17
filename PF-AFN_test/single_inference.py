import torch
import torch.nn as nn
import os
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
import argparse

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

# --- Argument Parser ---

def get_parser():
    parser = argparse.ArgumentParser(description="Run single-image inference for the Warping Module.")
    
    # Input paths
    parser.add_argument('-c', '--cloth', type=str, required=True, help="Path to the clothing image.")
    parser.add_argument('-d', '--densepose', type=str, required=True, help="Path to the person's DensePose map.")
    parser.add_argument('-a', '--agnostic', type=str, required=True, help="Path to the 'P' mode cloth-agnostic parse map.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the pre-trained warping model checkpoint.")
    
    # Output path
    parser.add_argument('-o', '--output', type=str, required=True, help="Path to save the final warped cloth image.")
    
    # Model/Execution parameters
    parser.add_argument('--gpu_id', type=int, default=0, help="GPU ID to use for inference.")
    
    return parser

# --- Main Inference Logic ---

def main(opt):
    """
    Main function to set up the model and process the single image pair.
    """
    # --- Setup ---
    print("--- Initializing ---")
    device = torch.device(f'cuda:{opt.gpu_id}')
    
    # Model's native input resolution
    MODEL_HEIGHT = 256
    MODEL_WIDTH = 192
    
    # Mock options object for model initialization, if needed.
    # This avoids needing to pass many irrelevant training options.
    class MockOptions:
        def __init__(self):
            self.label_nc = 20
            self.grid_size = 5
            self.semantic_nc = 13

    model_opt = MockOptions()

    # --- Load Model ---
    print(f"Loading model from checkpoint: {opt.checkpoint}")
    if not os.path.exists(opt.checkpoint):
        print(f"Error: Checkpoint file not found at '{opt.checkpoint}'")
        return
        
    warp_model = AFWM(model_opt, 4) # 4 channels = 1 (agnostic) + 3 (densepose)
    warp_model.eval()
    warp_model.to(device)
    load_checkpoint(warp_model, opt.checkpoint)
    
    # --- Load and Preprocess Inputs ---
    print("Processing input images...")
    try:
        clothing_tensor = preprocess_rgb_image(opt.cloth, MODEL_HEIGHT, MODEL_WIDTH).to(device)
        densepose_tensor = preprocess_rgb_image(opt.densepose, MODEL_HEIGHT, MODEL_WIDTH).to(device)
        agnostic_tensor = preprocess_p_mode_agnostic(opt.agnostic, MODEL_HEIGHT, MODEL_WIDTH).to(device)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except TypeError as e:
        print(f"Error: {e}. Make sure the agnostic map is a 'P' mode PNG.")
        return

    # --- Create Final Model Inputs ---
    # Concatenate the agnostic map and the densepose map to create the person representation.
    person_representation = torch.cat([agnostic_tensor, densepose_tensor], 1)
    
    # --- Run Inference ---
    print("Running inference...")
    with torch.no_grad():
        flow_out = warp_model(person_representation, clothing_tensor)
        warped_cloth, _ = flow_out
        
    # --- Save Result ---
    print("Saving result...")
    # Ensure the output directory exists
    output_dir = os.path.dirname(opt.output)
    if output_dir: # Handle case where output is in the current directory
        os.makedirs(output_dir, exist_ok=True)
    
    # Post-process the output tensor to a savable image
    warped_cloth_np = (warped_cloth.squeeze().permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0
    warped_cloth_rgb = (warped_cloth_np * 255).astype(np.uint8)
    warped_cloth_bgr = cv2.cvtColor(warped_cloth_rgb, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(opt.output, warped_cloth_bgr)
    
    print(f"\n--- Success! ---")
    print(f"Warped cloth saved to: {opt.output}")

if __name__ == '__main__':
    parser = get_parser()
    opt = parser.parse_args()
    main(opt)