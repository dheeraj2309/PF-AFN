import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import os

# --- Prerequisite: Ensure the 'models' directory is in your Python path ---
# This script assumes it can import the AFWM model definition.
from models.afwm import AFWM
from models.networks import load_checkpoint

# --- Helper Functions to Replicate CPDataset Logic ---

def get_transform(normalize=True):
    """Returns a composed transform to convert image to tensor."""
    transform_list = [transforms.ToTensor()]
    if normalize:
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return transforms.Compose(transform_list)

def load_agnostic_map(path, fine_height, fine_width):
    """
    Loads the single-channel agnostic parse map and converts it into the
    multi-channel one-hot format the model expects.
    """
    # 1. Load the single-channel label map
    im_parse_agnostic = Image.open(path)
    im_parse_agnostic = im_parse_agnostic.resize((fine_width, fine_height), Image.NEAREST)
    parse_agnostic_np = np.array(im_parse_agnostic)
    
    # Convert to a tensor
    parse_agnostic = torch.from_numpy(parse_agnostic_np[None]).long()

    # 2. Define the label grouping (critical part from cp_dataset.py)
    # This groups raw LIP IDs into 13 semantic classes.
    labels = {
        0: ['background', [0, 10]], 1: ['hair', [1, 2]],
        2: ['face', [4, 13]],       3: ['upper', [5, 6, 7]],
        4: ['bottom', [9, 12]],     5: ['left_arm', [14]],
        6: ['right_arm', [15]],     7: ['left_leg', [16]],
        8: ['right_leg', [17]],     9: ['left_shoe', [18]],
        10: ['right_shoe', [19]],   11: ['socks', [8]],
        12: ['noise', [3, 11]]
    }
    semantic_nc = 13 # Number of classes after grouping

    # 3. Create the one-hot encoded map
    parse_map = torch.FloatTensor(20, fine_height, fine_width).zero_()
    parse_map = parse_map.scatter_(0, parse_agnostic, 1.0)
    
    new_parse_agnostic_map = torch.FloatTensor(semantic_nc, fine_height, fine_width).zero_()
    for i in range(len(labels)):
        for label in labels[i][1]:
            new_parse_agnostic_map[i] += parse_map[label]
            
    return new_parse_agnostic_map

def run_warp_inference(
    cloth_path, 
    cloth_mask_path, 
    densepose_path, 
    agnostic_parse_path, 
    warp_checkpoint_path, 
    output_path
):
    """
    Performs warping inference on a single set of input files.

    Args:
        cloth_path (str): Path to the in-shop cloth image (RGB).
        cloth_mask_path (str): Path to the cloth mask (binary).
        densepose_path (str): Path to the person's DensePose map (RGB).
        agnostic_parse_path (str): Path to the single-channel agnostic parse map.
        warp_checkpoint_path (str): Path to the pre-trained .pth file for the warp model.
        output_path (str): Path to save the final warped cloth image.
    """
    # --- 1. Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fine_height = 256
    fine_width = 192
    label_nc = 13 # Number of semantic classes after grouping

    # --- 2. Load and Initialize Model ---
    # Create a dummy options object to satisfy the AFWM model constructor
    class Opt:
        def __init__(self):
            self.label_nc = label_nc
    
    opt = Opt()
    warp_model = AFWM(opt, 3 + label_nc) # 3 for densepose, 13 for agnostic map
    load_checkpoint(warp_model, warp_checkpoint_path)
    warp_model.to(device)
    warp_model.eval()
    print("Warp model loaded successfully.")

    # --- 3. Load and Preprocess Inputs ---
    transform = get_transform()
    
    # Cloth
    c_pil = Image.open(cloth_path).convert('RGB')
    c = transform(c_pil).unsqueeze(0).to(device)

    # Cloth Mask
    cm_pil = Image.open(cloth_mask_path)
    cm_array = np.array(cm_pil)
    cm_array = (cm_array >= 128).astype(np.float32)
    cm = torch.from_numpy(cm_array).unsqueeze(0).unsqueeze(0).to(device)

    # DensePose
    densepose_pil = Image.open(densepose_path).convert('RGB')
    densepose = transform(densepose_pil).unsqueeze(0).to(device)

    # Agnostic Parse Map (special handling)
    agnostic_map = load_agnostic_map(agnostic_parse_path, fine_height, fine_width)
    agnostic_map = agnostic_map.unsqueeze(0).to(device)

    # --- 4. Prepare Model Inputs (Resize and Concatenate) ---
    # Resize all inputs to the model's expected size
    c_down = F.interpolate(c, size=(fine_height, fine_width), mode='bilinear', align_corners=False)
    cm_down = F.interpolate(cm, size=(fine_height, fine_width), mode='nearest')
    densepose_down = F.interpolate(densepose, size=(fine_height, fine_width), mode='bilinear', align_corners=False)

    # Create the person representation by concatenating the agnostic map and densepose
    person_representation = torch.cat([agnostic_map, densepose_down], 1)

    # --- 5. Run Inference ---
    with torch.no_grad():
        # The model expects (person_representation, cloth_image)
        flow_out = warp_model(person_representation, c_down)
        warped_cloth_down, last_flow = flow_out

    print("Inference completed. Warping final image...")

    # --- 6. Upscale and Final Warp for High-Resolution Output ---
    # To get a high-quality result, we upscale the flow field and apply it to the original full-res cloth
    original_height, original_width = c_pil.height, c_pil.width
    last_flow_upscaled = F.interpolate(last_flow, size=(original_height, original_width), mode='bilinear', align_corners=False)
    
    # The grid_sample function needs the flow field in the range [-1, 1].
    # The last_flow from the model is already in this format.
    warped_cloth_highres = F.grid_sample(
        c, # Use original full-resolution cloth tensor
        last_flow_upscaled.permute(0, 2, 3, 1),
        mode='bilinear',
        padding_mode='border',
        align_corners=False
    )

    # --- 7. Save the Output ---
    output_tensor = warped_cloth_highres.squeeze(0).cpu()
    # Convert tensor from [-1, 1] to [0, 1] then to [0, 255]
    output_numpy = (output_tensor.permute(1, 2, 0).numpy() + 1) / 2.0 * 255.0
    output_numpy = output_numpy.astype(np.uint8)
    
    # Convert RGB to BGR for OpenCV
    output_bgr = cv2.cvtColor(output_numpy, cv2.COLOR_RGB2BGR)

    # Create directory if it doesn't exist and save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, output_bgr)
    print(f"Successfully saved warped cloth to: {output_path}")


if __name__ == '__main__':
    # =========================================================================
    # CONFIGURE YOUR INPUTS HERE
    # =========================================================================
    
    # Path to your prepared input files
    CLOTH_PATH           = 'single_inference/cloth/00000_00.jpg'
    CLOTH_MASK_PATH      = 'single_inference/cloth-mask/00000_00.jpg'
    DENSEPOSE_PATH       = 'single_inference/image-densepose/00000_00_IUV.png'
    AGNOSTIC_PARSE_PATH  = 'single_inference/image-parse-agnostic-v3.2/00000_00.png'

    # Path to your pre-trained model checkpoint
    WARP_CHECKPOINT_PATH = 'checkpoints/warp_viton.pth' #<-- CHANGE THIS

    # Where to save the final output image
    OUTPUT_PATH          = 'single_inference/warped_cloth.jpg'

    # =========================================================================
    
    # Run the inference function
    run_warp_inference(
        cloth_path=CLOTH_PATH,
        cloth_mask_path=CLOTH_MASK_PATH,
        densepose_path=DENSEPOSE_PATH,
        agnostic_parse_path=AGNOSTIC_PARSE_PATH,
        warp_checkpoint_path=WARP_CHECKPOINT_PATH,
        output_path=OUTPUT_PATH
    )