import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

def show_warped_results(results_root, image_name):
    warp_path = Path(results_root) / "warp" / f"{image_name}.jpg"
    warp_mask_path = Path(results_root) / "warp_mask" / f"{image_name}.png"

    if not warp_path.exists():
        print(f"❌ Warp image not found at {warp_path}")
        return
    if not warp_mask_path.exists():
        print(f"❌ Warp mask not found at {warp_mask_path}")
        return

    warp_img = Image.open(warp_path)
    warp_mask = Image.open(warp_mask_path)

    # Show side-by-side using matplotlib
    plt.figure(figsize=(10, 5))  # Larger width for side-by-side display

    plt.subplot(1, 2, 1)
    plt.imshow(warp_img)
    plt.title("Warped Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(warp_mask, cmap='gray')  # Use grayscale colormap for mask
    plt.title("Warped Mask")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Display VTON wrapped output image and mask.")

    parser.add_argument(
        "--results_root", type=str, required=True,
        help="Path to the 'results' folder containing 'warp' and 'warp_mask'."
    )
    parser.add_argument(
        "--image_name", type=str, default="00000_00",
        help="Base name of the image to visualize (without extension)."
    )

    args = parser.parse_args()

    show_warped_results(results_root=args.results_root, image_name=args.image_name)
