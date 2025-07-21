import os
import shutil
import argparse
from pathlib import Path

def copy_and_rename(
    src_folder, dst_folder, original_stem, new_name_stem,
    suffix_override="", keep_suffix=False
):
    src_folder = Path(src_folder)
    dst_folder = Path(dst_folder)
    dst_folder.mkdir(parents=True, exist_ok=True)

    expected_stem = f"{original_stem}{suffix_override}" if suffix_override else original_stem

    for file in src_folder.iterdir():
        if file.stem == expected_stem:
            ext = file.suffix
            new_name = f"{new_name_stem}{suffix_override}{ext}" if keep_suffix else f"{new_name_stem}{ext}"
            shutil.copy(file, dst_folder / new_name)
            return True

    print(f"⚠️ Warning: Could not find match in {src_folder} for {expected_stem}")
    return False

def prepare_vton_input(sample_root, model_img_path, cloth_img_path, dest_root):
    sample_root = Path(sample_root)
    dest_root = Path(dest_root)
    new_name = "00000_00"

    # Extract the stem (filename without extension)
    model_stem = Path(model_img_path).stem
    cloth_stem = Path(cloth_img_path).stem

    # Folders and suffixes for model and cloth data
    model_related = {
        "image": ("", False),
        "image-parse-v3": ("", False),
        "image-parse-agnostic-v3.2": ("", False),
        "image-densepose": ("", False),
        "openpose_img": ("_rendered", True),
        "openpose_json": ("_keypoints", True)
    }

    cloth_related = {
        "cloth": ("", False),
        "cloth-mask": ("", False)
    }

    # Copy model-related files
    for folder, (suffix, keep_suffix) in model_related.items():
        src = sample_root / folder
        dst = dest_root / folder
        copy_and_rename(src, dst, model_stem, new_name, suffix_override=suffix, keep_suffix=keep_suffix)

    # Copy cloth-related files
    for folder, (suffix, keep_suffix) in cloth_related.items():
        src = sample_root / folder
        dst = dest_root / folder
        copy_and_rename(src, dst, cloth_stem, new_name, suffix_override=suffix, keep_suffix=keep_suffix)

    # Write test_pairs.txt (VTON expects .jpg)
    with open(dest_root / "test_pairs.txt", "w") as f:
        f.write(f"{new_name}.jpg {new_name}.jpg\n")

    print(f"\n✅ VTON-ready input prepared at: {dest_root}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare VTON input structure from sample folders.")

    parser.add_argument("--sample_root", type=str, required=True, help="Root folder containing sample VTON input folders.")
    parser.add_argument("--model_img_path", type=str, required=True, help="Full path to selected model image.")
    parser.add_argument("--cloth_img_path", type=str, required=True, help="Full path to selected cloth image.")
    parser.add_argument("--dest_root", type=str, required=True, help="Output folder where final input will be placed.")

    args = parser.parse_args()

    prepare_vton_input(
        sample_root=args.sample_root,
        model_img_path=args.model_img_path,
        cloth_img_path=args.cloth_img_path,
        dest_root=args.dest_root
    )
