import os
import nibabel as nib


def check_data_consistency(image_dir, label_dir):
    for image_file in os.listdir(image_dir):
        if image_file.endswith('.nii.gz'):
            label_file = os.path.join(label_dir, image_file)
            image_path = os.path.join(image_dir, image_file)

            # Load images and labels
            img = nib.load(image_path)
            lbl = nib.load(label_file)

            # Check dimensions
            if img.shape != lbl.shape:
                print(f"Shape mismatch for {image_file}: Image shape {img.shape}, Label shape {lbl.shape}")
            else:
                print(f"{image_file} is consistent.")


# 使用示例
image_dir = r"D:\rs\data\nnUNet_raw\Dataset002_Heart\imagesTr"
label_dir = r"D:\rs\data\nnUNet_raw\Dataset002_Heart\labelsTr"
check_data_consistency(image_dir, label_dir)