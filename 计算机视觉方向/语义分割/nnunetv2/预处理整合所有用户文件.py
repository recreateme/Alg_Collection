import pydicom
import SimpleITK as sitk
import numpy as np
import os
import json
from skimage.draw import polygon

# 根文件夹路径
root_dir = r"D:\DevData\SDC003_200"
output_dir = os.path.join("D:\DevData", "output")
os.makedirs(os.path.join(output_dir, "imagesTr"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "labelsTr"), exist_ok=True)

all_roi_names = []

# 遍历根文件夹中的每个用户文件夹
for patient_idx, patient_dir in enumerate(os.listdir(root_dir)):
    patient_path = os.path.join(root_dir, patient_dir)

    # 查找 DICOM 文件夹和RTstructure文件
    dicom_files = []
    meta_file = None
    # 遍历每一个目录
    for sub_dir in os.listdir(patient_path):
        sub_dir_path = os.path.join(patient_path, sub_dir)
        # 如果sub_dir_path后缀是.dcm，则添加到dicom_dir中
        if os.path.isfile(sub_dir_path) and sub_dir_path.endswith('.dcm'):
            dicom_files.append(sub_dir_path)
        # 否则如果是文件夹，则找到以rs开头且后缀是.dcm的文件夹，添加到meta_file中
        else:
            # 找到sub_dir_path下以rs开头且后缀是.dcm的文件
            plnas_files = os.listdir(sub_dir_path)
            # 找到planns_files中以rs开头且后缀是.dcm的文件
            for plnas_file in plnas_files:
                if plnas_file.lower().startswith('rs') and plnas_file.endswith('.dcm'):
                    meta_file = os.path.join(sub_dir_path, plnas_file)
                    break
            if meta_file is not None:
                print("缺少RTStructure文件，跳过该用户")
                break
    if len(dicom_files) == 0 or meta_file is None:
        print(f"跳过 {patient_dir}，未找到 DICOM 或元文件夹。")
        continue

    # 读取 DICOM 文件并转换为 NIfTI
    dicom_files.sort()
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(dicom_files)
    image = reader.Execute()
    patient_id = f"patient{str(patient_idx + 1).zfill(3)}"
    sitk.WriteImage(image, os.path.join(output_dir, "imagesTr", f"{patient_id}_0000.nii.gz"))

    # 读取 RTStructure 文件
    rtss = pydicom.dcmread()
    roi_sequence = rtss.StructureSetROISequence
    roi_names = {roi.ROINumber: roi.ROIName for roi in roi_sequence}
    all_roi_names.append(set(roi_names.values()))

    # 生成单一掩码文件，包含所有 ROI
    def rtstruct_to_combined_mask(rtss, image, roi_names):
        # 初始化掩码，与图像大小一致
        image_array = sitk.GetArrayFromImage(image)  # z, y, x
        combined_mask = np.zeros(image_array.shape, dtype=np.uint8)

        # 为每个 ROI 分配整数标签
        for roi_number, roi_name in roi_names.items():
            label_value = list(roi_names.values()).index(roi_name) + 1  # 背景为 0，ROI 从 1 开始

            # 找到对应的轮廓序列
            contour_sequence = [cs for cs in rtss.ROIContourSequence if cs.ReferencedROINumber == roi_number]
            if not contour_sequence:
                continue

            # 遍历每个轮廓
            for contour in contour_sequence[0].ContourSequence:
                points = np.array(contour.ContourData).reshape(-1, 3)
                # 转换为图像坐标
                indices = [image.TransformPhysicalPointToIndex(p) for p in points]
                x_coords = [idx[0] for idx in indices]  # x 坐标
                y_coords = [idx[1] for idx in indices]  # y 坐标
                z_slice = indices[0][2]  # z 坐标（假设同一轮廓在同一切片）

                # 检查边界
                if not all(0 <= i < s for i, s in zip([min(x_coords), min(y_coords), z_slice], combined_mask.shape[::-1])):
                    continue

                # 填充多边形区域
                rr, cc = polygon(y_coords, x_coords, shape=combined_mask[z_slice].shape)
                combined_mask[z_slice, rr, cc] = label_value

        return combined_mask

    # 生成并保存单一掩码文件
    combined_mask = rtstruct_to_combined_mask(rtss, image, roi_names)
    mask_img = sitk.GetImageFromArray(combined_mask)
    mask_img.CopyInformation(image)  # 保持空间信息一致
    sitk.WriteImage(mask_img, os.path.join(output_dir, "labelsTr", f"{patient_id}.nii.gz"))

# 检查所有用户的分割标签是否一致
if len(set(frozenset(names) for names in all_roi_names)) > 1:
    print("警告：不同用户的分割标签不一致！")
else:
    print("所有用户的分割标签一致。")

# 生成统一的标签映射
unique_roi_names = sorted(all_roi_names[0]) if all_roi_names else []
label_mapping = {"background": 0}
for i, roi_name in enumerate(unique_roi_names, 1):
    label_mapping[roi_name] = i

# 生成 dataset.json
dataset_json = {
    "channel_names": {"0": "CT"},  # 根据模态调整
    "labels": label_mapping,
    "file_ending": ".nii.gz",
    "overwrite_image_reader_writer": "NibabelIO"
}
with open(os.path.join(output_dir, "dataset.json"), "w") as f:
    json.dump(dataset_json, f, indent=4)

print("数据集生成完成！")