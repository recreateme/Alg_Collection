import nibabel as nib
import numpy as np


def count_label_classes(nii_gz_file):
    """
    读取nii.gz格式的标签文件，并计算其中的不同标签类别数量。

    参数:
        nii_gz_file: .nii.gz格式的标签文件路径。

    返回:
        int: 标签文件中不同类别的数量。
    """
    # 加载.nii.gz文件
    img = nib.load(nii_gz_file)
    # 获取数据
    img_data = img.get_fdata()
    # 找到所有不同的标签值
    unique_labels = np.unique(img_data)

    # 如果最小值是0且代表背景，则从计数中排除
    if unique_labels[0] == 0:
        unique_labels = unique_labels[1:]

    # 返回不同标签的数量
    return len(unique_labels)


# 示例调用
# nii_gz_file_path = r'D:\Work\data\RAOS-Real\CancerImages(Set1)\labelsTr\1.2.840.113619.2.416.1620186646681570467055246874744467879.nii.gz'
nii_gz_file_path = r'D:\Work\data\RAOS-Real\CancerImages(Set1)\imagesTr\1.2.840.113619.2.416.1620186646681570467055246874744467879.nii.gz'
label_count = count_label_classes(nii_gz_file_path)
print(f"Label classes count: {label_count}")