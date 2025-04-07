from nnunetv2.dataset_conversion import convert_raw_dataset_from_old_nnunet_format
import os




if __name__ == '__main__':
    os.environ['nnUNet_raw'] = r"D:\rs\data\nnUNet_raw"
    os.environ['nnUNet_preprocessed'] = r"D:\rs\data\nnUNet_preprocessed"
    os.environ['nnUNet_results'] = r"D:\rs\data\nnUNet_results"

    ("D:\rs\Task002_Heart", "Dataset002_Heart")