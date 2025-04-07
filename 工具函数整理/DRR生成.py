import os
import tempfile
import zipfile


import dicom2nifti
import nibabel
import torch
import torchvision
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.filewriter import write_file


def dcm_to_nifti(input_path, output_path, tmp_dir=None, verbose=False):
    if zipfile.is_zipfile(input_path):
        if verbose: print(f"Extracting zip file: {input_path}")
        extract_dir = os.path.splitext(input_path)[0] if tmp_dir is None else tmp_dir / "extracted_dcm"
        with zipfile.ZipFile(input_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            input_path = extract_dir
    dicom2nifti.dicom_series_to_nifti(input_path, output_path, reorient_nifti=True)
    return output_path


def save_as_dicom(image, output_path, patient_id="Unknown", study_date="20230101"):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.7"  # Secondary Capture Image Storage
    meta.MediaStorageSOPInstanceUID = "1.2.3"
    meta.TransferSyntaxUID = "1.2.840.10008.1.2.1"  # Explicit VR Little Endian

    # 创建 DICOM 数据集
    ds = FileDataset(output_path, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.PatientID = patient_id
    ds.StudyDate = study_date
    ds.Rows, ds.Columns = image.shape
    ds.PixelData = image.tobytes()
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1  # Signed integer
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.SamplesPerPixel = 1

    # 保存 DICOM 文件
    write_file(output_path, ds, write_like_original=False)
    print(f"Saved DRR as DICOM: {output_path}")


if __name__ == '__main__':
    os.environ["CUDA_PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
    import deepdrr
    from deepdrr import geo
    from deepdrr.projector import Projector
    from deepdrr.utils import image_utils
    file_in = r"D:\rs\Head\WXL01"
    converted_nii = "ct.nii.gz"
    output_dir = os.path.dirname(file_in)

    with tempfile.TemporaryDirectory(prefix="nnunet_tmp_") as tmp_folder:
        nii_path = dcm_to_nifti(file_in, os.path.join(tmp_folder, converted_nii), verbose=True)
        patient = deepdrr.Volume.from_nifti(nii_path, use_thresholding=True)
    patient.faceup()
    carm = deepdrr.MobileCArm(patient.center_in_world + geo.v(0, 0, -300))

    # 检查PyCUDA是否支持CUDA
    import pycuda.driver as cuda
    import pycuda.autoinit
    os.environ["PATH"] += os.pathsep + os.path.join(os.environ["CUDA_PATH"], "bin")
    import torch
    print(torch.cuda.is_available())
    with Projector(patient, carm=None, device="cpu") as projector:
        carm.move_to(alpha=0, beta=0)
        image = projector()
    
    # path = output_dir / "result.png"
    # image_utils.save(path, image)
    # print(f"saved example projection image to {path.absolute()}")