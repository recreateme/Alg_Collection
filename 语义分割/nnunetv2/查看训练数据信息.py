import nibabel as nib

file = r"D:\Work\data\RAOS-Real\CancerImages(Set1)\imagesTr\1.2.840.113619.2.416.1620186646681570467055246874744467879.nii.gz"
img = nib.load(file)
print(img.header)