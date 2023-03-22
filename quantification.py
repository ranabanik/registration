import nibabel as nib
import os
from glob import glob
from Utilities import ImzmlAll
from utils import bbox2, displayImage, multi_dil, nnPixelCorrect, displayMR, closest
from scipy import ndimage
from skimage.transform import resize
from skimage.measure import regionprops, label
from skimage.morphology import area_closing
from skimage.exposure import rescale_intensity, adjust_gamma, equalize_hist, equalize_adapthist
import numpy as np
import h5py
import SimpleITK as sitk
import matplotlib.pyplot as plt

proteinDir = r'/media/banikr/banikr/SpinalCordInjury/MALDI/210603-Chen_protein_slide_F'
dataFold = r'/media/banikr/banikr/SpinalCordInjury/Chase_high_res'
niiPath = os.path.join(dataFold, '9.nii')
scMaskPath = os.path.join(dataFold, '9_SC_mask_wo_CSF.nii.gz')

nSliceMR = 4
nSliceMALDI = 2
pickMzList = [14115, 14197, 8566, 4961, 7063]
pickMz = pickMzList[1]
mrBlock = nib.load(niiPath).get_fdata()
mrSlice = mrBlock[..., nSliceMR]
scMask = nib.load(scMaskPath).get_fdata()
scMaskSlice = scMask[..., nSliceMR]
rmin, rmax, cmin, cmax = bbox2(scMaskSlice)
mrSlice_x3 = resize(mrSlice,
                    tuple(3 * x for x in mrSlice.shape),
                    # mode='edge',
                    anti_aliasing=False,  # to preserve label values
                    preserve_range=True,
                    order=0
                    )
# mrSlice_x3_normed = mrSlice_x3 / np.max(mrSlice_x3)
# ionImageMRsize = np.zeros(tuple(3 * x for x in mrSlice.shape))
# ionImageOnMRPath = os.path.join(dataFold, '{}_mr{}_ms{}_ionImage_mz_{}_onMR.npy'.format(
#                    os.path.basename(os.path.normpath(niiPath).split('.')[0]),
#                    nSliceMR, nSliceMALDI, pickMz))
# ionImageOnMR = np.load(ionImageOnMRPath)
# displayImage(ionImageOnMR, 'ion image on MR')

readSegPath = os.path.join(proteinDir, 'segmentation_vae_vae.h5')
with h5py.File(readSegPath, 'r') as pfile:  # saves the data
    segImg = np.array(pfile['seg'])

maskImg = np.zeros_like(segImg)
maskImg[np.where(segImg != 0)] = 1

labeled_array, nFeatures = ndimage.label(maskImg, structure=np.ones((3, 3)))
print(nFeatures, "sections found...")

sliceIdcsMALDI = [1, 2, 3, 4]       #  [1, 2, 3, 4]
sliceIdcsMRI = [0, 1, 2, 3, 4]      #  MRI
# for secID in nSliceMALDI:  # ,3,4]:
nSliceMALDI = 2     # sliceIdcsMALDI[2]
nSliceMRI = 4   # sliceIdcsMRI[2]
print("MRI slice: {}\nMSI slice: {}".format(nSliceMRI, nSliceMALDI))
minx, miny = np.inf, np.inf
maxx, maxy = -np.inf, -np.inf
for x in range(labeled_array.shape[0]):
    for y in range(labeled_array.shape[1]):
        if labeled_array[x, y] == nSliceMALDI:
            minx, miny = min(minx, x), min(miny, y)
            maxx, maxy = max(maxx, x), max(maxy, y)
regionshape = [maxx - minx + 1,
               maxy - miny + 1]
secImg = segImg[minx:maxx, miny:maxy]
displayImage(secImg, Title_='VAE segmentation {}'.format(nSliceMALDI))

if len(np.unique(secImg)) > 4:
    secImg_ = np.zeros_like(secImg)
    secImg_[np.where(secImg == 1)] = 2
    secImg_[np.where(secImg == 2)] = 2
    secImg_[np.where(secImg == 3)] = 1
    secImg_[np.where(secImg == 4)] = 1
    secImg_[np.where(secImg == 5)] = 0
    secImg = secImg_
displayImage(secImg, Title_='3 label segmentation')

secImgMorphed = np.zeros_like(secImg)
for tissue_label in [1, 2]:
    blobs_labels = label(secImg == tissue_label, background=0, connectivity=1)
    regionProperties = regionprops(label_image=blobs_labels)
    if tissue_label == 2:
        regionsBiggerToSmallerList = np.argsort([prop.area_filled for prop in regionProperties])[::-1][0:2]
    if tissue_label == 1:
        regionsBiggerToSmallerList = np.argsort([prop.area_filled for prop in regionProperties])[::-1][0:15]

    morphedTissueImg = np.zeros_like(blobs_labels)
    for region in regionsBiggerToSmallerList:
        for coord in regionProperties[region].coords:
            # print(coord)
            morphedTissueImg[coord[0], coord[1]] = 1
    square = np.array([[1, 1],
                       [1, 1]])
    morphedTissueImg = multi_dil(morphedTissueImg, 1, element=square)
    morphedTissueImg = area_closing(morphedTissueImg, area_threshold=1000, connectivity=50)
    secImgMorphed[np.where(morphedTissueImg)] = tissue_label

for tissue_label in [1, 2]:
    mask = (secImgMorphed == tissue_label)
    secImg[mask] = tissue_label
# displayImage(secImg, Title_='moving image')

secImg_ = np.zeros_like(secImg)
secImg_[np.where(secImg != 0)] = 1
blobs_labels2 = label(secImg_, background=0, connectivity=1)
regProps = regionprops(blobs_labels2)
largestcomp = np.argmax([prop.area_filled for prop in regProps])
for i, region in enumerate(regProps):
    if i == largestcomp: pass   # dont remove the largest conncomp
    else:
        for coord in region.coords:
            secImg[coord[0], coord[1]] = 0

# change the area_threshold 32, 64, 96 to get
secImg_filled = area_closing(secImg, area_threshold=32, connectivity=8)
# displayImage(secImg_filled, Title_='largest_filled')
newMat = secImg_filled - secImg
secImg[np.where(newMat == 1)] = 3

secImg = nnPixelCorrect(secImg, d=8, n_=3, plot_=False)
displayImage(secImg, 'corrected_secImg')
niiPath = os.path.join(dataFold, '9.nii')
mrBlock = nib.load(niiPath).get_fdata()
mrSlice = mrBlock[..., nSliceMRI]
scMaskPath = os.path.join(dataFold, '9_SC_mask_wo_CSF.nii.gz')
scMask = nib.load(scMaskPath).get_fdata()
scMaskSlice = scMask[..., nSliceMRI]

rmin, rmax, cmin, cmax = bbox2(scMaskSlice)
mrSlice[np.where(scMaskSlice == 0)] = 0
scSlice = mrSlice[rmin:rmax + 1, cmin:cmax + 1]

perc_ = (5, 99.5)
vmin, vmax = np.percentile(scSlice, q=perc_)
scClipped = rescale_intensity(
    scSlice,
    in_range=(vmin, vmax),
    out_range=(0, 255)
)
classes_ = 3
gamma_val = 1.9
scSliceGamma = adjust_gamma(scClipped, gamma=gamma_val)
scSliceGammaEq = equalize_hist(scSliceGamma)
scSliceGammaEqCLAHE = equalize_adapthist(scSliceGammaEq, kernel_size=8)

scSliceGammaEqCLAHEresized = resize(scSliceGammaEqCLAHE,
                        tuple(3 * x for x in scSliceGammaEqCLAHE.shape),
                        # mode='edge',
                        anti_aliasing=False,  # to preserve label values
                        preserve_range=True,
                        order=0)
scSliceGammaEqCLAHEresized = scSliceGammaEqCLAHEresized/np.max(scSliceGammaEqCLAHEresized)
displayMR(scSliceGammaEqCLAHEresized, Title_='resized MR')
# thresholds = filters.threshold_multiotsu(scSliceGammaEqCLAHEresized, classes=classes_)
# regions = np.digitize(scSliceGammaEqCLAHEresized, bins=thresholds)
# displayImage(regions, 'Multi-otsu segmentation')

imzPath = glob(os.path.join(proteinDir, '*.imzML'))[0]
proteinImzObj = ImzmlAll(imzPath)
peakmzs, I = proteinImzObj.parser.getspectrum(0)
clVal, clValInd = closest(peakmzs, pickMz)
ionImage = proteinImzObj.getionimage(regID=nSliceMALDI, tol=1, mz_value=clVal)
displayImage(ionImage, 'ion image')

finalCompTxFile = os.path.join(dataFold, '{}_mr{}_ms{}_finalCompTx.hdf'.format(os.path.basename(os.path.normpath(niiPath).split('.')[0]),
                                                            nSliceMR, nSliceMALDI))
print(finalCompTxFile)
finalCompTx = sitk.ReadTransform(finalCompTxFile)#.Downcast()
# print("outTx >> \n", outTx)
fixed_image = sitk.Cast(sitk.GetImageFromArray(scSliceGammaEqCLAHEresized), sitk.sitkFloat32)
moving_image = sitk.Cast(sitk.GetImageFromArray(ionImage), sitk.sitkFloat32)

# print("fixed moving size: ", fixed_image.GetSize(), moving_image.GetSize())
transformedIonImage = sitk.Resample(moving_image, fixed_image, finalCompTx, sitk.sitkNearestNeighbor, 0)
displayImage(sitk.GetArrayFromImage(transformedIonImage), 'transformed ion image')
moving_image = sitk.Cast(sitk.GetImageFromArray(secImg), sitk.sitkFloat32)

transformedSegImage = sitk.Resample(moving_image, fixed_image, finalCompTx, sitk.sitkNearestNeighbor, 0)
displayImage(sitk.GetArrayFromImage(transformedSegImage), 'transformed seg image')

from numpy import ma
from collections import defaultdict
ionList = defaultdict(list)
mtcList = defaultdict(list)
for x in range(0, sitk.GetArrayFromImage(transformedSegImage).shape[0]):
    for y in range(0, sitk.GetArrayFromImage(transformedSegImage).shape[1]):
        if sitk.GetArrayFromImage(transformedSegImage)[x, y] == 1:
            ionList[1].append(sitk.GetArrayFromImage(transformedIonImage)[x, y])
            mtcList[1].append(sitk.GetArrayFromImage(fixed_image)[x, y])
        if sitk.GetArrayFromImage(transformedSegImage)[x, y] == 2:
            ionList[2].append(sitk.GetArrayFromImage(transformedIonImage)[x, y])
            mtcList[2].append(sitk.GetArrayFromImage(fixed_image)[x, y])
print(ionList)

fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
ax.scatter(mtcList[1], ionList[1], label='wm',
            edgecolor='none',
            alpha=0.5, s=10,
            cmap=plt.cm.get_cmap('nipy_spectral', 1))
ax.scatter(mtcList[2], ionList[2], label='gm',
            edgecolor='none',
            alpha=0.5, s=10,
            cmap=plt.cm.get_cmap('nipy_spectral', 1))
ax.legend(loc='best')
ax.set_xlabel("MTC contrast(normalized)")
ax.set_ylabel("ion abundance")
ax.set_title("m/z: {:.4f}".format(clVal))
plt.show()




