#todo   +-----------------------------------------------+
#       |  1. Overlay of ion image and MRI is due       |
#       |    1.1. Done with illustrator opacity now...  |
#       +-----------------------------------------------+

import os
from glob import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TKAgg')
from scipy import ndimage
from skimage.color import label2rgb
from skimage.util import compare_images
from skimage.transform import resize
from skimage.exposure import rescale_intensity, adjust_gamma, equalize_hist, equalize_adapthist
from utils import displayImage, bbox2, displayMR, closest
import SimpleITK as sitk
import nibabel as nib
from Utilities import ImzmlAll

proteinDir = r'/media/banikr/banikr/SpinalCordInjury/MALDI/210603-Chen_protein_slide_F'
dataFold = r'/media/banikr/banikr/SpinalCordInjury/Chase_high_res'
niiPath = os.path.join(dataFold, '9.nii')
scMaskPath = os.path.join(dataFold, '9_SC_mask_wo_CSF.nii.gz')

with h5py.File(os.path.join(proteinDir, 'sum_argrel.h5'), 'r') as pfile:  # saves the data
    spectra = np.array(pfile['spectra'])
    peakmzs = np.array(pfile['peakmzs'])
    xloc = np.array(pfile['xloc'])
    yloc = np.array(pfile['yloc'])
    print(pfile.keys())

print(spectra.shape)
print(peakmzs)

# -> ubiquitin = 8566, mbp1Mz = 14115
# -> mbp2Mz = 14197, tbeta4Mz = 4961
ubiquitinMz = 8566
mbp1Mz = 14115
mbp2Mz = 14197  # is this actually mbp mz?
tbeta4Mz = 4961

pickMzList = [14115, 14197, 8566, 4961, 7063]
pickMz = pickMzList[4]
clVal, clValInd = closest(peakmzs, pickMz)

channel_spec = spectra[:, clValInd]
print(channel_spec)

nSliceMALDI = 3  # [1, 2, 3, 4]
nSliceMR = 2

if __name__ != '__main__':
    plateSecMask = np.zeros((np.max(xloc) + 1, np.max(yloc) + 1))
    print("plate: ", plateSecMask.shape)
    plateIonImage = np.zeros((np.max(xloc) + 1, np.max(yloc) + 1))

    for i, (x, y) in enumerate(zip(xloc, yloc)):
        plateSecMask[x, y] = 1
        plateIonImage[x, y] = channel_spec[i]

    # displayImage(plateSecMask, 'plate mask')
    # displayImage(plateIonImage, 'plate ion image')
    labeled_array, nFeatures = ndimage.label(plateSecMask, structure=np.ones((3, 3)))
    print(nFeatures, "sections found...")
    # displayImage(labeled_array, 'labeled sections')
    #  MRI
    for secID in nSliceMALDI:  # ,3,4]:
        minx, miny = np.inf, np.inf
        maxx, maxy = -np.inf, -np.inf
        for x in range(labeled_array.shape[0]):
            for y in range(labeled_array.shape[1]):
                if labeled_array[x, y] == secID:
                    minx, miny = min(minx, x), min(miny, y)
                    maxx, maxy = max(maxx, x), max(maxy, y)
        regionshape = [maxx - minx + 1,
                       maxy - miny + 1]
        ionImage = plateIonImage[minx:maxx, miny:maxy]

if __name__ == '__main__':
    imzPath = glob(os.path.join(proteinDir, '*.imzML'))[0]
    # print(imzPath)
    proteinImzObj = ImzmlAll(imzPath)
    peakmzs, I = proteinImzObj.parser.getspectrum(0)
    clVal, clValInd = closest(peakmzs, pickMz)
    ionImage = proteinImzObj.getionimage(regID=nSliceMALDI, tol=10, mz_value=clVal)
    displayImage(ionImage, 'ion image')
# outTxFile = os.path.join(dataFold, '{}_mr{}_ms{}_finalCompTx.hdf'.format(os.path.basename(os.path.normpath(niiPath).split('.')[0]),
#                                                             nSlice, nSliceMALDI))
mrBlock = nib.load(niiPath).get_fdata()
mrSlice = mrBlock[..., nSliceMR]
scMask = nib.load(scMaskPath).get_fdata()
scMaskSlice = scMask[..., nSliceMR]
rmin, rmax, cmin, cmax = bbox2(scMaskSlice)
mrSlice[np.where(scMaskSlice == 0)] = 0
scSlice = mrSlice[rmin:rmax + 1, cmin:cmax + 1]

perc_ = (5, 99.5)
vmin, vmax = np.percentile(scSlice, q=perc_)
scClipped = rescale_intensity(scSlice,
                              in_range=(vmin, vmax),
                              out_range=(0, 255)
                              )
classes_ = 4
gamma_val = 1.9
     #[0~4]
# scSliceGamma = adjust_gamma(scClipped[:, :, nSlice], gamma=gamma_val)     # updated to the below:
scSliceGamma = adjust_gamma(scClipped, gamma=gamma_val)
scSliceGammaEq = equalize_hist(scSliceGamma)
scSliceGammaEqCLAHE = equalize_adapthist(scSliceGammaEq, kernel_size=8)

scSliceGammaEqCLAHEresized = resize(scSliceGammaEqCLAHE,
                        tuple(3 * x for x in scSliceGammaEqCLAHE.shape),
                        # mode='edge',
                        anti_aliasing=False,  # to preserve label values
                        preserve_range=True,
                        order=0)
scSliceGammaEqCLAHEresized = scSliceGammaEqCLAHEresized / np.max(scSliceGammaEqCLAHEresized)
# displayMR(scSliceGammaEqCLAHEresized, Title_='CLAHE resized(normed 0 - 1)')

finalCompTxFile = os.path.join(dataFold, '{}_mr{}_ms{}_finalCompTx.hdf'.format(os.path.basename(os.path.normpath(niiPath).split('.')[0]),
                                                            nSliceMR, nSliceMALDI))
print(finalCompTxFile)
finalCompTx = sitk.ReadTransform(finalCompTxFile)#.Downcast()
# print("outTx >> \n", outTx)
fixed_image = sitk.Cast(sitk.GetImageFromArray(scSliceGammaEqCLAHEresized), sitk.sitkFloat32)
moving_image = sitk.Cast(sitk.GetImageFromArray(ionImage), sitk.sitkFloat32)

print("fixed moving size: ", fixed_image.GetSize(), moving_image.GetSize())
out = sitk.Resample(moving_image, fixed_image, finalCompTx, sitk.sitkNearestNeighbor, 0)
displayImage(sitk.GetArrayFromImage(out), 'transformed ion image')
outFile = os.path.join(dataFold, '{}_mr{}_ms{}_ionImage_mz_{}_onMR.npy'.format(os.path.basename(os.path.normpath(niiPath).split('.')[0]),
                                                            nSliceMR, nSliceMALDI, pickMz))
if __name__ == '__main__':
    np.save(outFile, sitk.GetArrayFromImage(out))






