#todo   +-----------------------------------------------+
#       |  1. Overlay of ion image and MRI is due       |
#       |    1.1. Done with illustrator opacity now...  |
#       +-----------------------------------------------+

import os
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
from utils import displayImage, bbox2, displayMR
import SimpleITK as sitk
import nibabel as nib

proteinDir = r'/media/banikr/banikr/SpinalCordInjury/MALDI/210603-Chen_protein_slide_F'
dataFold = r'/media/banikr/banikr/SpinalCordInjury/Chase_high_res'
niiPath = os.path.join(dataFold, '9.nii')
scMaskPath = os.path.join(dataFold, '9_SC_mask_wo_CSF.nii.gz')

def closest(lst, K):
    closestVal = lst[min(range(len(lst)), key=lambda i: abs(lst[i] - K))]
    closestValInd = np.squeeze(np.where(peakmzs==closestVal))
    return closestVal, closestValInd

with h5py.File(os.path.join(proteinDir, 'sum_argrel.h5'), 'r') as pfile:  # saves the data
    spectra = np.array(pfile['spectra'])
    peakmzs = np.array(pfile['peakmzs'])
    xloc = np.array(pfile['xloc'])
    yloc = np.array(pfile['yloc'])
    print(pfile.keys())

print(spectra.shape)
print(peakmzs)

plateSecMask = np.zeros((np.max(xloc) + 1, np.max(yloc) + 1))
print("plate: ", plateSecMask.shape)
plateIonImage = np.zeros((np.max(xloc) + 1, np.max(yloc) + 1))
# -> ubiquitin = 8566, mbp1Mz = 14115
# -> mbp2Mz = 14197, tbeta4Mz = 4961
ubiquitinMz = 4961
mbp1Mz = 14115
mbp2Mz = 14197  # is this actually mbp mz?
tbeta4Mz = 4961
clVal, clValInd = closest(peakmzs, ubiquitinMz)

channel_spec = spectra[:, clValInd]
print(channel_spec)

for i, (x, y) in enumerate(zip(xloc, yloc)):
    plateSecMask[x, y] = 1
    plateIonImage[x, y] = channel_spec[i]

# displayImage(plateSecMask, 'plate mask')
# displayImage(plateIonImage, 'plate ion image')
labeled_array, nFeatures = ndimage.label(plateSecMask, structure=np.ones((3, 3)))
print(nFeatures, "sections found...")
# displayImage(labeled_array, 'labeled sections')
nSliceMALDI = [2]       #  [1, 2, 3, 4]
nSlice = 4      #  MRI
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
displayImage(ionImage, Title_='section:{} ion image mz:{} '.format(secID, clVal))

outTxFile = os.path.join(dataFold, '{}_mr{}_ms{}_finalCompTx.hdf'.format(os.path.basename(os.path.normpath(niiPath).split('.')[0]),
                                                            nSlice, nSliceMALDI[0]))

mrBlock = nib.load(niiPath).get_fdata()
mrSlice = mrBlock[..., nSlice]
scMask = nib.load(scMaskPath).get_fdata()
scMaskSlice = scMask[..., nSlice]
rmin, rmax, cmin, cmax = bbox2(scMaskSlice)
mrSlice[np.where(scMaskSlice == 0)] = 0
scSlice = mrSlice[rmin:rmax + 1, cmin:cmax + 1]
#
perc_ = (5, 99.5)
vmin, vmax = np.percentile(scSlice, q=perc_)
scClipped = rescale_intensity(
    scSlice,
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
#
displayMR(scSlice, Title_='raw MR')
# # print()
scSliceGammaEqCLAHEresized = resize(scSliceGammaEqCLAHE,
                        tuple(3 * x for x in scSliceGammaEqCLAHE.shape),
                        # mode='edge',
                        anti_aliasing=False,  # to preserve label values
                        preserve_range=True,
                        order=0)
scSliceGammaEqCLAHEresized = scSliceGammaEqCLAHEresized / np.max(scSliceGammaEqCLAHEresized)
# displayMR(scSliceGammaEqCLAHEresized, Title_='CLAHE resized(normed 0 - 1)')

finalCompTxFile = os.path.join(dataFold, '{}_mr{}_ms{}_finalCompTx.hdf'.format(os.path.basename(os.path.normpath(niiPath).split('.')[0]),
                                                            nSlice, nSliceMALDI[0]))
print(finalCompTxFile)
finalCompTx = sitk.ReadTransform(finalCompTxFile)#.Downcast()
# print("outTx >> \n", outTx)
fixed_image = sitk.Cast(sitk.GetImageFromArray(scSliceGammaEqCLAHEresized), sitk.sitkFloat32)
moving_image = sitk.Cast(sitk.GetImageFromArray(ionImage), sitk.sitkFloat32)

print("fixed moving size: ", fixed_image.GetSize(), moving_image.GetSize())
out = sitk.Resample(moving_image, fixed_image, finalCompTx, sitk.sitkNearestNeighbor, 0)
displayImage(sitk.GetArrayFromImage(out), 'transformed ion image')

if __name__ != '__main__':
    # save for overlay in adobe
    mrBlock = nib.load(niiPath).get_fdata()
    mrSlice = mrBlock[..., nSlice]
    scSlice = mrSlice[rmin:rmax + 1, cmin:cmax + 1]
    plt.imshow(scSlice, cmap='gray')
    plt.show()
    scSliceresized = resize(scSlice,
                            tuple(3 * x for x in scSlice.shape),
                            # mode='edge',
                            anti_aliasing=False,  # to preserve label values
                            preserve_range=True,
                            order=0)
    filename = os.path.join(dataFold, 'ubiquitin_slice_{}.png'.format(nSliceMALDI[0]))
    plt.imshow(sitk.GetArrayFromImage(out), origin='lower', cmap='magma')
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.show()

    filename = os.path.join(dataFold, 'mri_slice_{}.png'.format(nSlice))
    plt.imshow(scSliceresized, origin='lower', cmap='gray')
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.show()

# plt.imshow(label2rgb(10 * sitk.GetArrayFromImage(out) / np.max(sitk.GetArrayFromImage(out))), origin='lower', cmap='msml_list', alpha=1.0)
# plt.colorbar()
# # plt.imshow(2550 * sitk.GetArrayFromImage(out), cmap='msml_list', alpha=1.0)
# plt.imshow(scSliceresized, cmap='gray', origin='lower', alpha=0.5)
# plt.title("mbp1 in mr")
# plt.show()
# displayMR(scSliceresized)
# # displayImage(sitk.GetArrayFromImage(out))
# # plt.imshow(sitk.GetArrayFromImage(out), cmap='msml_list')
# # plt.imshow(scSliceresized, cmap='gray', interpolation=None, alpha=0.3)
# # plt.show()
# import cv2
#
# mri = cv2.merge((scSliceresized,scSliceresized,scSliceresized))
# # img = np.zeros((scSliceresized.shape[0], scSliceresized.shape[1], 3), dtype=scSliceresized.dtype)
# # img[..., 0] = 255 * sitk.GetArrayFromImage(out)
# # img[..., 1] = scSliceresized
# # img[..., 2] = scSliceresized
# #
# plt.imshow(mri, cmap='msml_list')
# plt.title("3D mr")
# # # # img = cv2.cvtColor(scSliceresized, cv2.IMREAD_GRAYSCALE)
# # # cv2.imshow('a', scSliceresized)
# # # cv2.waitKey(0)
# # # cv2.destroyAllWindows()
# plt.show()




