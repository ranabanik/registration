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
from utils import displayImage, bbox2, displayMR
import SimpleITK as sitk
import nibabel as nib
from Utilities import ImzmlAll

proteinDir = r'/media/banikr/banikr/SpinalCordInjury/MALDI/210603-Chen_protein_slide_F'
dataFold = r'/media/banikr/banikr/SpinalCordInjury/Chase_high_res'
niiPath = os.path.join(dataFold, '9.nii')
scMaskPath = os.path.join(dataFold, '9_SC_mask_wo_CSF.nii.gz')

def closest(lst, K):
    closestVal = lst[min(range(len(lst)), key=lambda i: abs(lst[i] - K))]
    closestValInd = np.squeeze(np.where(lst==closestVal))
    return closestVal, closestValInd

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

nSliceMALDI = [2]  # [1, 2, 3, 4]
nSlice = 4

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
    ionImage = proteinImzObj.getionimage(regID=nSliceMALDI[0], tol=10, mz_value=clVal)
    displayImage(ionImage, 'ion image')
# outTxFile = os.path.join(dataFold, '{}_mr{}_ms{}_finalCompTx.hdf'.format(os.path.basename(os.path.normpath(niiPath).split('.')[0]),
#                                                             nSlice, nSliceMALDI[0]))

mrBlock = nib.load(niiPath).get_fdata()
mrSlice = mrBlock[..., nSlice]
scMask = nib.load(scMaskPath).get_fdata()
scMaskSlice = scMask[..., nSlice]
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
classes_ = 4
gamma_val = 1.9
     #[0~4]
# scSliceGamma = adjust_gamma(scClipped[:, :, nSlice], gamma=gamma_val)     # updated to the below:
scSliceGamma = adjust_gamma(scClipped, gamma=gamma_val)
scSliceGammaEq = equalize_hist(scSliceGamma)
scSliceGammaEqCLAHE = equalize_adapthist(scSliceGammaEq, kernel_size=8)

# displayMR(scSlice, Title_='raw MR')
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
outFile = os.path.join(dataFold, '{}_mr{}_ms{}_ionImage_mz_{}_onMR.npy'.format(os.path.basename(os.path.normpath(niiPath).split('.')[0]),
                                                            nSlice, nSliceMALDI[0], pickMz))
np.save(outFile, sitk.GetArrayFromImage(out))

import matplotlib as mtl
from matplotlib.colors import LinearSegmentedColormap
g2rgb = 128/255
if 'msml2' not in plt.colormaps():
    # colors = [(0.1, 0.1, 0.1), (0.0, 0, 1.0), (0.0, 0.5, 0.5), (0.0, 1.0, 0.0), (0.5, 0.5, 0.0), (1.0, 0, 0.0)]  # Bk -> bl -> G -> r
    colors = [
             (0.0, 0.0, 0.0, 0.0),
             (0.2, 0.0, 0.8, 1.0),
             (0.85, 0.0, 0.4, 1.0),
             (1.0, 0.0, 0.0, 1.0)
             ]
    # (1.0, 0, 0.0)]  # Bk -> R -> G -> Bl
    color_bin = 256
    mtl.colormaps.register(LinearSegmentedColormap.from_list(name='msml2', colors=colors, N=color_bin))
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
    #
    # ionImageMRsize = resize(sitk.GetArrayFromImage(out),
    #                         tuple(x/3 for x in sitk.GetArrayFromImage(out).shape),
    #                         anti_aliasing=True,
    #                         preserve_range=False,
    #                         order=2)
    # print("MMMM >>> ", ionImageMRsize.shape)

    mrSlice_x3 = resize(mrSlice,
                        tuple(3 * x for x in mrSlice.shape),
                        # mode='edge',
                        anti_aliasing=False,  # to preserve label values
                        preserve_range=True,
                        order=0
                        )
    mrSlice_x3_normed = mrSlice_x3 / np.max(mrSlice_x3)
    displayMR(mrSlice_x3_normed, 'bigger')
    # displayMR(scSliceGammaEqCLAHEresized, 'clahe')
    mrSlice_x3_normed[3 * rmin: 3 * (rmax + 1), 3 * cmin: 3 * (cmax + 1)] = scSliceGammaEqCLAHEresized
    displayMR(mrSlice_x3_normed, 'replaced')
    ionImageMRsize = np.zeros(tuple(3 * x for x in mrSlice.shape))
    print("//", ionImageMRsize.shape)
    ionImageMRsize[3 * rmin: 3 * (rmax + 1), 3 * cmin: 3 * (cmax + 1)] = sitk.GetArrayFromImage(out)
    displayImage(ionImageMRsize, 'ion image mr size')
if __name__ != '__main__':
    filename = os.path.join(dataFold, 'ubiquitin_slice_{}.png'.format(nSliceMALDI[0]))
    plt.imshow(sitk.GetArrayFromImage(out), origin='lower', cmap='magma')
    plt.gca().set_axis_off()
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.margins(0, 0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.show()

    filename = os.path.join(dataFold, 'mri_slice_{}.png'.format(nSlice))
    plt.imshow(scSliceresized, origin='lower', cmap='msml_list_gray')
    plt.gca().set_axis_off()
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.margins(0, 0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.savefig(filename, bbox_inches='tight', pad_inches=0)
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
# fig = plt.figure(dpi=300)
if __name__ != '__main__':
    fig, ax = plt.subplots(figsize=[5, 4], dpi=100)
    ax.imshow(mrSlice_x3, origin='lower', cmap='gray', alpha=1.0, zorder=0)
    ax.imshow(ionImageMRsize, origin='lower', cmap='msml2', alpha=1.0, zorder=1)
    pcm = ax.pcolormesh(ionImageMRsize, cmap='msml2')
    fig.colorbar(pcm, ax=ax, shrink=0.5, pad=0.02)
    extent_ = (0, 0, 0, 26)

    # bounds[x0, y0, width, height]
    # Lower - left corner of inset Axes, and its width and height.

    # cb = ax.colorbar(pcm, shrink=0.5,)     # ticks=[0, 1, 2, 3, 4, 5], format='%.0f')
    # cb.ax.set_yticklabels(['backgr/////ound', 'GM-1', 'GM-2', 'WM-1', 'WM-2', 'connective\ntissue'], fontsize=5)
    # fig.tight_layout(rect=[0.0, 0.0, .95, 1])
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
    axins = ax.inset_axes([0.5, -0.05, 0.5, 0.5]) # bottom right corner
    axins.imshow(ionImageMRsize, origin='lower', zorder=2, cmap='msml2',)# loc='best')

    # subregion of the original image
    x1, x2, y1, y2 = 330, 440, 350, 430
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    ax.indicate_inset_zoom(axins, edgecolor="yellow")
    # axins = zoomed_inset_axes(ax, zoom=3, loc=1)
    # axins.imshow(ax, origin='lower', cmap='msml2', alpha=1.0, zorder=2)
    # axins.set_xlim(3 * rmin, 3 * (rmax + 1))
    # axins.set_ylim(3 * cmin, 3 * (cmax + 1))
    # mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    # plt.draw()
    # ax.axis('off')
    ax.set_title('Ion image on MR, m/z: {:.4f}'.format(clVal))
    plt.show()
if __name__ != '__main__':
    fig, ax = plt.subplots(figsize=[5, 4], dpi=1000)
    ax.imshow(mrSlice_x3, origin='lower', cmap='gray', alpha=1.0, zorder=0)
    ax.imshow(ionImageMRsize, origin='lower', cmap='msml2', alpha=1.0, zorder=1,
              # label=
              )
    # ax.legend('Ion image on MR, m/z: {:.4f}'.format(clVal))
    pcm = ax.pcolormesh(ionImageMRsize, cmap='msml2')
    fig.colorbar(pcm, orientation='vertical', anchor=(0.0, 1.0), ax=ax, shrink=0.5, pad=0.02)
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
    axins = zoomed_inset_axes(ax, zoom=3, loc='upper right')#, edgecolor='yellow')
    axins.imshow(mrSlice_x3, origin='lower', cmap='gray', alpha=1.0, zorder=2)
    axins.imshow(ionImageMRsize, origin='lower', cmap='msml2', alpha=1.0, zorder=5)
    x1, x2, y1, y2 = 330, 440, 350, 430
    # axins.spines['bottom'].set_color('yellow')
    # axins.spines['top'].set_color('#dddddd')
    # axins.spines['right'].set_color('red')
    # axins.spines['left'].set_color('red')
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    # axins.set_xticklabels([])
    # axins.set_yticklabels([])
    axins.axis('off')
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="yellow", zorder=4,
               # animated=True
               )
    # plt.draw()
    ax.indicate_inset_zoom(axins, edgecolor="yellow")
    ax.axis('off')
    # ax.set_title('Ion image on MR, m/z: {:.4f}'.format(clVal))
    # plt.legend()
    plt.show()




