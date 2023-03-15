import os
import numpy as np
import h5py
from scipy import ndimage
from skimage.transform import resize
from skimage.util import compare_images
from skimage.measure import regionprops, label
from skimage.morphology import area_closing
from utils import displayImage, multi_dil

proteinDir = r'/media/banikr/banikr/SpinalCordInjury/MALDI/210603-Chen_protein_slide_F'
saveSegPath = os.path.join(proteinDir, 'segmentation_vae_vae.h5')
with h5py.File(saveSegPath, 'r') as pfile:  # saves the data
    segImg = np.array(pfile['seg'])

maskImg = np.zeros_like(segImg)
maskImg[np.where(segImg != 0)] = 1

labeled_array, nFeatures = ndimage.label(maskImg, structure=np.ones((3, 3)))
print(nFeatures, "sections found...")

nSliceMALDI = [3] #[1, 2, 3, 4]

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
    secImg = segImg[minx:maxx, miny:maxy]
    displayImage(secImg, Title_='VAE segmentation {}'.format(secID))
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
    for tissue_label in [2, 1]:
        blobs_labels = label(secImg == tissue_label, background=0, connectivity=1)
        regionProperties = regionprops(label_image=blobs_labels)
        if tissue_label == 2:
            regionsBiggerToSmallerList = np.argsort([prop.area_filled for prop in regionProperties])[::-1][0:1]
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
    displayImage(secImg, Title_='moving image')
    # break
