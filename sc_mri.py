# +------------------------------------------------------+
# |   this file is to analyze the qMT or MTC MRI data    |
# |   .fdf files from Chase were converted to nifti      |
# |   files                                              |
# +------------------------------------------------------+
import nibabel as nib
import os
from glob import glob
import matplotlib as mtl
# mtl.use('TKAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from skimage.util import compare_images
from skimage.transform import resize
from skimage.measure import regionprops, label
from skimage.morphology import area_closing, dilation
import h5py
from sklearn.mixture import GaussianMixture as GMM
from scipy import ndimage
from kneed import KneeLocator
import SimpleITK as sitk
import cv2

def displayMR(MR, Title_= 'Demo'):
    plt.imshow(MR, origin='lower', cmap='gray')
    plt.title(Title_)
    plt.colorbar()
    plt.show()

def displayImage(matrix, Title_='demo'):
    if 'msml_list' not in plt.colormaps():
        colors = [(0.1, 0.1, 0.1), (0.9, 0, 0), (0, 0.9, 0), (0, 0, 0.9)]  # Bk -> R -> G -> Bl
        color_bin = 100
        mtl.colormaps.register(LinearSegmentedColormap.from_list(name='msml_list', colors=colors, N=color_bin))
    plt.imshow(matrix, origin='lower', cmap='msml_list')
    plt.title(Title_)
    plt.colorbar()
    plt.show()

def command_iteration(filter):
    global metric_values
    print(f"{filter.GetElapsedIterations():3} = {filter.GetMetric():10.5f}")
    metric_values.append(filter.GetMetric())
# +~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
# |  convert the individual MRI slices into a 3D block     |
# |  should run only once, if the MR cube is ok            |
# +~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
if __name__ != '__main__':
    dataDir = r'E:\SpinalCordInjury\from_Chase\MaldiRatNii\MaldiRat\Week 2\gems_MT_anat_axial_09.img'
    dataFiles = glob(os.path.join(dataDir, '*.nii'))
    for i, filepath in enumerate(dataFiles):
        img = nib.load(filepath).get_fdata()
        # print(img.shape)
        # displayImage(img)
        if i == 0:
            mr_block = np.zeros([img.shape[0], img.shape[1], len(dataFiles)])
            print(mr_block.shape)
        mr_block[..., i] = img

    out_nii_file = os.path.join(dataDir, '{}.nii.gz'.format(os.path.basename(dataDir)))
    print(out_nii_file)
    mr_img = nib.Nifti1Image(mr_block, affine=np.eye(4))
    nib.save(mr_img, out_nii_file)

# +~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
# |  perform segmentation and save it to directory     |
# |  should run only once, if the segmentation is ok.  |
# +~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
if __name__ != '__main__':
    proteinDir = r'E:\SpinalCordInjury\MALDI\210603-Chen_protein_slide_F'
    peakPath = os.path.join(proteinDir, 'sum_argrel.h5')   # change here for different sections.
    with h5py.File(peakPath, 'r') as pfile:
        print(pfile.keys())
        xloc = np.array(pfile.get('xloc'))
        yloc = np.array(pfile.get('yloc'))

    latentPath = glob(os.path.join(proteinDir, 'lat*.h5'))[0]
    print(latentPath)
    with h5py.File(latentPath, 'r') as pfile:
        print(pfile.keys())
        latent = np.array(pfile.get('latent'))
    print(latent.shape)
    with h5py.File(latentPath, 'r') as pfile:  # saves the data
        latent_space = np.array(pfile['latent'])#, dtype=np.float32)
    cov_Type = 'full'
    n_components = np.arange(3, 10)
    models = [GMM(n, covariance_type=cov_Type, max_iter=10000, random_state=1001, warm_start=True).fit(latent_space)
              for n in n_components]

    # print(np.unique(labels))
    # elements, counts = np.unique(labels, return_counts=True)
    # print(elements, counts)
    BIC_Scores = [m.bic(latent_space) for m in models]
    kneedle_point = KneeLocator(n_components, BIC_Scores, curve='convex', direction='decreasing')
    print('The suggested number of clusters = ', kneedle_point.knee)
    Elbow_idx = np.where(BIC_Scores == kneedle_point.knee_y)[0]
    from matplotlib.ticker import MaxNLocator
    plt.plot(n_components, BIC_Scores, '-g', marker='o', markerfacecolor='blue', markeredgecolor='orange',
             markeredgewidth='2', markersize=10, markevery=Elbow_idx)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(loc='best')
    plt.xlabel('Number of clusters')
    plt.ylabel('BIC score')
    plt.title('The suggested number of clusters = '+ np.str(kneedle_point.knee))
    plt.show()

    gmm = GMM(n_components=5, max_iter=10000, #kneedle_point.knee
              random_state=1001, warm_start=True)  # max_iter does matter, no random seed assigned
    labels = gmm.fit_predict(latent_space)
    labels += 1  # To Avoid conflict with the natural background value of 0
    lImg = np.zeros([max(xloc) + 1, max(yloc) + 1])
    for x, y, lb in zip(xloc, yloc, labels):
        lImg[x, y] = lb
    displayImage(lImg.T)
    saveSegPath = os.path.join(proteinDir, 'segmentation_vae_vae.h5')
    with h5py.File(saveSegPath, 'w') as pfile:  # saves the data
        pfile['seg'] = lImg

# +~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
# |    get segmentation of a section    |
# +~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
if __name__ == '__main__':
    # proteinDir = r'E:\SpinalCordInjury\MALDI\210603-Chen_protein_slide_F'
    proteinDir = r'/media/banikr/banikr/SpinalCordInjury/MALDI/210603-Chen_protein_slide_F'
    saveSegPath = os.path.join(proteinDir, 'segmentation_vae_vae.h5')

    with h5py.File(saveSegPath, 'r') as pfile:  # saves the data
        segImg = np.array(pfile['seg'])
    # displayImage(segImg)
    maskImg = np.zeros_like(segImg)
    maskImg[np.where(segImg != 0)] = 1
    # displayImage(maskImg)
    labeled_array, nFeatures = ndimage.label(maskImg, structure=np.ones((3, 3)))
    print(nFeatures, "sections found...")
    # displayImage(labeled_array)
    # secID = 1  # must be 1 to 4
    for secID in [2]:  # ,3,4]:
        minx, miny = np.inf, np.inf
        maxx, maxy = -np.inf, -np.inf
        for x in range(labeled_array.shape[0]):
            for y in range(labeled_array.shape[1]):
                if labeled_array[x, y] == secID:
                    minx, miny = min(minx, x), min(miny, y)
                    maxx, maxy = max(maxx, x), max(maxy, y)
        regionshape = [maxx - minx + 1,
                       maxy - miny + 1]
        # print(minx, miny, maxx, maxy)
        secImg = segImg[minx:maxx, miny:maxy]
    displayImage(secImg, Title_='VAE segmentation')
    # secImg[np.where(secImg == 1) or
    secImg[np.where(secImg == 2)] = 1
    secImg[np.where(secImg == 3)] = 2
    secImg[np.where(secImg == 4)] = 2
    secImg[np.where(secImg == 5)] = 3
    displayImage(secImg, Title_='3 label segmentation')

# +~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
# |    get anatomical mri image.....                           |
# |  1. How to locate sections to MR slices?                   |
# |     |_ Rostral or caudal?                                  |
# |  2. What kind of MR image preprocessings are required?     |
# |     |_ Cropped?                                            |
# |     |_ Normalization?                                      |
# |     |_ Any contrastive normalization?                      |
# |     |_ Creating a mask?                                    |
# |  3. MSI segmentations need to be resized/processed?        |
# |  4. Registration:                                          |
# |     |_ Affine/rigid?                                       |
# |     |_ Demon/Bspline?                                      |
# +~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
if __name__ == '__main__':
    mr_slice = 1
    # dataFold = r'E:\SpinalCordInjury\from_Chase\MaldiRatNii\MaldiRat\Week 2\gems_MT_anat_axial_09.img'
    dataFold = r'/media/banikr/banikr/SpinalCordInjury/from_Chase/MaldiRatNii/MaldiRat/Week 2/gems_MT_anat_axial_09.img'
    print(os.path.basename(dataFold))
    dataFiles = glob(os.path.join(dataFold, '*.nii'))
    # print(dataFiles)
    niiPath = os.path.join(dataFold, 'gems_MT_anat_axial_09.img.nii.gz')
    mr_block = nib.load(niiPath).get_fdata()
    # print(mr_block.shape)
    # mr_cropped = mr_block[55:72, 55:80, mr_slice]
    # rotated_mr_cropped = ndimage.rotate(mr_cropped, 180)
    # plt.imshow(mr_cropped, cmap='gray')
    # plt.colorbar()
    # plt.show()
    #
    # plt.imshow(rotated_mr_cropped, cmap='gray')
    # plt.colorbar()
    # plt.show()
    # mr_resized = resize(mr_cropped,
    #                     (560, 900),
    #                     mode='edge',
    #                     anti_aliasing=False,     # to preserve label values
    #                     preserve_range=True,
    #                     order=0)
    # # mr_cropped = mr_block[55:80, 55:70, 0]
    # plt.imshow(mr_resized, cmap='gray')
    # # plt.xlim(55, 80)
    # # plt.ylim(55, 70)
    # plt.colorbar()
    # plt.show()
    # print(np.max(rotated_mr_cropped))
    # mr_norm = rotated_mr_cropped/np.max(rotated_mr_cropped)
    # plt.imshow(mr_norm, cmap='gray')
    # plt.colorbar()
    # plt.show()
    cropMaskPath = os.path.join(dataFold, 'SC_mask_for_crop.nii.gz')
    crop_block = nib.load(cropMaskPath).get_fdata()

    mrImg = mr_block[..., mr_slice]
    mrMask = crop_block[..., mr_slice]
    mrImg[np.where(mrMask == 0)] = 0
    displayMR(mrImg)
    def bbox2(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax
    # print(bbox2(crop_block))
    rmin, rmax, cmin, cmax = bbox2(mrImg)
    # mr_block[np.where(crop_block == 0)] = 0
    # mr_cropped = mrImg[rmin-1:rmax+2, cmin-1:cmax+2]#, :]
    mr_cropped = mrImg[rmin:rmax + 1, cmin:cmax + 1]
    print(mr_cropped.shape)
    # resize mri just for visualization...
    # mr_resized = resize(mr_cropped,
    #                     secImg.T.shape,
    #                     # mode='edge',
    #                     anti_aliasing=False,  # to preserve label values
    #                     preserve_range=True,
    #                     order=0)
    # mr_slice = 2
    # plt.imshow(mr_cropped[..., mr_slice], cmap='gray')
    # plt.colorbar()
    # plt.show()
    # print("Finally we have a MSI segmentation and a MRI slice scan...")
    # mrImg = mr_cropped[..., mr_slice]
    # mrImgNorm = mr_cropped / np.max(mr_cropped)
    mrImgNorm = mr_cropped/ np.max(mr_cropped)
    displayMR(mrImgNorm, Title_="resized - normalized")
    # from scipy.ndimage.filters import maximum_filter, minimum_filter, gaussian_filter
    # blurred_image = maximum_filter(mrImgNorm, size=2)
    # # displayMR(blurred_image, Title_='maximum')
    # mrImgNorm = minimum_filter(blurred_image, size=3)
    # displayMR(mrImgNorm, Title_='maximum - minimum')
    #
    # radius = 3
    # mrImgNorm = gaussian_filter(mrImgNorm, sigma=radius)
    # displayMR(mrImgNorm, Title_='Gaussian')

    # todo
    # +~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
    # |   enhance the MR image: what can be done?
    # |   1. Get better data from Chase
    # |      |_ if not
    # |   2. Median or filter to denoise
    # |   3. CLAHE or mCLAHE for contrast enhancement
    # |   4. ITK-snap or photoshop
    # +~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
    # hist, bins = np.histogram(mr_cropped.flatten(), np.max(mr_cropped))
    # cdf = hist.cumsum()
    # cdf_normalized = cdf * float(hist.max()) / cdf.max()
    # plt.plot(cdf_normalized, color='b')
    # plt.hist(mr_cropped.flatten(), 256, [0, 256], color='r')
    # # plt.xlim([0, 256])
    # plt.legend(('cdf', 'histogram'), loc='upper left')
    # plt.show()
    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(mr_cropped)
    # new_img = np.zeros(mr_cropped.shape, dtype=np.uint8)
    # new_img = cv2.normalize(mr_cropped, new_img, 0, 1, cv2.NORM_MINMAX)
    # displayMR(mr_cropped, Title_='old')
    # displayMR(new_img, Title_='new')

    # from skimage import exposure
    # data_rescaled = exposure.rescale_intensity(mr_cropped, out_range=(0, 1))
    # data_eq = exposure.equalize_hist(data_rescaled)
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    #
    # axes[0].imshow(data_rescaled, cmap='gray')
    # axes[0].set_title('Rescaled')
    # axes[1].imshow(data_eq, cmap='gray')
    # axes[1].set_title('Histogram Equalized')
    # plt.show()

    # array = mr_cropped # sitk.GetArrayFromImage(mr_cropped)
    #
    # # Normalize the array
    # array = (array - np.min(array)) / (np.max(array) - np.min(array))
    #
    # # Perform histogram equalization
    # array_eq = sitk.GetArrayFromImage(sitk.EqualizeHistogram(sitk.GetImageFromArray(array)))
    # # Convert back to an image
    # # image_eq = sitk.GetImageFromArray(array_eq)
    # # image_eq.CopyInformation(image)
    # displayMR(array, Title_='old')
    # displayMR(array_eq, Title_='new')
# +~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
# |    morphological operations     |
# |    1. Gray Matter(butterfly)    |
# |    2. White Matter              |
# |    3. Connective tissues        |
# +~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
if __name__ == '__main__':
    secImgMorphed = np.zeros_like(secImg)
    for tissue_label in [2, 1]:
        blobs_labels = label(secImg == tissue_label, background=0, connectivity=2)
        regionProperties = regionprops(label_image=blobs_labels)
        if tissue_label == 2:
            regionsBiggerToSmallerList = np.argsort([prop.area_filled for prop in regionProperties])[::-1][0:3]
        if tissue_label == 1:
            regionsBiggerToSmallerList = np.argsort([prop.area_filled for prop in regionProperties])[::-1][0:1]
        morphedTissueImg = np.zeros_like(blobs_labels)
        for region in regionsBiggerToSmallerList:
            for coord in regionProperties[region].coords:
                # print(coord)
                morphedTissueImg[coord[0], coord[1]] = 1
        # displayImage(morphedTissueImg, )
        square = np.array([[1, 1],
                           [1, 1]])

        def multi_dil(im, num, element=square):
            for i in range(num):
                im = dilation(im, element)
            return im

        morphedTissueImg = multi_dil(morphedTissueImg, 1)
        morphedTissueImg = area_closing(morphedTissueImg, area_threshold=1000, connectivity=50)
        # displayImage(morphedTissueImg)
        secImgMorphed[np.where(morphedTissueImg)] = tissue_label
    displayImage(secImgMorphed, Title_="morphed")
    # secImgMorphed[np.where(secImg != 0) and np.where(secImgMorphed == 0)] = secImg
    # secImg[np.where(secImgMorphed != 0)] = secImgMorphed
    # print("label morphed...")
    # displayImage(secImg, Title_="morphed 2")
    for tissue_label in [1, 2]:
        mask = (secImgMorphed == tissue_label)
        secImg[mask] = tissue_label
    displayImage(secImg, 'Morphed 2')

    # secImg_resized = resize(secImg,
    #                         mrImgNorm.shape,
    #                         # mode='edge',
    #                         anti_aliasing=False,  # to preserve label values
    #                         preserve_range=True,
    #                         order=0)
    # displayImage(secImg_resized, Title_="resized")
    # print("Resized segmentation image..")
    mrImgNorm = resize(mrImgNorm,
                       secImg.T.shape,
                       # mode='edge',
                       anti_aliasing=False,  # to preserve label values
                       preserve_range=True,
                       order=1)
    displayMR(mrImgNorm, Title_="resized-norm")
    # print("Resized segmentation image..")
    # secImg[np.where(secImg == 3)] = 0
    lut = np.array([0, 3, 1, 2])
    # displayImage(lut[secImg])
    secImg_ = np.zeros_like(secImg)
    for i, l in enumerate(lut):
        secImg_[np.where(secImg == i)] = l
    displayImage(secImg_, Title_='label exchanged')
    secImg = secImg_

# +~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
# |   Segmentation - MR registration   |
# +~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
if __name__ == '__main__':
    mrItk = sitk.GetImageFromArray(mrImgNorm)
    segItk = sitk.GetImageFromArray(secImg)
    fixed_image = sitk.Cast(mrItk, sitk.sitkFloat32)
    moving_image = sitk.Cast(segItk, sitk.sitkFloat32)
    print("image shapes: ", fixed_image.GetSize(), moving_image.GetSize())
    all_orientations = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
    # print(all_orientations)

    # Evaluate the similarity metric using the rotation parameter space sampling, translation remains the same for all.
    initial_transform = sitk.Euler2DTransform(sitk.CenteredTransformInitializer(fixed_image,
                                                                                moving_image,
                                                                                sitk.Euler2DTransform(),
                                                                                sitk.CenteredTransformInitializerFilter.GEOMETRY))
    # Registration framework setup.
    registration_method = sitk.ImageRegistrationMethod()
    # registration_method.SetMetricAsMeanSquares()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=500)
    registration_method.SetMetricSamplingStrategy(registration_method.NONE)
    registration_method.SetMetricSamplingPercentage(0.1)
    registration_method.SetInitialTransform(initial_transform, inPlace=True)
    registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0,
                                                                 minStep=1e-4,
                                                                 numberOfIterations=500,
                                                                 gradientMagnitudeTolerance=1e-8)
    registration_method.SetInterpolator(sitk.sitkNearestNeighbor)
    registration_method.SetOptimizerScalesFromIndexShift()
    # best_orientation = (0.0, 0.0)
    best_similarity_value = registration_method.MetricEvaluate(fixed_image, moving_image)
    similarity_value = []
    # Iterate over all other rotation parameter settings.
    for key, orientation in enumerate(all_orientations):  # .items():
        initial_transform.SetAngle(orientation)
        registration_method.SetInitialTransform(initial_transform)
        current_similarity_value = registration_method.MetricEvaluate(fixed_image, moving_image)
        similarity_value.append(current_similarity_value)
        # print("current similarity value: ", current_similarity_value)
        if current_similarity_value <= best_similarity_value:
            best_similarity_value = current_similarity_value
            best_orientation = orientation
        # else:
        #     best_orientation = orientation
    print('best orientation is: ' + str(best_orientation))
    print(current_similarity_value)

    plt.plot(all_orientations, similarity_value, 'b')
    plt.plot(best_orientation, best_similarity_value, 'rv')
    plt.show()

    initial_transform.SetAngle(best_orientation)
    registration_method.SetInitialTransform(initial_transform, inPlace=True)
    eulerTx = registration_method.Execute(fixed_image, moving_image)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(eulerTx)

    out = resampler.Execute(moving_image)
    displayImage(sitk.GetArrayFromImage(out), Title_='registered image(Euler2D)')
    # displayImage(sitk.GetArrayFromImage(moving_image), Title_='moving image')
    # displayImage(sitk.GetArrayFromImage(fixed_image), Title_='fixed image')
    checkerImg = compare_images(sitk.GetArrayFromImage(out),
                                sitk.GetArrayFromImage(fixed_image),
                                method='blend')
    displayImage(checkerImg, Title_='mask-label overlapped')
    del registration_method, initial_transform, eulerTx
    moving_image = sitk.Cast(out, sitk.sitkFloat32)

    metric_values = []
    # demons = sitk.DemonsRegistrationFilter()
    demons = sitk.FastSymmetricForcesDemonsRegistrationFilter()
    demons.SetNumberOfIterations(5000)
    demons.SetStandardDeviations(1.2)
    # demons.AddCommand(sitk.sitkStartEvent, start_plot)
    # demons.AddCommand(sitk.sitkEndEvent, end_plot)
    # demons.AddCommand(sitk.sitkIterationEvent, lambda: plot_values_(demons))
    demons.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(demons))
    # # metric_values.append()
    transform_to_displacment_field_filter = sitk.TransformToDisplacementFieldFilter()
    transform_to_displacment_field_filter.SetReferenceImage(fixed_image)
    displacementTx = sitk.DisplacementFieldTransform(transform_to_displacment_field_filter.Execute
                                                     (sitk.Transform(2, sitk.sitkIdentity)))
    displacementTx.SetSmoothingGaussianOnUpdate(varianceForUpdateField=7.75, varianceForTotalField=0.5)
    # registration_method.SetMovingInitialTransform(eulerTx)
    # registration_method.SetMetricAsANTSNeighborhoodCorrelation(4)
    # registration_method.MetricUseFixedImageGradientFilterOff()
    # registration_method.Execute(fixed_image, moving_image)
    # compositeTx = sitk.CompositeTransform([displacementTx])  #
    displacementField = transform_to_displacment_field_filter.Execute(displacementTx)
    displacementField = demons.Execute(fixed_image, moving_image, displacementField)
    outTx = sitk.DisplacementFieldTransform(displacementField)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(outTx)
    out = resampler.Execute(moving_image)
    # registration_method.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(registration_method))
    displayImage(sitk.GetArrayFromImage(out), Title_='Euler + FastSymmetricForcesDemonsRegistrationFilter')
    displayImage(sitk.GetArrayFromImage(moving_image), Title_='moving image')
    displayMR(sitk.GetArrayFromImage(fixed_image), Title_='fixed image')
    checkerImg = compare_images(sitk.GetArrayFromImage(out),
                                sitk.GetArrayFromImage(fixed_image),
                                method='blend')
    displayImage(checkerImg, Title_='Euler + FastSymmetricForcesDemonsRegistrationFilter')
    # transform_to_displacment_field_filter = sitk.TransformToDisplacementFieldFilter()
    # transform_to_displacment_field_filter.SetReferenceImage(fixed_image)
    # initial_transform = sitk.DisplacementFieldTransform(
    #     transform_to_displacment_field_filter.Execute(sitk.Transform(2, sitk.sitkIdentity)))
    # initial_transform.SetSmoothingGaussianOnUpdate(varianceForUpdateField=0.0, varianceForTotalField=2.0)
    # registration_method = sitk.ImageRegistrationMethod()
    # registration_method.SetInitialTransform(initial_transform)
    # registration_method.SetInterpolator(sitk.sitkNearestNeighbor)
    # registration_method.SetMetricAsDemons(10)
    # # Optimizer settings.
    # registration_method.SetOptimizerAsGradientDescent(
    #     learningRate=5.0,
    #     numberOfIterations=1000,
    #     convergenceMinimumValue=1e-6,
    #     convergenceWindowSize=20)
    # registration_method.SetOptimizerScalesFromPhysicalShift()
    #
    # final_transform = registration_method.Execute(fixed_image, out)
    # moving_resampled = sitk.Resample(
    #     out,
    #     fixed_image,
    #     final_transform,
    #     sitk.sitkNearestNeighbor,
    #     0.0,
    #     moving_image.GetPixelID())
    # # plt.imshow(sitk.GetArrayFromImage(moving_resampled))
    # # plt.colorbar()
    # # plt.show()
    # # plt.imshow(sitk.GetArrayFromImage(moving_image)-sitk.GetArrayFromImage(moving_resampled))
    # # plt.colorbar()
    # displayImage(sitk.GetArrayFromImage(moving_resampled), Title_='registered segmentation')
    # displayImage(sitk.GetArrayFromImage(mask_itk), Title_='annotated hne mask')
    plt.plot(metric_values)
    plt.ylabel("Registration metric")
    plt.xlabel("Iterations")
    plt.show()









