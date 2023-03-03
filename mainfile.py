import numpy as np
import SimpleITK as sitk
import os
from glob import glob
import nibabel as nib
import h5py
import matplotlib.pyplot as plt
import matplotlib as mtl
from matplotlib.colors import LinearSegmentedColormap
from skimage.transform import resize
from skimage.util import compare_images
from skimage.measure import regionprops, label
from skimage.morphology import area_closing, dilation
from scipy import ndimage
import sys
# sys.path.append('C:\\Users\\ranab\\PycharmProjects\\Registration\\registraion')
# sys.path.append(r'/home/banikr/PycharmProjects/registration')
import random
# import registration_utilities as ru
from PIL import Image
random.seed(1001)

# +-------------------------------------+
# |    prepare mask and segmentation    |
# +-------------------------------------+
# fileDir = r'E:\SpinalCordInjury\MALDI\HistologyHnE'
fileDir = r'/media/banikr/banikr/SpinalCordInjury/MALDI/HistologyHnE'
filePaths = glob(os.path.join(fileDir, '*.Jpeg'))
print(filePaths)
if __name__ != '__main__':  # this should be run only once... because it saves the images.
    fig, axes = plt.subplots(1, 4, figsize=(30, 20), dpi=200)
    ax = axes.ravel()
    for i, inputFileName in enumerate(filePaths[0:4]):
        reader = sitk.ImageFileReader()
        reader.SetImageIO("JPEGImageIO")
        reader.SetFileName(inputFileName)
        image = reader.Execute()

        dataMat = sitk.GetArrayFromImage(image)
        ax[i].imshow(dataMat)
        ax[i].set_title("{}".format(image.GetSize()), color='c', fontsize=40)
    fig.tight_layout()
    plt.show()

    nImg = 2    # image index
    fig, axes = plt.subplots(1, 3, figsize=(30, 20), dpi=100)
    ax = axes.ravel()
    img_rgb = np.asarray(Image.open(filePaths[nImg]).convert('RGB'))
    img_L = Image.open(filePaths[nImg]).convert('L')
    mk = np.asarray(img_L)
    # print(mk.shape, np.max(mk), np.min(mk))
    binarr = np.where(mk > 90, 0, 1)  # this value must be found?
    # plt.imshow(binarr, cmap='gray')
    # plt.show()
    minx, miny = np.inf, np.inf
    maxx, maxy = -np.inf, -np.inf
    for x in range(binarr.shape[0]):
        for y in range(binarr.shape[1]):
            if binarr[x, y] == 1:
                minx, miny = min(minx, x), min(miny, y)
                maxx, maxy = max(maxx, x), max(maxy, y)
    regionshape = [maxx - minx + 1,
                   maxy - miny + 1]
    # print(regionshape)
    # plt.imshow(binarr[minx:maxx, miny:maxy])
    ax[0].imshow(img_rgb)
    ax[1].imshow(binarr, cmap='gray')
    ax[2].imshow(img_rgb[minx:maxx, miny:maxy, :])
    fig.tight_layout()
    plt.show()
    hneImg = img_rgb[minx:maxx, miny:maxy, :]
    hne_gray = mk[minx:maxx, miny:maxy]
    print(hne_gray.shape)
    hnegray_itk = sitk.GetImageFromArray(hne_gray)
    nib.save(nib.Nifti1Image(hneImg, np.eye(4)), os.path.join(fileDir, 'cropped_hne_image_3.nii'))

# +----------------------+
# |   helper functions   |
# +----------------------+
def displayImage(matrix, Title_='demo'):
    if 'msml_list' not in plt.colormaps():
        colors = [(0.1, 0.1, 0.1), (0.9, 0, 0), (0, 0.9, 0), (0, 0, 0.9)]  # Bk -> R -> G -> Bl
        color_bin = 100
        mtl.colormaps.register(LinearSegmentedColormap.from_list(name='msml_list', colors=colors, N=color_bin))
    plt.imshow(matrix, origin='lower', cmap='msml_list')
    plt.title(Title_)
    plt.colorbar()
    plt.show()

# def command_iteration(method):
#     global metric_values
#     print("is this running? ")
#     if method.GetOptimizerIteration() == 31:
#         # metric_values = []
#         print(f"\tLevel: {method.GetCurrentLevel()}")
#         print(f"\tScales: {method.GetOptimizerScales()}")
#     print(f"#{method.GetOptimizerIteration()}")
#     print(f"\tMetric Value: {method.GetMetricValue():10.5f}")
#     print(f"\tLearningRate: {method.GetOptimizerLearningRate():10.5f}")
#     if method.GetOptimizerConvergenceValue() != sys.float_info.max:
#         print(
#             "\t Convergence Value: "
#             + f"{method.GetOptimizerConvergenceValue():.5e}"
#         )
#     metric_values.append(method.GetMetricValue())
#     # # print(f"{method.():3} = {method.GetMetricValue():10.5f}")
#     plt.plot(metric_values, 'r')
#     # # plt.plot(multires_iterations, [metric_values[index] for index in multires_iterations], 'b*')
#     plt.xlabel('Iteration Number', fontsize=12)
#     plt.ylabel('Metric Value', fontsize=12)
#     plt.show()
#     # return metric_values

def command_iteration2(method):
    global metric_values
    # print("metric value: {}".format(method.GetMetricValue()))
    metric_values.append(method.GetMetricValue())

def start_plot():
    global metric_values, multires_iterations

    metric_values = []
    multires_iterations = []

# Callback invoked when the EndEvent happens, do cleanup of data and figure.
def end_plot():
    global metric_values, multires_iterations

    del metric_values
    del multires_iterations
    # Close figure, we don't want to get a duplicate of the plot latter on.
    plt.close()

# Callback invoked when the IterationEvent happens, update our data and display new figure.
def plot_values(registration_method):
    global metric_values, multires_iterations

    metric_values.append(registration_method.GetMetricValue())
    # Clear the output area (wait=True, to reduce flickering), and plot current data
    # clear_output(wait=True)
    # Plot the similarity metric values
    plt.plot(metric_values, 'r')
    # plt.plot(multires_iterations, [metric_values[index] for index in multires_iterations], 'b*')
    plt.xlabel('Iteration Number', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    plt.show()

def plot_values_(filter):
    global metric_values, multires_iterations

    metric_values.append(filter.GetMetricValue())
    # Clear the output area (wait=True, to reduce flickering), and plot current data
    # clear_output(wait=True)
    # Plot the similarity metric values
    plt.plot(metric_values, 'r')
    # plt.plot(multires_iterations, [metric_values[index] for index in multires_iterations], 'b*')
    plt.xlabel('Iteration Number', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    plt.show()

# +----------------------------------------------+
# |    load and preprocess segmentation files    |
# +----------------------------------------------+
if __name__ == '__main__':
    # reconPath = os.path.join(fileDir, 'seg_3.h5')   # change here for different sections.
    # with h5py.File(reconPath, 'r') as pfile:     # saves the data
    #     labelMapMALDI = np.array(pfile['segmentation_3'])
    #
    # # plt.imshow(segImg==2)
    # # segImg[segImg==2]= 0
    # labelMapMALDI_t = labelMapMALDI.T   # aligns
    # labelMapMALDI_t[labelMapMALDI_t == 2] = 0
    # labelMapMALDI_t[labelMapMALDI_t == 1] = 2  # changing WM to match mask
    # labelMapMALDI_t[labelMapMALDI_t == 4] = 1
    # # segImg[5:7, 65:67] = 3
    # displayImage(labelMapMALDI_t, Title_='After alignment and label correction')
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
    for secID in [3]:  # ,3,4]:
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
    # displayImage(secImgMorphed, Title_="morphed")
    # secImgMorphed[np.where(secImg != 0) and np.where(secImgMorphed == 0)] = secImg
    # secImg[np.where(secImgMorphed != 0)] = secImgMorphed
    # print("label morphed...")
    # displayImage(secImg, Title_="morphed 2")
    for tissue_label in [1, 2]:
        mask = (secImgMorphed == tissue_label)
        secImg[mask] = tissue_label
    # displayImage(secImg, 'Morphed 2')

    # lut = np.array([0, 3, 1, 2])
    # # displayImage(lut[secImg])
    # secImg_ = np.zeros_like(secImg)
    # for i, l in enumerate(lut):
    #     secImg_[np.where(secImg == i)] = l
    # displayImage(secImg_, Title_='label exchanged')
    # secImg = secImg_

    maskpath = glob(os.path.join(fileDir, 'cropped_hne_mask_3.nii.gz'))[0]
    maskHnE = nib.load(maskpath).get_fdata()
    maskHnE_0 = maskHnE[..., 0]
    displayImage(maskHnE_0, Title_='before resize')
    print(np.unique(maskHnE_0))
    maskHnE_0_resized = resize(maskHnE_0,
                               secImg.T.shape,
                               # mode='edge',
                               anti_aliasing=False,      # to preserve label values
                               preserve_range=True,
                               order=0)
    print(np.unique(maskHnE_0_resized), np.mean(maskHnE_0_resized), np.std(maskHnE_0_resized))
    count_fraction = np.zeros_like(maskHnE_0_resized)
    count_fraction[np.where(maskHnE_0_resized == 1)] = 1
    count_fraction[np.where(maskHnE_0_resized == 2)] = 1
    if (np.unique(maskHnE_0) == np.unique(maskHnE_0_resized)).all():
        print("all labels are same after resize")
    else:
        print("labels changed after resize")
        raise ValueError
    displayImage(maskHnE_0_resized, Title_='after resize')

    mask_itk = sitk.GetImageFromArray(secImg)
    label_itk = sitk.GetImageFromArray(maskHnE_0_resized)
    fixed_image = sitk.Cast(label_itk, sitk.sitkFloat32)
    moving_image = sitk.Cast(mask_itk, sitk.sitkFloat32)

if __name__ != '__main__': # doesn't work
    transform_to_displacment_field_filter = sitk.TransformToDisplacementFieldFilter()
    transform_to_displacment_field_filter.SetReferenceImage(fixed_image)
    initial_transform = sitk.DisplacementFieldTransform(transform_to_displacment_field_filter.Execute(sitk.Transform(2, sitk.sitkIdentity)))

    initial_transform.SetSmoothingGaussianOnUpdate(varianceForUpdateField=0.0, varianceForTotalField=2.0)
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetInitialTransform(initial_transform)
    registration_method.SetInterpolator(sitk.sitkNearestNeighbor)
    registration_method.SetMetricAsDemons(10)
    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=50.0,
        numberOfIterations=1000,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=20)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    final_transform = registration_method.Execute(fixed_image, moving_image)
    out_itk = sitk.Resample(
        moving_image,
        fixed_image,
        final_transform,
        sitk.sitkNearestNeighbor,
        0.0,
        moving_image.GetPixelID())
    displayImage(sitk.GetArrayFromImage(out_itk), Title_='registered mask')
    displayImage(sitk.GetArrayFromImage(mask_itk), Title_='annotated hne mask')

if __name__ != '__main__':
    def command_iteration2(method):
        global metric_values
        # print(f"{filter.GetElapsedIterations():3} = {filter.GetMetric():10.5f}")
    # def command_iteration_(filter):
    #     global metric_values
        metric_values.append(method.GetMetricValue())
    metric_values = []
    initialTx = sitk.CenteredTransformInitializer(fixed_image, moving_image,
                                                  sitk.AffineTransform(fixed_image.GetDimension()))
    # print(initialTx)
    registrationMethod = sitk.ImageRegistrationMethod()

    # registrationMethod.SetShrinkFactorsPerLevel([1, 1, 1])
    # registrationMethod.SetSmoothingSigmasPerLevel([1, 1, 1])

    # registrationMethod.SetMetricAsJointHistogramMutualInformation(20)
    registrationMethod.MetricUseFixedImageGradientFilterOff()

    registrationMethod.SetOptimizerAsGradientDescent(
        learningRate=0.50,
        numberOfIterations=1000,
        estimateLearningRate=registrationMethod.EachIteration,
    )
    registrationMethod.SetOptimizerScalesFromPhysicalShift()
    registrationMethod.SetInitialTransform(initialTx)
    registrationMethod.SetInterpolator(sitk.sitkNearestNeighbor)

    # # registrationMethod.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(registrationMethod))
    # # registrationMethod.AddCommand(sitk.sitkMultiResolutionIterationEvent,
    # #                               lambda: command_multiresolution_iteration(registrationMethod))

    outTx1 = registrationMethod.Execute(fixed_image, moving_image)

    print("-------")
    print(outTx1)
    # print(f"Optimizer stop condition: {registrationMethod.GetOptimizerStopConditionDescription()}")
    # print(f" Iteration: {registrationMethod.GetOptimizerIteration()}")
    # print(f" Metric value: {registrationMethod.GetMetricValue()}")

    displacementField = sitk.Image(fixed_image.GetSize(), sitk.sitkVectorFloat64)
    displacementField.CopyInformation(fixed_image)
    displacementTx = sitk.DisplacementFieldTransform(displacementField)
    # # del displacementField
    displacementTx.SetSmoothingGaussianOnUpdate(
        varianceForUpdateField=0.0, varianceForTotalField=1.5
    )

    registrationMethod.SetMovingInitialTransform(outTx1)
    registrationMethod.SetInitialTransform(displacementTx, inPlace=False)

    registrationMethod.SetMetricAsANTSNeighborhoodCorrelation(4)
    registrationMethod.MetricUseFixedImageGradientFilterOff()

    # registrationMethod.SetShrinkFactorsPerLevel([3, 2, 1])
    # registrationMethod.SetSmoothingSigmasPerLevel([2, 1, 1])

    registrationMethod.SetOptimizerScalesFromPhysicalShift()
    registrationMethod.SetOptimizerAsGradientDescent(
        learningRate=10.0,
        numberOfIterations=3000,
        estimateLearningRate=registrationMethod.EachIteration,
    )

    registrationMethod.Execute(fixed_image, moving_image)
    registrationMethod.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration2(registrationMethod))
    # print("-------")
    # print(displacementTx)
    # print(f"Optimizer stop condition: {registrationMethod.GetOptimizerStopConditionDescription()}")
    # print(f" Iteration: {registrationMethod.GetOptimizerIteration()}")
    # print(f" Metric value: {registrationMethod.GetMetricValue()}")

    compositeTx = sitk.CompositeTransform([outTx1, displacementTx])

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(compositeTx)

    out = resampler.Execute(moving_image)
    displayImage(sitk.GetArrayFromImage(out), Title_='registered(Affine+Disp) ')
    displayImage(sitk.GetArrayFromImage(moving_image), Title_='label map')
    # displayImage(sitk.GetArrayFromImage(fixed_image))
    checkerImg = compare_images(sitk.GetArrayFromImage(out),
                                sitk.GetArrayFromImage(fixed_image),
                                method='diff')
    displayImage(checkerImg, Title_='final registration, fixed-moving overlapped')
    plt.plot(metric_values)
    plt.show()

# +------------------------+
# |    distance map/SDF    |
# +------------------------+
if __name__ != '__main__':
    b_value = 2
    displayImage(sitk.GetArrayFromImage(fixed_image) == b_value, Title_='selected region in fixed image')
    fixed_image_distance_map = sitk.SignedMaurerDistanceMap(fixed_image == b_value, squaredDistance=True, useImageSpacing=True)
    b_value = 1
    displayImage(sitk.GetArrayFromImage(moving_image) == b_value, Title_='selected region in moving image')
    moving_image_distance_map = sitk.SignedMaurerDistanceMap(moving_image == b_value, squaredDistance=True, useImageSpacing=True)

    initialTx = sitk.CenteredTransformInitializer(fixed_image_distance_map,     #  when orientation is not similar in our case shouldn't use.
                                                  moving_image_distance_map,
                                                  sitk.AffineTransform(fixed_image.GetDimension()))
    registrationMethod = sitk.ImageRegistrationMethod()
    # registrationMethod.SetShrinkFactorsPerLevel([1, 1, 1])
    # registrationMethod.SetSmoothingSigmasPerLevel([1, 1, 1])

    # registrationMethod.SetMetricAsJointHistogramMutualInformation(20)
    registrationMethod.MetricUseFixedImageGradientFilterOff()
    registrationMethod.SetOptimizerAsGradientDescent(
        learningRate=10.0,
        numberOfIterations=1000,
        estimateLearningRate=registrationMethod.EachIteration,
    )
    registrationMethod.SetOptimizerScalesFromPhysicalShift()
    registrationMethod.SetInitialTransform(initialTx)
    registrationMethod.SetInterpolator(sitk.sitkNearestNeighbor)

    # registrationMethod.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(registrationMethod))
    # registrationMethod.AddCommand(sitk.sitkMultiResolutionIterationEvent,
    #                               lambda: command_multiresolution_iteration(registrationMethod))

    outTx1 = registrationMethod.Execute(fixed_image_distance_map, moving_image_distance_map)

    # print("-------")
    # print(outTx1)
    # print(f"Optimizer stop condition: {registrationMethod.GetOptimizerStopConditionDescription()}")
    # print(f" Iteration: {registrationMethod.GetOptimizerIteration()}")
    # print(f" Metric value: {registrationMethod.GetMetricValue()}")

    displacementField = sitk.Image(fixed_image_distance_map.GetSize(), sitk.sitkVectorFloat64)
    displacementField.CopyInformation(fixed_image_distance_map)
    displacementTx = sitk.DisplacementFieldTransform(displacementField)
    # del displacementField
    displacementTx.SetSmoothingGaussianOnUpdate(
        varianceForUpdateField=0.0, varianceForTotalField=1.5
    )

    registrationMethod.SetMovingInitialTransform(outTx1)
    registrationMethod.SetInitialTransform(displacementTx, inPlace=True)

    registrationMethod.SetMetricAsANTSNeighborhoodCorrelation(4)
    registrationMethod.MetricUseFixedImageGradientFilterOff()

    # registrationMethod.SetShrinkFactorsPerLevel([3, 2, 1])
    # registrationMethod.SetSmoothingSigmasPerLevel([2, 1, 1])

    registrationMethod.SetOptimizerScalesFromPhysicalShift()
    registrationMethod.SetOptimizerAsGradientDescent(
        learningRate=1,
        numberOfIterations=300,
        estimateLearningRate=registrationMethod.EachIteration,
    )

    registrationMethod.Execute(fixed_image_distance_map, moving_image_distance_map)

    # print("-------")
    # print(displacementTx)
    # print(f"Optimizer stop condition: {registrationMethod.GetOptimizerStopConditionDescription()}")
    # print(f" Iteration: {registrationMethod.GetOptimizerIteration()}")
    # print(f" Metric value: {registrationMethod.GetMetricValue()}")

    compositeTx = sitk.CompositeTransform([outTx1, displacementTx])

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image_distance_map)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(outTx1)

    out = resampler.Execute(moving_image)

    displayImage(sitk.GetArrayFromImage(out), Title_='registered image')
    displayImage(sitk.GetArrayFromImage(moving_image), Title_='moving image')
    displayImage(sitk.GetArrayFromImage(fixed_image), Title_='fixed image')
    checkerImg = compare_images(sitk.GetArrayFromImage(out),
                                sitk.GetArrayFromImage(fixed_image),
                                method='blend')
    displayImage(checkerImg)

# +--------------------------------+
# |    direct affine transform     |
# +--------------------------------+
if __name__ != '__main__':
    affine = sitk.AffineTransform(fixed_image.GetDimension())
    print(affine)
    metric_values = []
    registration_method = sitk.ImageRegistrationMethod()
    # registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=500)
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.001)
    registration_method.SetInitialTransform(affine, inPlace=True)
    registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0,
                                                                 minStep=1e-4,
                                                                 numberOfIterations=500,
                                                                 gradientMagnitudeTolerance=1e-8)
    registration_method.SetInterpolator(sitk.sitkNearestNeighbor)

    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration2(registration_method))
    tx = registration_method.Execute(fixed_image, moving_image)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(tx)
    out = resampler.Execute(moving_image)

    plt.plot(metric_values)
    plt.show()

# +--------------------------------+
# |    set angle euler works!!     |
# +--------------------------------+
if __name__ != '__main__':   # Reference 4 in the notebook
    # all_orientations = {'x=0, y=180': (0.0, np.pi),
    #                     'x=180, y=180': (np.pi, 0.0)}
                        # 'x=0, y=180, z=180': (0.0, np.pi, np.pi)}

    # all_orientations = {'demo angle:': np.pi/2}
    # print(np.pi)
    all_orientations = np.linspace(-2*np.pi, 2*np.pi, 100)
    # print(all_orientations)

    # Evaluate the similarity metric using the rotation parameter space sampling, translation remains the same for all.
    initial_transform = sitk.Euler2DTransform(sitk.CenteredTransformInitializer(fixed_image,
                                                                                moving_image,
                                                                                sitk.Euler2DTransform(),
                                                                                sitk.CenteredTransformInitializerFilter.GEOMETRY))
    # Registration framework setup.
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=500)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.001)
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
    for key, orientation in enumerate(all_orientations):    #   .items():
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
    displayImage(sitk.GetArrayFromImage(moving_image), Title_='moving image')
    displayImage(sitk.GetArrayFromImage(fixed_image), Title_='fixed image')
    checkerImg = compare_images(sitk.GetArrayFromImage(out),
                                sitk.GetArrayFromImage(fixed_image),
                                method='diff')
    displayImage(checkerImg, Title_='mask-label overlapped')
    # del registration_method, initial_transform, eulerTx
    moving_image = sitk.Cast(out, sitk.sitkFloat32)

# +----------------------------------------------------------------------+
# |   demon registration is working amazingly.                           |
# |   the iteration plateued after ~700 iterations.                      |
# |   1. can it be improved ?                                            |
# |   2. what are other better methods ?                                 |
# +----------------------------------------------------------------------+
if __name__ != '__main__':
    def command_iteration(filter):
        global metric_values
        print(f"{filter.GetElapsedIterations():3} = {filter.GetMetric():10.5f}")
    # def command_iteration_(filter):
    #     global metric_values
        metric_values.append(filter.GetMetric())
        # global metric_values
        # print(f"{filter.GetElapsedIterations():3} = {filter.GetMetric():10.5f}")
        # # plt.plot(metric_values, 'r')
        # metric_values.append(filter.GetMetric())
    #     # plt.plot(multires_iterations, [metric_values[index] for index in multires_iterations], 'b*')
    #     plt.xlabel('Iteration Number', fontsize=12)
    #     plt.ylabel('Metric Value', fontsize=12)
    #     plt.show()

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
    displacementTx.SetSmoothingGaussianOnUpdate(varianceForUpdateField=0.0, varianceForTotalField=1.5)
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
    displayImage(sitk.GetArrayFromImage(fixed_image), Title_='fixed image')
    checkerImg = compare_images(sitk.GetArrayFromImage(out),
                                sitk.GetArrayFromImage(fixed_image),
                                method='diff')
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

# +------------------------------------------+
# |  works for all the demons                |
# |  multiresolution demons + displacement   |
# +------------------------------------------+
if __name__ != '__main__':
    shrink_factors = [2, 2, 1]
    smoothing_sigmas = [4, 4, 0]
    # if np.isscalar(shrink_factors):
    #     shrink_factors = [shrink_factors] * moving_image.GetDimension()
    # if np.isscalar(smoothing_sigmas):
    #     smoothing_sigmas = [smoothing_sigmas] * moving_image.GetDimension()
    #
    # smoothed_image = sitk.SmoothingRecursiveGaussian(moving_image, smoothing_sigmas)
    #
    # original_spacing = moving_image.GetSpacing()
    # original_size = moving_image.GetSize()
    # new_size = [
    #     int(sz / float(sf) + 0.5) for sf, sz in zip(shrink_factors, original_size)
    # ]
    # new_spacing = [
    #     ((original_sz - 1) * original_spc) / (new_sz - 1)
    #     for original_sz, original_spc, new_sz in zip(
    #         original_size, original_spacing, new_size
    #     )
    # ]
    # print("new size >> ", new_size)
    # print("new spacing >>", new_spacing)
    def smooth_and_resample(image, shrink_factors, smoothing_sigmas):
        """
        Args:
            image: The image we want to resample.
            shrink_factor(s): Number(s) greater than one, such that the new image's size is original_size/shrink_factor.
            smoothing_sigma(s): Sigma(s) for Gaussian smoothing, this is in physical units, not pixels.
        Return:
            Image which is a result of smoothing the input and then resampling it using the given sigma(s) and shrink factor(s).
        """
        if np.isscalar(shrink_factors):
            shrink_factors = [shrink_factors] * image.GetDimension()
        if np.isscalar(smoothing_sigmas):
            smoothing_sigmas = [smoothing_sigmas] * image.GetDimension()

        smoothed_image = sitk.SmoothingRecursiveGaussian(image, smoothing_sigmas)

        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        new_size = [
            int(sz / float(sf) + 0.5) for sf, sz in zip(shrink_factors, original_size)
        ]
        new_spacing = [
            ((original_sz - 1) * original_spc) / (new_sz - 1)
            for original_sz, original_spc, new_sz in zip(
                original_size, original_spacing, new_size
            )
        ]
        return sitk.Resample(
            smoothed_image,
            new_size,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            image.GetOrigin(),
            new_spacing,
            image.GetDirection(),
            0.0,
            image.GetPixelID(),
        )

    # out = smooth_and_resample(moving_image, shrink_factors=shrink_factors,
    #                           smoothing_sigmas=smoothing_sigmas)
    # displayImage(sitk.GetArrayFromImage(out), Title_='smoothed image')


    def multiscale_demons(
            registration_algorithm,
            fixed_image,
            moving_image,
            initial_transform=None,
            shrink_factors=None,
            smoothing_sigmas=None,
    ):
        """
        Run the given registration algorithm in a multiscale fashion. The original scale should not be given as input as the
        original images are implicitly incorporated as the base of the pyramid.
        Args:
            registration_algorithm: Any registration algorithm that has an Execute(fixed_image, moving_image, displacement_field_image)
                                    method.
            fixed_image: Resulting transformation maps points from this image's spatial domain to the moving image spatial domain.
            moving_image: Resulting transformation maps points from the fixed_image's spatial domain to this image's spatial domain.
            initial_transform: Any SimpleITK transform, used to initialize the displacement field.
            shrink_factors (list of lists or scalars): Shrink factors relative to the original image's size. When the list entry,
                                                       shrink_factors[i], is a scalar the same factor is applied to all axes.
                                                       When the list entry is a list, shrink_factors[i][j] is applied to axis j.
                                                       This allows us to specify different shrink factors per axis. This is useful
                                                       in the context of microscopy images where it is not uncommon to have
                                                       unbalanced sampling such as a 512x512x8 image. In this case we would only want to
                                                       sample in the x,y axes and leave the z axis as is: [[[8,8,1],[4,4,1],[2,2,1]].
            smoothing_sigmas (list of lists or scalars): Amount of smoothing which is done prior to resmapling the image using the given shrink factor. These
                              are in physical (image spacing) units.
        Returns:
            SimpleITK.DisplacementFieldTransform
        """

        # Create image pyramid in a memory efficient manner using a generator function.
        # The whole pyramid never exists in memory, each level is created when iterating over
        # the generator.
        def image_pair_generator(
                fixed_image, moving_image, shrink_factors, smoothing_sigmas
        ):
            end_level = 0
            start_level = 0
            if shrink_factors is not None:
                end_level = len(shrink_factors)
            for level in range(start_level, end_level):
                f_image = smooth_and_resample(
                    fixed_image, shrink_factors[level], smoothing_sigmas[level]
                )
                m_image = smooth_and_resample(
                    moving_image, shrink_factors[level], smoothing_sigmas[level]
                )
                yield (f_image, m_image)
            yield (fixed_image, moving_image)

        # Create initial displacement field at lowest resolution.
        # Currently, the pixel type is required to be sitkVectorFloat64 because
        # of a constraint imposed by the Demons filters.
        if shrink_factors is not None:
            original_size = fixed_image.GetSize()
            original_spacing = fixed_image.GetSpacing()
            s_factors = (
                [shrink_factors[0]] * len(original_size)
                if np.isscalar(shrink_factors[0])
                else shrink_factors[0]
            )
            df_size = [
                int(sz / float(sf) + 0.5) for sf, sz in zip(s_factors, original_size)
            ]
            df_spacing = [
                ((original_sz - 1) * original_spc) / (new_sz - 1)
                for original_sz, original_spc, new_sz in zip(
                    original_size, original_spacing, df_size
                )
            ]
        else:
            df_size = fixed_image.GetSize()
            df_spacing = fixed_image.GetSpacing()

        # if initial_transform:
        #     initial_displacement_field = sitk.TransformToDisplacementField(
        #         initial_transform,
        #         sitk.sitkVectorFloat64,
        #         df_size,
        #         fixed_image.GetOrigin(),
        #         df_spacing,
        #         fixed_image.GetDirection(),
        #     )
        # else:
        #     # initial_displacement_field = sitk.Image(
        #     #     df_size, sitk.sitkVectorFloat64, fixed_image.GetDimension()
        #     # )
        #     # initial_displacement_field.SetSpacing(df_spacing)
        #     # initial_displacement_field.SetOrigin(fixed_image.GetOrigin())

            # rana changed:
            # initial_displacement_field = sitk.TransformToDisplacementFieldFilter()
            # initial_displacement_field.SetReferenceImage(fixed_image)


        # Run the registration.
        # Start at the top of the pyramid and work our way down.
        for f_image, m_image in image_pair_generator(
                fixed_image, moving_image, shrink_factors, smoothing_sigmas
        ):
            # initial_displacement_field = sitk.Resample(initial_displacement_field, f_image)
            # initial_displacement_field = registration_algorithm.Execute(
            #     f_image, m_image, initial_displacement_field
            # )

            # rana changed
            initialTx = sitk.CenteredTransformInitializer(f_image, m_image,
                                                          sitk.AffineTransform(f_image.GetDimension()))
            transform_to_displacment_field_filter = sitk.TransformToDisplacementFieldFilter()
            transform_to_displacment_field_filter.SetReferenceImage(f_image)
            displacementTx = sitk.DisplacementFieldTransform(transform_to_displacment_field_filter.Execute
                                                             (#initialTx))
                                                              sitk.Transform(2, sitk.sitkIdentity)))
            displacementTx.SetSmoothingGaussianOnUpdate(varianceForUpdateField=0.0, varianceForTotalField=1.5)
            displacementField = transform_to_displacment_field_filter.Execute(displacementTx)
            displacementField = registration_algorithm.Execute(f_image, m_image, displacementField)
            outTx = sitk.DisplacementFieldTransform(displacementField)
            # break
        return outTx #sitk.DisplacementFieldTransform(initial_displacement_field)

    def iteration_callback(filter):
        global metric_values
        print(f"\r{filter.GetElapsedIterations()}: {filter.GetMetric():.2f}", end="")
        metric_values.append(filter.GetMetric())
    metric_values = []
    # Select a Demons filter and configure it.
    demons_filter = sitk.FastSymmetricForcesDemonsRegistrationFilter()
    demons_filter.SetNumberOfIterations(2000)
    # Regularization (update field - viscous, total field - elastic).
    demons_filter.SetSmoothDisplacementField(True)
    demons_filter.SetStandardDeviations(1.2)

    # Add our simple callback to the registration filter.
    demons_filter.AddCommand(
        sitk.sitkIterationEvent, lambda: iteration_callback(demons_filter)
    )

    # Run the registration.
    tx = multiscale_demons(
        registration_algorithm=demons_filter,
        fixed_image=fixed_image,
        moving_image=moving_image,
        shrink_factors=[4, 2, 1],
        smoothing_sigmas=[8, 4, 2],
    )

    # # Compare the initial and final TREs.
    # (
    #     initial_errors_mean,
    #     initial_errors_std,
    #     _,
    #     initial_errors_max,
    #     initial_errors,
    # ) = ru.registration_errors(
    #     sitk.Euler2DTransform(), points[fixed_image_index], points[moving_image_index]
    # )
    # (
    #     final_errors_mean,
    #     final_errors_std,
    #     _,
    #     final_errors_max,
    #     final_errors,
    # ) = ru.registration_errors(tx, points[fixed_image_index], points[moving_image_index])
    #
    # plt.hist(initial_errors, bins=20, alpha=0.5, label="before registration", color="blue")
    # plt.hist(final_errors, bins=20, alpha=0.5, label="after registration", color="green")
    # plt.legend()
    # plt.title("TRE histogram")
    # print(
    #     f"\nInitial alignment errors in millimeters, mean(std): {initial_errors_mean:.2f}({initial_errors_std:.2f}), max: {initial_errors_max:.2f}"
    # )
    # print(
    #     f"Final alignment errors in millimeters, mean(std): {final_errors_mean:.2f}({final_errors_std:.2f}), max: {final_errors_max:.2f}"
    # )
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(tx)
    out = resampler.Execute(moving_image)

    displayImage(sitk.GetArrayFromImage(out), Title_='registered - multiresolution demon')
    displayImage(sitk.GetArrayFromImage(moving_image), Title_='label map')
    # displayImage(sitk.GetArrayFromImage(fixed_image))
    checkerImg = compare_images(sitk.GetArrayFromImage(out),
                                sitk.GetArrayFromImage(fixed_image),
                                method='diff')
    displayImage(checkerImg, Title_='multiresolution Diffeomorphic Demon')
    plt.plot(metric_values)
    plt.show()

# +-------------------------------------------+
# |    Pyramid scheme!                        |
# |   the registration performance of         |
# +-------------------------------------------+
if __name__ != '__main__':
    def command_iteration2(method):
        global metric_values
        # print(f"{filter.GetElapsedIterations():3} = {filter.GetMetric():10.5f}")
        print("metric value: {}".format(method.GetMetricValue()))
        metric_values.append(method.GetMetricValue())

    # def demons_registration(fixed_image, moving_image, fixed_points=None, moving_points=None):
        # global metric_values
    metric_values = []
    registration_method = sitk.ImageRegistrationMethod()

    # Create initial identity transformation.
    transform_to_displacment_field_filter = sitk.TransformToDisplacementFieldFilter()
    transform_to_displacment_field_filter.SetReferenceImage(fixed_image)
    # The image returned from the initial_transform_filter is transferred to the transform and cleared out.
    initial_transform = sitk.DisplacementFieldTransform(
        transform_to_displacment_field_filter.Execute(sitk.Transform(2, sitk.sitkIdentity)))

    # Regularization (update field - viscous, total field - elastic).
    initial_transform.SetSmoothingGaussianOnUpdate(
        varianceForUpdateField=0.0, varianceForTotalField=2.0
    )

    registration_method.SetInitialTransform(initial_transform)

    registration_method.SetMetricAsDemons(1)  # intensities are equal if the difference is less than 10HU

    # Multi-resolution framework.
    # registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    # registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[8, 4, 0])

    registration_method.SetInterpolator(sitk.sitkNearestNeighbor)
    # If you have time, run this code as is, otherwise switch to the gradient descent optimizer
    # registration_method.SetOptimizerAsConjugateGradientLineSearch(learningRate=1.0, numberOfIterations=20, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=200,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    # If corresponding points in the fixed and moving image are given then we display the similarity metric
    # and the TRE during the registration.
    # if fixed_points and moving_points:
    #     registration_method.AddCommand(
    #         sitk.sitkStartEvent, rc.metric_and_reference_start_plot
    #     )
    #     registration_method.AddCommand(
    #         sitk.sitkEndEvent, rc.metric_and_reference_end_plot
    #     )
    #     registration_method.AddCommand(
    #         sitk.sitkIterationEvent,
    #         lambda: rc.metric_and_reference_plot_values(
    #             registration_method, fixed_points, moving_points
    #         ),
    #     )
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration2(registration_method))
    tx = registration_method.Execute(fixed_image, moving_image)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(tx)
    out = resampler.Execute(moving_image)

    displayImage(sitk.GetArrayFromImage(out), Title_='registered - displacement')
    displayImage(sitk.GetArrayFromImage(moving_image), Title_='label map')
    # displayImage(sitk.GetArrayFromImage(fixed_image))
    checkerImg = compare_images(sitk.GetArrayFromImage(out),
                                sitk.GetArrayFromImage(fixed_image),
                                method='diff')
    displayImage(checkerImg, Title_='displacementfield with demon as metric')
        # return tx, metric_values

    # metric_values = []
    # tx, metric_values = demons_registration(
    #     fixed_image=fixed_image,
    #     moving_image=moving_image,
    #     fixed_points=None,
    #     moving_points=None)
    plt.plot(metric_values)
    plt.show()

# +-----------------------------------------------+
# |    Displacement field transform               |
# |    metric works but registration not good     |
# |    1. How to fix the number of iterations ?   |
# +-----------------------------------------------+
if __name__ != '__main__':
    metric_values = []
    initial_transform = sitk.CenteredTransformInitializer(fixed_image, moving_image, sitk.AffineTransform(fixed_image.GetDimension()))
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=500)
    registration_method.SetOptimizerAsGradientDescent(learningRate=15,
                                                      numberOfIterations=3000,
                                                      estimateLearningRate=registration_method.EachIteration,
                                                     )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetInitialTransform(initial_transform)
    registration_method.SetInterpolator(sitk.sitkNearestNeighbor)
    outTx1 = registration_method.Execute(fixed_image, moving_image)
    displacementField = sitk.TransformToDisplacementFieldFilter()
    displacementField.SetReferenceImage(fixed_image)
    displacementTx = sitk.DisplacementFieldTransform(displacementField.Execute(sitk.Transform(2, sitk.sitkIdentity)))
    displacementTx.SetSmoothingGaussianOnUpdate(varianceForUpdateField=0.0, varianceForTotalField=1.5)

    registration_method.SetMovingInitialTransform(outTx1)
    registration_method.SetInitialTransform(displacementTx, inPlace=True)
    registration_method.SetMetricAsANTSNeighborhoodCorrelation(40)
    registration_method.MetricUseFixedImageGradientFilterOff()
    # if registration_method.GetOptimizerIteration() == 31:
        # metric_v = []
    # metric_v.append(registration_method.GetMetricValue())
    # registration_method.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(registration_method))
    # registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
    # registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration2(registration_method))
    registration_method.Execute(fixed_image, moving_image)
    compositeTx = sitk.CompositeTransform([outTx1, displacementTx])
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(compositeTx)
    out = resampler.Execute(moving_image)

    displayImage(sitk.GetArrayFromImage(out), Title_='registered(Affine+Disp) ')
    displayImage(sitk.GetArrayFromImage(moving_image), Title_='label map')
    # displayImage(sitk.GetArrayFromImage(fixed_image))
    checkerImg = compare_images(sitk.GetArrayFromImage(out),
                                sitk.GetArrayFromImage(fixed_image),
                                method='diff')
    displayImage(checkerImg, Title_='final registration, fixed-moving overlapped')
    plt.plot(metric_values)
    plt.show()

# +------------------------------------------+
# |    free from deformation registration    |
# +------------------------------------------+
if __name__ != '__main__':
    def command_iteration2(method):
        global metric_values
        # print(f"{filter.GetElapsedIterations():3} = {filter.GetMetric():10.5f}")
    # def command_iteration_(filter):
    #     global metric_values
    # metric_values.append(method.GetMetricValue())
    metric_values = []
    for i in range(1):
        registration_method = sitk.ImageRegistrationMethod()
        grid_physical_spacing = [50.0, 50.0, 50.0]
        image_physical_size = [size * spacing for size, spacing in zip(fixed_image.GetSize(), fixed_image.GetSpacing())]
        # print("image physical size: ", image_physical_size)
        mesh_size = [int(image_size / grid_spacing + 0.5) for image_size, grid_spacing in zip(image_physical_size, grid_physical_spacing)]
        # print("mesh size: ", mesh_size)
        initial_transform = sitk.BSplineTransformInitializer(image1=fixed_image,
                                                             transformDomainMeshSize=mesh_size, order=3)
        registration_method.SetInitialTransform(initial_transform)
        registration_method.SetMetricAsMeanSquares()
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.01)
        # fixed_image_mask = None
        # if fixed_image_mask:
        registration_method.SetMetricFixedMask(fixed_image)

        # Multi-resolution framework.
        # registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        # registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        registration_method.SetInterpolator(sitk.sitkLinear)
        registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-7, numberOfIterations=1000)

        # If corresponding points in the fixed and moving image are given then we display the similarity metric
        # and the TRE during the registration.
        # if fixed_points and moving_points:
        #     registration_method.AddCommand(sitk.sitkStartEvent, rc.metric_and_reference_start_plot)
        #     registration_method.AddCommand(sitk.sitkEndEvent, rc.metric_and_reference_end_plot)
        #     registration_method.AddCommand(sitk.sitkIterationEvent,
        #                                    lambda: rc.metric_and_reference_plot_values(registration_method, fixed_points,
        #                                                                                moving_points))
        tx = registration_method.Execute(fixed_image, moving_image)
        # tx = bspline_intra_modal_registration(fixed_image=images[fixed_image_index],
        #                                       moving_image=images[moving_image_index],
        #                                       fixed_image_mask=(masks[fixed_image_index] == lung_label),
        #                                       fixed_points=points[fixed_image_index],
        #                                       moving_points=points[moving_image_index]
        #                                       )
        # initial_errors_mean, initial_errors_std, _, initial_errors_max, initial_errors = ru.registration_errors(
        #     sitk.Euler3DTransform(), points[fixed_image_index], points[moving_image_index])
        # final_errors_mean, final_errors_std, _, final_errors_max, final_errors = ru.registration_errors(tx, points[
        #     fixed_image_index], points[moving_image_index])

        # plt.hist(initial_errors, bins=20, alpha=0.5, label='before registration', color='blue')
        # plt.hist(final_errors, bins=20, alpha=0.5, label='after registration', color='green')
        # plt.legend()
        # plt.title('TRE histogram');
        # print('Initial alignment errors in millimeters, mean(std): {:.2f}({:.2f}), max: {:.2f}'.format(initial_errors_mean,
        #                                                                                                initial_errors_std,
        #                                                                                                initial_errors_max))
        # print('Final alignment errors in millimeters, mean(std): {:.2f}({:.2f}), max: {:.2f}'.format(final_errors_mean,
        #                                                                                              final_errors_std,
        #                                                                                              final_errors_max))
        # Transfer the segmentation via the estimated transformation. Use Nearest Neighbor interpolation to retain the labels.
        # transformed_labels = sitk.Resample(masks[moving_image_index],
        #                                    images[fixed_image_index],
        #                                    tx,
        #                                    sitk.sitkNearestNeighbor,
        #                                    0.0,
        #                                    masks[moving_image_index].GetPixelID())
        registration_method.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration2(registration_method))
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_image)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(tx)
        resampler.SetOutputPixelType(moving_image.GetPixelID())
        out = resampler.Execute(moving_image)
        # moving_image = out
        metric_values.append(registration_method.GetMetricValue())
    segmentations_before_and_after = [fixed_image, out]
    # interact(display_coronal_with_label_maps_overlay, coronal_slice = (0, images[0].GetSize()[1]-1),
    #          mask_index=(0,len(segmentations_before_and_after)-1),
    #          image = fixed(images[fixed_image_index]), masks = fixed(segmentations_before_and_after),
    #          label=fixed(lung_label), window_min = fixed(-1024), window_max=fixed(976));

    # Compute the Dice coefficient and Hausdorff distance between the segmentations before, and after registration.
    # ground_truth = masks[fixed_image_index] == lung_label
    # before_registration = masks[moving_image_index] == lung_label
    # after_registration = transformed_labels == lung_label

    # label_overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    # label_overlap_measures_filter.Execute(ground_truth, before_registration)
    # print("Dice coefficient before registration: {:.2f}".format(label_overlap_measures_filter.GetDiceCoefficient()))
    # label_overlap_measures_filter.Execute(ground_truth, after_registration)
    # print("Dice coefficient after registration: {:.2f}".format(label_overlap_measures_filter.GetDiceCoefficient()))
    #
    # hausdorff_distance_image_filter = sitk.HausdorffDistanceImageFilter()
    # hausdorff_distance_image_filter.Execute(ground_truth, before_registration)
    # print("Hausdorff distance before registration: {:.2f}".format(hausdorff_distance_image_filter.GetHausdorffDistance()))
    # hausdorff_distance_image_filter.Execute(ground_truth, after_registration)
    # print("Hausdorff distance after registration: {:.2f}".format(hausdorff_distance_image_filter.GetHausdorffDistance()))
    displayImage(sitk.GetArrayFromImage(out), Title_='registered(FFD) ')
    displayImage(sitk.GetArrayFromImage(moving_image), Title_='label map')
    # displayImage(sitk.GetArrayFromImage(fixed_image))
    checkerImg = compare_images(sitk.GetArrayFromImage(out),
                                sitk.GetArrayFromImage(fixed_image),
                                method='diff')
    displayImage(checkerImg, Title_='final registration, fixed-moving overlapped')
    plt.plot(metric_values)
    plt.show()

# +--------------------------------+
# |     BSpline method intro       |
# +--------------------------------+
if __name__ != '__main__':
    metric_values = []
    grid_physical_spacing = [50.0, 50.0, 50.0]  # A control point every 50mm
    image_physical_size = [size * spacing for size, spacing in zip(fixed_image.GetSize(), fixed_image.GetSpacing())]
    # mesh_size = [int(image_size / grid_spacing + 0.5)\
    #              for image_size, grid_spacing in zip(image_physical_size, grid_physical_spacing)]
    # print("image physical size: ", image_physical_size)
    # print("mesh size: ", mesh_size)
    # mesh_size = [int(sz / 4 + 0.5) for sz in mesh_size]
    # print("mesh size: ", mesh_size)
    mesh_size = [2] * moving_image.GetDimension()
    initial_transform = sitk.BSplineTransformInitializer(image1=fixed_image,
                                                         transformDomainMeshSize=mesh_size,
                                                         order=3)
    # tx = sitk.BSplineTransformInitializer(fixed_image, transformDomainMeshSize)
    #
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsJointHistogramMutualInformation()    #SetMetricAsCorrelation()
    # # registration_method.SetOptimizerAsLBFGSB(
    # #     gradientConvergenceTolerance=1e-5,
    # #     numberOfIterations=100,
    # #     maximumNumberOfCorrections=50,
    # #     maximumNumberOfFunctionEvaluations=1000,
    # #     costFunctionConvergenceFactor=1e7)
    # registration_method.SetOptimizerAsGradientDescentLineSearch(
    #     5.0, 100,
    #     convergenceMinimumValue=1e-4,
    #     convergenceWindowSize=50
    # )
    # # registration_method.SetInitialTransform(tx, True)
    registration_method.SetInitialTransformAsBSpline(initial_transform,
                                   inPlace=False,
                                   scaleFactors=[1, 2, 4])
    # registration_method.SetMetricAsMeanSquares()
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    # registration_method.SetMetricFixedMask(fixed_image)
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 1])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration_method.SetInterpolator(sitk.sitkNearestNeighbor)
    registration_method.SetOptimizerAsLBFGS2(solutionAccuracy=1e-6, numberOfIterations=100,
                                             deltaConvergenceTolerance=0.01)
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration2(registration_method))
    outTx = registration_method.Execute(fixed_image, moving_image)
    # resampler = sitk.ResampleImageFilter()
    # resampler.SetReferenceImage(fixed_image)
    # resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    # resampler.SetDefaultPixelValue(0)
    # resampler.SetTransform(outTx)
    # out = resampler.Execute(moving_image)
    # displayImage(sitk.GetArrayFromImage(out), Title_='Euler + BSplineTransformInitializer')
    # displayImage(sitk.GetArrayFromImage(moving_image), Title_='moving image')
    # displayImage(sitk.GetArrayFromImage(fixed_image), Title_='fixed image')
    # checkerImg = compare_images(sitk.GetArrayFromImage(out),
    #                             sitk.GetArrayFromImage(fixed_image),
    #                             method='diff')
    # displayImage(checkerImg, Title_='Euler + BSplineTransformInitializer')
    #
    plt.plot(metric_values)
    plt.xlabel("Iterations")
    plt.ylabel("Metric value")
    plt.show()
# +------------------------------------------------------+
# | 1. implement and check BSpline registration method   |
# | 2. check affine also                                 |
# | 3. once best method finalized run all 4 sections     |
# | 4. run registration with Chase/MRI images            |
# |  |__ 4.1. create 3D spinal cord model                |
# |  |__ 4.2. incorporate double VAE segmentations       |
# +------------------------------------------------------+
if __name__ != '__main__':
    metric_values = []
    initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                          moving_image,
                                                          sitk.AffineTransform(fixed_image.GetDimension()),
                                                          # sitk.Euler2DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.MOMENTS)

    registration_method = sitk.ImageRegistrationMethod()
    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=100)
    # registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    # registration_method.SetMetricSamplingPercentage(0.01)
    tsScale = 1.0/100.0
    # Keeping in mind that the scale of units in scaling, rotation and translation are quite different,
    # we take advantage of the scaling functionality provided by the optimizers. We know that the first N ×N elements
    # of the parameters array correspond to the rotation matrix factor, and the last N are the components of the translation
    # to be applied after multiplication with the matrix is performed.
    registration_method.SetOptimizerScales([1, 1, 1, 1, tsScale])
    registration_method.SetInterpolator(sitk.sitkNearestNeighbor)
    # affine = sitk.AffineTransform()
    # affine.Execute()
    # Optimizer settings.
    registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=0.0001, numberOfIterations=100)
    # registration_method.Set
    # registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100,
    #                                                   convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    # registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    # registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    # registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    # registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()



    # Connect all of the observers so that we can perform plotting during registration.
    # registration_method.AddCommand(sitk.sitkStartEvent, rgui.start_plot)
    # registration_method.AddCommand(sitk.sitkEndEvent, rgui.end_plot)
    # registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, rgui.update_multires_iterations)
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration2(registration_method))
    final_transform = registration_method.Execute(fixed_image, moving_image)
    plt.plot(metric_values)
    plt.show()
    print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))

if __name__ != '__main__':
    orientations_list = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
    print(orientations_list)

    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image,
        moving_image,
        sitk.AffineTransform(2),  # AffineTransform(2) or AffineTransform()?
        sitk.CenteredTransformInitializerFilter.GEOMETRY)
    # def evaluate_metric(current_rotation, tx, f_image, m_image):
    #     print(current_rotation)
    current_rotation = orientations_list[0]
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetInterpolator(sitk.sitkNearestNeighbor)
    current_transform = sitk.AffineTransform(initial_transform)
        # from sitk.AffineTransform: Rotate (int axis1, int axis2, double angle, bool pre=false)
    current_transform.Rotate(axis1=0, axis2=1, angle=current_rotation)
    registration_method.SetInitialTransform(current_transform)
    final_transform = registration_method.MetricEvaluate(fixed_image, moving_image)
    # final_transform = registration_method.Execute(fixed_image, moving_image)
    # return final_transform

    from multiprocessing.pool import ThreadPool
    from functools import partial

    print("<> ", final_transform)
    # print("center >>", fixed_image.GetSize(),  initial_transform.GetCenter())
    # 2D orientations
    orientations_list = [np.pi / 2, np.pi, 0, 3 * np.pi / 2]
    # p = ThreadPool(len(orientations_list))
    # all_metric_values = p.map(partial(evaluate_metric,
    #                                   tx=initial_transform,
    #                                   f_image=fixed_image,
    #                                   m_image=moving_image),
    #                           orientations_list)
    # print(all_metric_values)
    # best_orientation = orientations_list[np.argmin(all_metric_values)]
    # print('best orientation: {}/'.format(best_orientation))
    # final_transform = evaluate_metric(orientations_list[0], initial_transform, fixed_image, moving_image)
    # out = sitk.Resample(moving_image, fixed_image, final_transform)
    # resampler = sitk.ResampleImageFilter()
    # resampler.SetReferenceImage(fixed_image)
    # resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    # resampler.SetDefaultPixelValue(0)
    # resampler.SetTransform(final_transform)
    # out = resampler.Execute(moving_image)
    # displayImage(sitk.GetArrayFromImage(out), Title_='final transform')

if __name__ != '__main__':
    metric_values = []
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=0.01, numberOfIterations=100)
    # registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100,
    #                                                   convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    # registration_method.SetOp
    registration_method.SetInitialTransform(sitk.AffineTransform(fixed_image.GetDimension()))
    registration_method.SetInterpolator(sitk.sitkNearestNeighbor)
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration2(registration_method))
    outTx = registration_method.Execute(fixed_image, moving_image)
    plt.plot(metric_values)
    plt.show()
    print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))

# +-----------------------------+
# |   experiment with txforms   |
# +-----------------------------+
if __name__ != '__main__':
    # dimension = 2
    # point = (1.0, 1.0)
    # affine = sitk.AffineTransform(dimension)
    # print('Parameters: ' + str(affine.GetParameters()))
    # print('FixedParameters: ' + str(affine.GetFixedParameters()))
    # # transform_point(affine, point)
    # transformed_point = affine.TransformPoint(point)
    # print('Point ' + str(point) + ' transformed is ' + str(transformed_point))
    # affine.SetTranslation((3.1, 4.4))
    # print('Parameters: ' + str(affine.GetParameters()))
    # # transform_point(affine, point)
    # transformed_point = affine.TransformPoint(point)
    # print('Point ' + str(point) + ' transformed is ' + str(transformed_point))

    def multires_registration(fixed_image, moving_image, initial_transform):
        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetInterpolator(sitk.sitkNearestNeighbor)
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.01)
        registration_method.SetOptimizerAsGradientDescent(learningRate=1.5, numberOfIterations=100,
                                                          convergenceMinimumValue=1e-6, convergenceWindowSize=50
                                                          )
        registration_method.SetOptimizerScalesFromPhysicalShift()
        registration_method.SetInitialTransform(initial_transform, inPlace=True)
        # registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        # registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        registration_method.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration2(registration_method))
        final_transform = registration_method.Execute(fixed_image, moving_image)

        print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
        print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
        return final_transform
    # Read the images and modify so that they are 2D, the original size is [x,y,1] and we want [x,y].
    # fixed_image = sitk.ReadImage('cxr.dcm', sitk.sitkFloat32)[:, :, 0]
    # moving_image = sitk.ReadImage('photo.dcm', sitk.sitkFloat32)[:, :, 0]
    # Display the original images and resamples moving_image (onto the
    # fixed_image grid using the identity transformation)
    # sitk.Show(fixed_image, 'fixed')
    # sitk.Show(moving_image, 'moving')
    # out = sitk.Resample(moving_image, fixed_image, sitk.Transform())
    # displayImage(sitk.GetArrayFromImage(out), Title_='identity transform')
    # Centered 2D affine transform and show the resampled moving_image using this transform.
    initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                               moving_image,
                                                               sitk.AffineTransform(2),
                                                               sitk.CenteredTransformInitializerFilter.GEOMETRY)
    out = sitk.Resample(moving_image, fixed_image, initial_transform)
    displayImage(sitk.GetArrayFromImage(out), Title_='initial affine transform')
    print(initial_transform)
    # Register using 2D affine initial transform that is overwritten
    # and show the resampled moving_image using this transform.
    metric_values = []
    final_transform = sitk.Euler2DTransform(initial_transform) #multires_registration(fixed_image, out, initial_transform))
    print(final_transform)
    out = sitk.Resample(image1=out,
                        referenceImage=fixed_image,
                        transform=final_transform)
    displayImage(sitk.GetArrayFromImage(out), Title_='final affine transform')
    # plt.plot(metric_values, 'r')
    # plt.xlabel('Iteration Number', fontsize=12)
    # plt.ylabel('Metric Value', fontsize=12)
    # plt.show()

# +-----------------------------+
# |   affine registration ...   |
# +-----------------------------+
if __name__ != '__main__':
    metric_values = []
    initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                          moving_image,
                                                          sitk.AffineTransform(2),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetInterpolator(sitk.sitkNearestNeighbor)
    registration_method.SetInitialTransform(initial_transform, inPlace=True)
    registration_method.SetMetricSamplingStrategy(registration_method.NONE)
    # registration_method.SetOptimizerAsGradientDescent(lear)
    translationScale = 1.0 / 100.0
    registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=1e-4,
                                                                 numberOfIterations=100)
    registration_method.SetOptimizerScales(scales=[1.0, 1.0, 1.0, 1.0, translationScale, translationScale])
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration2(registration_method))
    final_transform = registration_method.Execute(fixed_image, moving_image)
    plt.plot(metric_values)
    plt.show()
    print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
    print("here")

if __name__ != '__main__':
    print("here")
    degrees = np.linspace(0, 360, 1000)
    radians = np.deg2rad(degrees)

    initial_transform = sitk.Euler2DTransform(sitk.CenteredTransformInitializer(
        fixed_image,
        moving_image,
        sitk.Euler2DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY))
    # print("Start of registration: \n", initial_transform)
    # Registration framework setup.
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=500)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.001)
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

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
    for i, angle in enumerate(radians):  # .items():
        initial_transform.SetAngle(angle)
        registration_method.SetInitialTransform(initial_transform)
        current_similarity_value = registration_method.MetricEvaluate(fixed_image, moving_image)
        similarity_value.append(current_similarity_value)
        # print("current similarity value: ", current_similarity_value)
        if current_similarity_value < best_similarity_value:
            best_similarity_value = current_similarity_value
            best_angle = np.rad2deg(angle)
        # else:
        #     best_orientation = orientation
    print('best orientation is: ' + str(best_angle))
    print("best similarity value", best_similarity_value)

    plt.plot(degrees, similarity_value, 'b')
    plt.plot(best_angle, best_similarity_value, 'r^')
    plt.show()

    initial_transform.SetAngle(np.deg2rad(best_angle))
    registration_method.SetInitialTransform(initial_transform, inPlace=True)
    eulerTx = registration_method.Execute(fixed_image, moving_image)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(eulerTx)

    out = resampler.Execute(moving_image)
    displayImage(sitk.GetArrayFromImage(out), Title_='registered image(Euler2D)')
    displayImage(sitk.GetArrayFromImage(moving_image), Title_='moving image')
    displayImage(sitk.GetArrayFromImage(fixed_image), Title_='fixed image')
    # print("End of registration: \n", initial_transform)

    # Affine Transform now:
    metric_values = []
    affine = sitk.AffineTransform(2)
    # print(initial_transform.GetMatrix())
    # print(initial_transform.GetTranslation())
    # print(initial_transform.GetCenter())
    affine.SetMatrix(initial_transform.GetMatrix())
    affine.SetTranslation(initial_transform.GetTranslation())
    affine.SetCenter(initial_transform.GetCenter())
    registration_method.SetInitialTransform(affine, inPlace=True)
    # print("1")
    print("before affine: \n", affine)
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration2(registration_method))
    affine_tx = registration_method.Execute(fixed_image, moving_image)
    # print("2")
    # resampler = sitk.ResampleImageFilter()
    # resampler.SetReferenceImage(fixed_image)
    # resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    # resampler.SetDefaultPixelValue(0)
    # print(affine_tx)
    # print("3")
    resampler.SetTransform(affine_tx)
    # resampler.SetOutputPixelType(moving_image.GetPixelID())
    out = resampler.Execute(moving_image)
    print("after affine \n", affine_tx)
    displayImage(sitk.GetArrayFromImage(out), Title_='registered image(Affine)')
    plt.plot(metric_values)
    plt.show()

# todo: `Euler + Affine + FSFDemon` works nicely but ...
#       1. Run same method(finalized) for all sections
#          1.1. Morphological output of VAE must be curated before registration
#       2. Parameters could be optimized for better results
#          2.1. How ?
#       3. Plotting issues of affine metrics


if __name__ == '__main__':
    def command_iteration2(method):
        global metric_values
        print("metric value: {}".format(method.GetMetricValue()))
        metric_values.append(method.GetMetricValue())

    def command_iteration(filter):
        global metric_values
        print(f"{filter.GetElapsedIterations():3} = {filter.GetMetric():10.5f}")
        metric_values.append(filter.GetMetric())

    metric_values = []
    degrees = np.linspace(0, 360, 1000)
    radians = np.deg2rad(degrees)

    initial_transform = sitk.Euler2DTransform(sitk.CenteredTransformInitializer(
        fixed_image,
        moving_image,
        sitk.Euler2DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY))
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=500)
    registration_method.SetMetricSamplingStrategy(registration_method.NONE)
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=0.50,
                                                                 minStep=1e-3,
                                                                 numberOfIterations=500,
                                                                 gradientMagnitudeTolerance=1e-2)
    registration_method.SetInterpolator(sitk.sitkNearestNeighbor)
    registration_method.SetOptimizerScalesFromIndexShift()
    best_angle = 0.0
    best_similarity_value = registration_method.MetricEvaluate(fixed_image, moving_image)
    similarity_value = []
    for i, angle in enumerate(radians):  # .items():
        initial_transform.SetAngle(angle)
        registration_method.SetInitialTransform(initial_transform)
        current_similarity_value = registration_method.MetricEvaluate(fixed_image, moving_image)
        similarity_value.append(current_similarity_value)
        if current_similarity_value < best_similarity_value:
            best_similarity_value = current_similarity_value
            best_angle = np.rad2deg(angle)
    print('best orientation is: ' + str(best_angle))
    print("best similarity value", best_similarity_value)
    plt.plot(degrees, similarity_value, 'b')
    plt.plot(best_angle, best_similarity_value, 'r^')
    plt.show()
    initial_transform.SetAngle(np.deg2rad(best_angle))
    registration_method.SetInitialTransform(initial_transform, inPlace=True)
    eulerTx = registration_method.Execute(fixed_image, moving_image)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(eulerTx)
    out = resampler.Execute(moving_image)
    displayImage(sitk.GetArrayFromImage(fixed_image), Title_='fixed image')
    displayImage(sitk.GetArrayFromImage(moving_image), Title_='moving image')
    displayImage(sitk.GetArrayFromImage(out), Title_='Euler2D image')
    affine = sitk.AffineTransform(fixed_image.GetDimension())
    # affine = sitk.CenteredTransformInitializer(fixed_image,
    #                                            moving_image,
    #                                            sitk.AffineTransform(fixed_image.GetDimension()),
    #                                            # sitk.CenteredTransformInitializerFilter.GEOMETRY
    #                                            )
    # affine.SetMatrix(initial_transform.GetMatrix())
    # affine.SetTranslation(initial_transform.GetTranslation())
    # affine.SetCenter(initial_transform.GetCenter())
    registration_method.SetMovingInitialTransform(eulerTx)
    registration_method.SetInitialTransform(affine, inPlace=True)
    registration_method.SetMetricAsANTSNeighborhoodCorrelation(4)
    # registration_method.SetOptimizerScalesFromIndexShift()
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=.50,
        numberOfIterations=300,
        estimateLearningRate=registration_method.EachIteration,
    )
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration2(registration_method))
    affineTx = registration_method.Execute(fixed_image, moving_image)
    compositeTx = sitk.CompositeTransform([eulerTx, affineTx])
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(compositeTx)
    out = resampler.Execute(moving_image)
    displayImage(sitk.GetArrayFromImage(out), Title_='Euler2D + Affine image')
    # plt.plot(metric_values)
    # plt.show()
    checkerImg = compare_images(sitk.GetArrayFromImage(out),
                                sitk.GetArrayFromImage(fixed_image),
                                method='diff')
    displayImage(checkerImg, Title_='Difference')

    # deformable registration
    metric_values = []
    demons = sitk.FastSymmetricForcesDemonsRegistrationFilter()
    demons.SetNumberOfIterations(5000)
    demons.SetStandardDeviations(1.2)
    demons.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(demons))
    # # metric_values.append()
    transform_to_displacment_field_filter = sitk.TransformToDisplacementFieldFilter()
    transform_to_displacment_field_filter.SetReferenceImage(fixed_image)
    displacementTx = sitk.DisplacementFieldTransform(transform_to_displacment_field_filter.Execute
                                                     (sitk.Transform(fixed_image.GetDimension(),
                                                                     sitk.sitkIdentity)))
    displacementTx.SetSmoothingGaussianOnUpdate(varianceForUpdateField=0.0, varianceForTotalField=1.5)
    displacementField = transform_to_displacment_field_filter.Execute(displacementTx)
    displacementField = demons.Execute(fixed_image, out, displacementField)
    outTx = sitk.DisplacementFieldTransform(displacementField)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(outTx)
    out = resampler.Execute(out)
    displayImage(sitk.GetArrayFromImage(out), Title_='Euler + Affine + FSFDemonsRegistrationFilter')
    # displayImage(sitk.GetArrayFromImage(moving_image), Title_='moving image')
    # displayImage(sitk.GetArrayFromImage(fixed_image), Title_='fixed image')
    checkerImg = compare_images(sitk.GetArrayFromImage(out),
                                sitk.GetArrayFromImage(fixed_image),
                                method='diff')
    displayImage(checkerImg, Title_='Difference 2')
    plt.plot(metric_values)
    plt.ylabel("Registration metric")
    plt.xlabel("Iterations")
    plt.show()



