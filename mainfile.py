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

def command_iteration(method):
    global metric_values
    print("is this running? ")
    if method.GetOptimizerIteration() == 31:
        # metric_values = []
        print(f"\tLevel: {method.GetCurrentLevel()}")
        print(f"\tScales: {method.GetOptimizerScales()}")
    print(f"#{method.GetOptimizerIteration()}")
    print(f"\tMetric Value: {method.GetMetricValue():10.5f}")
    print(f"\tLearningRate: {method.GetOptimizerLearningRate():10.5f}")
    if method.GetOptimizerConvergenceValue() != sys.float_info.max:
        print(
            "\t Convergence Value: "
            + f"{method.GetOptimizerConvergenceValue():.5e}"
        )
    metric_values.append(method.GetMetricValue())
    # # print(f"{method.():3} = {method.GetMetricValue():10.5f}")
    plt.plot(metric_values, 'r')
    # # plt.plot(multires_iterations, [metric_values[index] for index in multires_iterations], 'b*')
    plt.xlabel('Iteration Number', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    plt.show()
    # return metric_values

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
    reconPath = os.path.join(fileDir, 'seg_3.h5')   # change here for different sections.
    with h5py.File(reconPath, 'r') as pfile:     # saves the data
        labelMapMALDI = np.array(pfile['segmentation_3'])

    # plt.imshow(segImg==2)
    # segImg[segImg==2]= 0
    labelMapMALDI_t = labelMapMALDI.T   # aligns
    labelMapMALDI_t[labelMapMALDI_t == 2] = 0
    labelMapMALDI_t[labelMapMALDI_t == 1] = 2  # changing WM to match mask
    labelMapMALDI_t[labelMapMALDI_t == 4] = 1
    # segImg[5:7, 65:67] = 3
    displayImage(labelMapMALDI_t, Title_='After alignment and label correction')

    maskpath = glob(os.path.join(fileDir, 'cropped_hne_mask_3.nii.gz'))[0]
    maskHnE = nib.load(maskpath).get_fdata()
    maskHnE_0 = maskHnE[..., 0]
    displayImage(maskHnE_0, Title_='before resize')
    print(np.unique(maskHnE_0))
    maskHnE_0_resized = resize(maskHnE_0,
                               labelMapMALDI_t.shape,
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

    mask_itk = sitk.GetImageFromArray(maskHnE_0_resized)
    label_itk = sitk.GetImageFromArray(labelMapMALDI_t)
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
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
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
    del registration_method, initial_transform, eulerTx
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
    def command_iteration2(method):
        global metric_values
        # print(f"{filter.GetElapsedIterations():3} = {filter.GetMetric():10.5f}")
    # def command_iteration_(filter):
    #     global metric_values
        metric_values.append(method.GetMetricValue())
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
# if __name__ == '__main__':
# todo:
# +------------------------------------------------------+
# | 1. implement and check BSpline registration method   |
# | 2. check affine also                                 |
# | 3. once best method finalized run all 4 sections     |
# | 4. run registration with Chase/MRI images            |
# |  |__ 4.1. create 3D spinal cord model                |
# |  |__ 4.2. incorporate double VAE segmentations       |
# +------------------------------------------------------+
