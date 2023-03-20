import os
import numpy as np
import h5py
from scipy import ndimage
from skimage.color import label2rgb
from skimage.transform import resize
from skimage.util import compare_images
from skimage.measure import regionprops, label
from skimage.morphology import area_closing
from skimage import filters
from skimage.exposure import rescale_intensity, adjust_gamma, equalize_hist, equalize_adapthist
from utils import displayImage, displayMR, multi_dil, bbox2 #, command_iteration#, command_iteration2
from utils import smooth_and_resample, image_pair_generator, multiscale_demons_filter, multiscale_demons, nnPixelCorrect
import nibabel as nib
import SimpleITK as sitk
import matplotlib.pyplot as plt
import random
random.seed(1001)
# proteinDir = r'/media/banikr/banikr/SpinalCordInjury/MALDI/210603-Chen_protein_slide_F'
# saveSegPath = os.path.join(proteinDir, 'segmentation_vae_vae.h5')
# with h5py.File(saveSegPath, 'r') as pfile:  # saves the data
#     segImg = np.array(pfile['seg'])
#
# maskImg = np.zeros_like(segImg)
# maskImg[np.where(segImg != 0)] = 1
#
# labeled_array, nFeatures = ndimage.label(maskImg, structure=np.ones((3, 3)))
# print(nFeatures, "sections found...")
#
# nSliceMALDI = [1]#[1, 2, 3, 4]
# lutSeg = [3, 1, 2]    # from center GM, WM, connective/peripheral, this is
# for secID in nSliceMALDI:  # ,3,4]:
#     minx, miny = np.inf, np.inf
#     maxx, maxy = -np.inf, -np.inf
#     for x in range(labeled_array.shape[0]):
#         for y in range(labeled_array.shape[1]):
#             if labeled_array[x, y] == secID:
#                 minx, miny = min(minx, x), min(miny, y)
#                 maxx, maxy = max(maxx, x), max(maxy, y)
#     regionshape = [maxx - minx + 1,
#                    maxy - miny + 1]
#     secImg = segImg[minx:maxx, miny:maxy]
#     displayImage(secImg, Title_='VAE segmentation:{} area: {}'.format(secID, regionshape[0]*regionshape[1]))
#
#     if len(np.unique(secImg)) > 4:
#         secImg_ = np.zeros_like(secImg)
#         secImg_[np.where(secImg == 1)] = lutSeg[0]
#         secImg_[np.where(secImg == 2)] = lutSeg[0]
#         secImg_[np.where(secImg == 3)] = lutSeg[1]
#         secImg_[np.where(secImg == 4)] = lutSeg[1]
#         secImg_[np.where(secImg == 5)] = 0  # lutSeg[2]
#         displayImage(label2rgb(secImg_), Title_='3 label segmentation')
#         secImg = secImg_

# secImg = label2rgb(secImg_)
# print("here", secImg.shape)

# +~~~~~~~~~~+
# |   mri    |
# +~~~~~~~~~~+
proteinDir = r'/media/banikr/banikr/SpinalCordInjury/MALDI/210603-Chen_protein_slide_F'
dataFold = r'/media/banikr/banikr/SpinalCordInjury/Chase_high_res'
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
nSliceMALDI = sliceIdcsMALDI[2]
nSliceMRI = sliceIdcsMRI[2]
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
# displayImage(secImg, 'before nn')

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
classes_ = 4
gamma_val = 1.9
     #[0~4]
# scSliceGamma = adjust_gamma(scClipped[:, :, nSlice], gamma=gamma_val)     # updated to the below:
scSliceGamma = adjust_gamma(scClipped, gamma=gamma_val)
scSliceGammaEq = equalize_hist(scSliceGamma)
scSliceGammaEqCLAHE = equalize_adapthist(scSliceGammaEq, kernel_size=8)

# displayMR(scSliceGammaEqCLAHE, Title_='CLAHE')
# print()
scSliceGammaEqCLAHEresized = resize(scSliceGammaEqCLAHE,
                        tuple(3 * x for x in scSliceGammaEqCLAHE.shape),
                        # mode='edge',
                        anti_aliasing=False,  # to preserve label values
                        preserve_range=True,
                        order=0)
scSliceGammaEqCLAHEresized = scSliceGammaEqCLAHEresized/ np.max(scSliceGammaEqCLAHEresized)
displayMR(scSliceGammaEqCLAHEresized, Title_='CLAHE resized(normed 0 - 1)')
thresholds = filters.threshold_multiotsu(scSliceGammaEqCLAHEresized, classes=classes_)
regions = np.digitize(scSliceGammaEqCLAHEresized, bins=thresholds)
# print("regions \n", np.unique(regions))
# displayImage(regions, Title_='regions')

# secImg = resize(secImg,
#                 scSliceGammaEqCLAHEresized.shape,
#                 # mode='edge',
#                 anti_aliasing=False,  # to preserve label values
#                 preserve_range=True,
#                 order=0)

def command_iteration2(method):
    global metric_values
    print("metric value: {}".format(method.GetMetricValue()))
    metric_values.append(method.GetMetricValue())

def command_iteration(filter):
    global metric_values
    print(f"{filter.GetElapsedIterations():3} = {filter.GetMetric():10.5f}")
    metric_values.append(filter.GetMetric())

fixed_image = sitk.Cast(sitk.GetImageFromArray(scSliceGammaEqCLAHEresized), sitk.sitkFloat32)
# moving_image = sitk.Cast(sitk.GetImageFromArray(np.fliplr(np.rot90(secImg, axes=(1, 0)))), sitk.sitkFloat32)
moving_image = sitk.Cast(sitk.GetImageFromArray(secImg), sitk.sitkFloat32)
print("fixed moving size: ", fixed_image.GetSize(), moving_image.GetSize())
displayImage(sitk.GetArrayFromImage(fixed_image), Title_='fixed image(normed)')
displayImage(sitk.GetArrayFromImage(moving_image), Title_='moving image')
if __name__ == '__main__':
    metric_values = []
    degrees = np.linspace(0, 360, 1000)
    radians = np.deg2rad(degrees)

    initial_transform = sitk.Euler2DTransform(sitk.CenteredTransformInitializer(
        fixed_image,
        moving_image,
        sitk.Euler2DTransform(),
        sitk.CenteredTransformInitializerFilter.MOMENTS))      # updated: from GEOMETRY
    registration_method = sitk.ImageRegistrationMethod()
    # registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=500)
    # registration_method.SetMetricSamplingStrategy(registration_method.NONE)
    registration_method.SetInitialTransform(initial_transform, inPlace=False)   # updated from False
    # registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=0.50,
    #                                                              minStep=1e-3,
    #                                                              numberOfIterations=500,
    #                                                              gradientMagnitudeTolerance=1e-2)
    registration_method.SetOptimizerAsConjugateGradientLineSearch(learningRate=1.0,
                                                                  numberOfIterations=100,
                                                                  convergenceMinimumValue=1e-6,
                                                                  convergenceWindowSize=10)
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
    # best_angle = 310.99099099099095
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
    # displayImage(sitk.GetArrayFromImage(fixed_image), Title_='fixed image')
    # displayImage(sitk.GetArrayFromImage(moving_image), Title_='moving image')
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
    metric_values = []
    registration_method.SetMovingInitialTransform(eulerTx)
    registration_method.SetInitialTransform(affine, inPlace=True)
    # registration_method.SetMetricAsMeanSquares()
    # registration_method.SetMetricAsJointHistogramMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricAsANTSNeighborhoodCorrelation(100)      # updated from 4
    registration_method.SetOptimizerScalesFromIndexShift()
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=0.075,
        numberOfIterations=300,
        convergenceMinimumValue=10, convergenceWindowSize=100,
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
    plt.plot(metric_values)
    plt.show()
    checkerImg = compare_images(sitk.GetArrayFromImage(out),
                                sitk.GetArrayFromImage(fixed_image),
                                method='diff')
    displayImage(checkerImg, Title_='mri - (euler + affine) diff')

    if __name__ != '__main__':
        # works good... can be improved ?
        # deformable registration
        metric_values = []
        demons = sitk.FastSymmetricForcesDemonsRegistrationFilter()
        demons.SetNumberOfIterations(2000)
        # demons.SetStandardDeviations(3.0)      # updated from 1.2
        demons.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(demons))
        # # metric_values.append()
        transform_to_displacment_field_filter = sitk.TransformToDisplacementFieldFilter()
        transform_to_displacment_field_filter.SetReferenceImage(fixed_image)
        displacementTx = sitk.DisplacementFieldTransform(transform_to_displacment_field_filter.Execute
                                                         (sitk.Transform(fixed_image.GetDimension(),
                                                                         sitk.sitkIdentity)))
        displacementTx.SetSmoothingGaussianOnUpdate(varianceForUpdateField=0.0, varianceForTotalField=1.5)      #  updated varTotField = 1.5

        if __name__ != '__main__':
            shrinkFactors = [4, 2, 1]
            smoothingSigmas = [2, 1, 0]
            registration_method.setInitialTransform(displacementTx)
            registration_method.SetMetricAsDemons(0.5)
            registration_method.SetShrinkFactorsPerLevel(shrinkFactors=shrinkFactors)
            registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=smoothingSigmas)
            registration_method.SetOptimizerScalesFromIndexShift()
            registration_method.SetOptimizerAsConjugateGradientLineSearch(learningRate=1.0,
                                                                          numberOfIterations=20,
                                                                          convergenceMinimumValue=1e-6,
                                                                          convergenceWindowSize=10)

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
        displayImage(checkerImg, Title_='msi - mri')
        plt.plot(metric_values)
        plt.ylabel("Registration metric")
        plt.xlabel("Iterations")
        plt.show()

    # +~~~~~~~~~~~~~~~~~~~~~~~~~~+
    # |   works almost perfect   |
    # +~~~~~~~~~~~~~~~~~~~~~~~~~~+
    if __name__ == '__main__':
        metric_values = []
        registration_method = sitk.ImageRegistrationMethod()

        # Create initial identity transformation.
        transform_to_displacment_field_filter = sitk.TransformToDisplacementFieldFilter()
        transform_to_displacment_field_filter.SetReferenceImage(fixed_image)
        # The image returned from the initial_transform_filter is transferred to the transform and cleared out.
        initial_transform = sitk.DisplacementFieldTransform(
            transform_to_displacment_field_filter.Execute(sitk.Transform(fixed_image.GetDimension(),
                                                                         sitk.sitkIdentity)))

        # Regularization (update field - viscous, total field - elastic).
        initial_transform.SetSmoothingGaussianOnUpdate(varianceForUpdateField=0.5, varianceForTotalField=2.5)
        registration_method.SetMovingInitialTransform(compositeTx)
        registration_method.SetInitialTransform(initial_transform, inPlace=True)

        # registration_method.SetMetricAsDemons(5)  # intensities are equal if the difference is less than 10HU

        # Multi-resolution framework.
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[8, 4, 0.0001])

        registration_method.SetInterpolator(sitk.sitkNearestNeighbor)
        # If you have time, run this code as is, otherwise switch to the gradient descent optimizer
        # registration_method.SetOptimizerAsConjugateGradientLineSearch(learningRate=1.0, numberOfIterations=20, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
        # registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=20,
        #                                                   convergenceMinimumValue=1e-6, convergenceWindowSize=10)

        registration_method.SetMetricAsANTSNeighborhoodCorrelation(4)  # updated from 4
        # registration_method.SetMetricAsDemons(7)
        registration_method.MetricUseFixedImageGradientFilterOff() # not sure what it does but saw it in example
        registration_method.SetOptimizerScalesFromPhysicalShift()
        # registration_method.SetOptimizerScalesFromIndexShift()
        registration_method.SetOptimizerAsGradientDescent(
            learningRate=1.05,
            numberOfIterations=250,
            convergenceMinimumValue=1e-6, convergenceWindowSize=250,
            estimateLearningRate=registration_method.EachIteration,
        )

        # registration_method.SetOptimizerAsConjugateGradientLineSearch(learningRate=0.75,
        #                                                               numberOfIterations=150,
        #                                                               convergenceMinimumValue=1e-6,
        #                                                               convergenceWindowSize=100,
        #                                                               estimateLearningRate=registration_method.EachIteration)
        registration_method.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration2(registration_method))
        outTx = registration_method.Execute(fixed_image, moving_image)
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
        displayImage(checkerImg, 'msi - mri')
        plt.plot(metric_values)
        plt.ylabel("Registration metric")
        plt.xlabel("Iterations")
        plt.show()

# +--------------------------+
# |  multi-res registration  |
# +--------------------------+
if __name__ != '__main__':
    def demons_registration(fixed_image, moving_image,
                            fixed_points=None, moving_points=None):
        registration_method = sitk.ImageRegistrationMethod()

        # Create initial identity transformation.
        transform_to_displacment_field_filter = sitk.TransformToDisplacementFieldFilter()
        transform_to_displacment_field_filter.SetReferenceImage(fixed_image)
        # The image returned from the initial_transform_filter is transferred to the transform and cleared out.
        initial_transform = sitk.DisplacementFieldTransform(
            transform_to_displacment_field_filter.Execute(sitk.Transform())
        )

        # Regularization (update field - viscous, total field - elastic).
        initial_transform.SetSmoothingGaussianOnUpdate(
            varianceForUpdateField=0.0, varianceForTotalField=2.0
        )

        registration_method.SetInitialTransform(initial_transform)

        registration_method.SetMetricAsDemons(
            10
        )  # intensities are equal if the difference is less than 10HU

        # Multi-resolution framework.
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[8, 4, 0])

        registration_method.SetInterpolator(sitk.sitkNearestNeighbor)
        # If you have time, run this code as is, otherwise switch to the gradient descent optimizer
        registration_method.SetOptimizerAsConjugateGradientLineSearch(learningRate=1.0,
                                                                      numberOfIterations=20,
                                                                      convergenceMinimumValue=1e-6,
                                                                      convergenceWindowSize=10)
        # registration_method.SetOptimizerAsGradientDescent(
        #     learningRate=1.0,
        #     numberOfIterations=20,
        #     convergenceMinimumValue=1e-6,
        #     convergenceWindowSize=10,
        # )
        registration_method.SetOptimizerScalesFromPhysicalShift()

        # If corresponding points in the fixed and moving image are given then we display the similarity metric
        # and the TRE during the registration.
        if fixed_points and moving_points:
            registration_method.AddCommand(
                sitk.sitkStartEvent, rc.metric_and_reference_start_plot
            )
            registration_method.AddCommand(
                sitk.sitkEndEvent, rc.metric_and_reference_end_plot
            )
            registration_method.AddCommand(
                sitk.sitkIterationEvent,
                lambda: rc.metric_and_reference_plot_values(
                    registration_method, fixed_points, moving_points
                ),
            )
        return registration_method.Execute(fixed_image, moving_image)

# +------------------------------+
# |     multi-scale demon        |
# +------------------------------+
if __name__ != '__main__':
    shrinkFactors = [4, 2, 1]
    smoothingSigmas = [8, 4, 0.000000001]
    # fixed_images = [fixed_image]
    # moving_images = [moving_image]
    # i=0
    # for shrink_factor, smoothing_sigma in reversed(list(zip(shrinkFactors, smoothingSigmas))):
    #     fixed_images.append(smooth_and_resample(fixed_images[0], shrink_factor, smoothing_sigma))
    #     moving_images.append(smooth_and_resample(moving_images[0], shrink_factor, smoothing_sigma))
    #     displayImage(sitk.GetArrayFromImage(fixed_images[i+1]), Title_='sf: {}, ss: {}'.format(shrink_factor, smoothing_sigma))
    #     i+=1
    metric_values = []
    demons_filter = sitk.FastSymmetricForcesDemonsRegistrationFilter()
    demons_filter.SetNumberOfIterations(2000)
    demons_filter.SetSmoothDisplacementField(True)
    demons_filter.SetStandardDeviations(2)
    # demons_filter.SetMaximumError(0.01)
    demons_filter.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(demons_filter))
    print("image size >> ", fixed_image.GetSize(), out.GetSize())
    # Run the registration.
    outTx = multiscale_demons_filter(registration_algorithm=demons_filter,
                           fixed_image=fixed_image,
                           moving_image=out,        # updated: from moving_image
                           shrink_factors=shrinkFactors,
                           smoothing_sigmas=smoothingSigmas)
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
    displayImage(checkerImg, Title_='msi - mri')
    plt.plot(metric_values)
    plt.ylabel("Registration metric")
    plt.xlabel("Iterations")
    plt.show()

    # print("low res image >> ", fixed_images[-1].GetSize())
    #
    # initial_displacement_field = sitk.Image(fixed_images[-1].GetWidth(),
    #                                         fixed_images[-1].GetHeight(),
    #                                         fixed_images[-1].GetDepth(),
    #                                         fixed_images[-1].GetDimension(),
    #                                         # sitk.sitkVectorFloat64
    #                                         )
    # # initial_displacement_field.SetReferenceImage(fixed_images[-1])
    # initial_displacement_field.CopyInformation(fixed_images[-1])

# +--------------------------+
# |    saving transforms     |
# +--------------------------+
if __name__ == '__main__':
    # import pickle, copy
    # .pickle , .bin none works
    finalCompTx = sitk.CompositeTransform([eulerTx, affineTx, outTx])
    finalCompTxFile = os.path.join(dataFold, '{}_mr{}_ms{}_finalCompTx.hdf'.format(os.path.basename(os.path.normpath(niiPath).split('.')[0]),
                                                                           nSliceMRI, nSliceMALDI))
    # outTxFile = os.path.join(dataFold, '{}_mr{}_ms{}_outTx.hdf'.format(os.path.basename(os.path.normpath(niiPath).split('.')[0]),
    #                                                             nSlice, nSliceMALDI[0]))
    sitk.WriteTransform(finalCompTx, finalCompTxFile)
    # with open(finalCompTxFile, 'wb') as fp:
    #     p = pickle.dump(copy.deepcopy(finalCompTx), fp)
    # sitk.WriteTransform(outTx, outTxFile)
