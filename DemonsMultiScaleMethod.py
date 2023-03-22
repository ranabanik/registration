import os
import numpy as np
import nibabel as nib
import h5py
from scipy import ndimage
from skimage.transform import resize
from skimage.measure import regionprops, label
from skimage.morphology import area_closing
from skimage import filters
from skimage.exposure import rescale_intensity, adjust_gamma, equalize_hist, equalize_adapthist
from utils import displayImage, displayMR, multi_dil, bbox2, nnPixelCorrect, smooth_and_resample
from utils import multiscale_demons_filter, multiscale_demons
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib as mtl
from matplotlib.colors import LinearSegmentedColormap

def displayOverlap(mr, seg, colorbins=3, Title_=None):
    if 'msml2' not in plt.colormaps():
        colors = [
            (0.0, 0.0, 0.0, 0.0),
            (0.2, 0.0, 0.8, 0.1),
            (0.85, 0.0, 0.4, 0.1),
            (1.0, 0.0, 0.0, 0.1)
        ]
        color_bin = colorbins
        mtl.colormaps.register(LinearSegmentedColormap.from_list(name='msml2', colors=colors, N=color_bin))
    fig, ax = plt.subplots(figsize=[5, 4], dpi=1000)
    ax.imshow(seg, origin='lower', cmap='msml2', alpha=0.5, zorder=1,)
    pcm = ax.pcolormesh(seg, cmap='msml2')
    fig.colorbar(pcm, orientation='vertical', anchor=(0.0, 1.0), ax=ax, shrink=0.5, pad=0.02)
    ax.imshow(mr, origin='lower', cmap='gray', alpha=1.0, zorder=0)
    if Title_ is not None:
        plt.title(Title_)
    plt.show()

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
nSliceMALDI = sliceIdcsMALDI[3]
nSliceMRI = sliceIdcsMRI[1]
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
scSliceGammaEqCLAHEresized = scSliceGammaEqCLAHEresized/ np.max(scSliceGammaEqCLAHEresized)
displayMR(scSliceGammaEqCLAHEresized, Title_='CLAHE resized(normed 0 - 1)')
thresholds = filters.threshold_multiotsu(scSliceGammaEqCLAHEresized, classes=classes_)
regions = np.digitize(scSliceGammaEqCLAHEresized, bins=thresholds)
displayImage(regions, 'Multi-otsu segmentation')

def command_iteration2(method):
    global metric_values
    print("metric value: {}".format(method.GetMetricValue()))
    metric_values.append(method.GetMetricValue())

def command_iteration(filter):
    global metric_values
    print(f"{filter.GetElapsedIterations():3} = {filter.GetMetric():10.5f}")
    metric_values.append(filter.GetMetric())

fixed_image = sitk.Cast(sitk.GetImageFromArray(scSliceGammaEqCLAHEresized), sitk.sitkFloat32) # regions
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
    # best_angle = 278.9189189189189#310.99099099099095
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

    displayImage(sitk.GetArrayFromImage(out), 'Euler2D')
    affine = sitk.AffineTransform(fixed_image.GetDimension())

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
    displayImage(sitk.GetArrayFromImage(out), 'Euler2D + Affine')
    plt.plot(metric_values)
    plt.show()
    displayOverlap(sitk.GetArrayFromImage(fixed_image),
                                sitk.GetArrayFromImage(out),
                                Title_='mri - (euler + affine) diff')
# so far best
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
    initial_transform.SetSmoothingGaussianOnUpdate(varianceForUpdateField=0.5, varianceForTotalField=12.5)
    registration_method.SetMovingInitialTransform(compositeTx)
    registration_method.SetInitialTransform(initial_transform, inPlace=True)

    # Multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[4, 2, 0.01])

    registration_method.SetInterpolator(sitk.sitkNearestNeighbor)
    # If you have time, run this code as is, otherwise switch to the gradient descent optimizer
    # registration_method.SetOptimizerAsConjugateGradientLineSearch(learningRate=1.0, numberOfIterations=20, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    # registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=20,
    #                                                   convergenceMinimumValue=1e-6, convergenceWindowSize=10)

    registration_method.SetMetricAsANTSNeighborhoodCorrelation(16)  # updated from 4
    # registration_method.SetMetricAsDemons(0.07)  # distorts the image...
    registration_method.MetricUseFixedImageGradientFilterOff()  # not sure what it does but saw it in example
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.05,
        numberOfIterations=250,
        convergenceMinimumValue=1e-6, convergenceWindowSize=250,
        estimateLearningRate=registration_method.EachIteration,
    )
    # registration_method.SetOptimizerAsConjugateGradientLineSearch(learningRate=.05, # this optimizer doesn't work
    #                                                               numberOfIterations=250,
    #                                                               convergenceMinimumValue=1e-6,
    #                                                               convergenceWindowSize=250,
    #                                                               estimateLearningRate=registration_method.EachIteration)
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration2(registration_method))
    outTx = registration_method.Execute(fixed_image, moving_image)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(outTx)
    out = resampler.Execute(out)
    displayImage(sitk.GetArrayFromImage(out), 'Euler + Affine + DemonsDFRegistration')
    displayOverlap(sitk.GetArrayFromImage(fixed_image),
                   sitk.GetArrayFromImage(out),
                   Title_='mri - (euler + affine + demon) diff')
    plt.plot(metric_values)
    plt.ylabel("Registration metric")
    plt.xlabel("Iterations")
    plt.show()

# +--------------------------+
# |    saving transforms     |
# +--------------------------+
if __name__ == '__main__':
    finalCompTx = sitk.CompositeTransform([eulerTx, affineTx, outTx])
    finalCompTxFile = os.path.join(dataFold, '{}_mr{}_ms{}_finalCompTx.hdf'.format(os.path.basename(os.path.normpath(niiPath).split('.')[0]),
                                                                           nSliceMRI, nSliceMALDI))
    sitk.WriteTransform(finalCompTx, finalCompTxFile)

# +-------------------------------------------+
# |  demon w/o `registration` method          |
# |     - distorts the m_image too much       |
# +-------------------------------------------+
if __name__ != '__main__':
    shrinkFactors = [4, 2, 1]
    smoothingSigmas = [4, 2, 0.01]
    metric_values = []
    demons_filter = sitk.FastSymmetricForcesDemonsRegistrationFilter()
    demons_filter.SetNumberOfIterations(50)
    # demons_filter.SetStandardDeviations(1.2)
    demons_filter.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(demons_filter))
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
    displayOverlap(sitk.GetArrayFromImage(fixed_image),
                   sitk.GetArrayFromImage(out),
                   Title_='mri - (euler + affine + demon) diff')
    plt.plot(metric_values)
    plt.ylabel("Registration metric")
    plt.xlabel("Iterations")
    plt.show()
if __name__ != '__main__':
    shrinkFactors = [4, 2, 1]
    smoothingSigmas = [4, 2, 0.000000001]
    metric_values = []
    demons_filter = sitk.DemonsRegistrationFilter()
    demons_filter.SetNumberOfIterations(2000)
    demons_filter.SetSmoothDisplacementField(True)
    demons_filter.SetStandardDeviations(2.5)
    demons_filter.SetMaximumError(0.1)
    demons_filter.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(demons_filter))
    outTx = multiscale_demons(registration_algorithm=demons_filter,
                                     fixed_image=fixed_image,
                                     moving_image=out,  # updated: from moving_image
                                     shrink_factors=shrinkFactors,
                                     smoothing_sigmas=smoothingSigmas)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(outTx)
    out = resampler.Execute(out)
    displayImage(sitk.GetArrayFromImage(out), Title_='Euler + Affine + FSFDemonsRegistrationFilter')
    displayOverlap(sitk.GetArrayFromImage(fixed_image),
                   sitk.GetArrayFromImage(out),
                   Title_='mri - (euler + affine + demon) diff')
    plt.plot(metric_values)
    plt.ylabel("Registration metric")
    plt.xlabel("Iterations")
    plt.show()
if __name__ != '__main__':
    # deformable registration runs but results are not good.
    shrink_factors = [4, 2, 1]
    smoothing_sigmas = [4, 2, 0.000000001]
    metric_values = []
    demons_filter = sitk.FastSymmetricForcesDemonsRegistrationFilter()
    demons_filter.SetNumberOfIterations(500)
    demons_filter.SetStandardDeviations(1.2)
    demons_filter.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(demons_filter))
    # # metric_values.append()

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
    print("dfs \n", df_spacing, '\n', df_size)
    fixed_images = [fixed_image]
    moving_images = [out]
    for shrink_factor, smoothing_sigma in reversed(list(zip(shrink_factors, smoothing_sigmas))):
        fixed_images.append(smooth_and_resample(fixed_images[0], shrink_factor, smoothing_sigma))
        moving_images.append(smooth_and_resample(moving_images[0], shrink_factor, smoothing_sigma))
    print(">>", fixed_images[-1].GetSize())
    transform_to_displacment_field_filter = sitk.TransformToDisplacementFieldFilter()
    transform_to_displacment_field_filter.SetReferenceImage(fixed_images[-1])
    displacementTx = sitk.DisplacementFieldTransform(transform_to_displacment_field_filter.Execute
                                                     (sitk.Transform(fixed_images[-1].GetDimension(),
                                                                     sitk.sitkIdentity)))
    displacementTx.SetSmoothingGaussianOnUpdate(varianceForUpdateField=10.0, varianceForTotalField=4.5)
    for f_image, m_image in reversed(list(zip(fixed_images, moving_images))):
        transform_to_displacment_field_filter.SetReferenceImage(f_image)
        displacementTx = sitk.DisplacementFieldTransform(transform_to_displacment_field_filter.Execute
                                                         (sitk.Transform(f_image.GetDimension(),
                                                                         sitk.sitkIdentity)))
        # displacementTx.SetSmoothingGaussianOnUpdate(varianceForUpdateField=0.0, varianceForTotalField=1.5)
        displacementField = transform_to_displacment_field_filter.Execute(displacementTx)
        displacementField = demons_filter.Execute(f_image, m_image, displacementField)
    outTx = sitk.DisplacementFieldTransform(displacementField)

    # until the above multiscale
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(outTx)
    out = resampler.Execute(out)
    displayImage(sitk.GetArrayFromImage(out), Title_='Euler + Affine + FSFDemonsRegistrationFilter')
    displayOverlap(sitk.GetArrayFromImage(fixed_image),
                   sitk.GetArrayFromImage(out),
                   Title_='mri - (euler + affine + demon) diff')
    plt.plot(metric_values)
    plt.ylabel("Registration metric")
    plt.xlabel("Iterations")
    plt.show()
