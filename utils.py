import matplotlib as mtl
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from skimage.morphology import area_closing, dilation
import SimpleITK as sitk
import copy
from numpy.lib.stride_tricks import as_strided

def nnPixelCorrect(arr, n_, d, bg_=0, plot_=True):
    """
    corrects the pixel value based on neighnoring pixels
    n_: value of noise pixel to correct
    bg_: backgroud pixel value, default 0.
    d: degree of neighbor
    """
    def sliding_window(arr, window_size):
        """ Construct a sliding window view of the array"""
        arr = np.asarray(arr)
        window_size = int(window_size)
        if arr.ndim != 2:
            raise ValueError("need 2-D input")
        if not (window_size > 0):
            raise ValueError("need a positive window size")
        shape = (arr.shape[0] - window_size + 1,
                 arr.shape[1] - window_size + 1,
                 window_size, window_size)
        if shape[0] <= 0:
            shape = (1, shape[1], arr.shape[0], shape[3])
        if shape[1] <= 0:
            shape = (shape[0], 1, shape[2], arr.shape[1])
        strides = (arr.shape[1]*arr.itemsize, arr.itemsize,
                   arr.shape[1]*arr.itemsize, arr.itemsize)
        return as_strided(arr, shape=shape, strides=strides)

    def cell_neighbors(arr, i, j, d):
        """Return d-th neighbors of cell (i, j)"""
        w = sliding_window(arr, 2*d+1)
        ix = np.clip(i - d, 0, w.shape[0]-1)
        jx = np.clip(j - d, 0, w.shape[1]-1)
        i0 = max(0, i - d - ix)
        j0 = max(0, j - d - jx)
        i1 = w.shape[2] - max(0, d - i + ix)
        j1 = w.shape[3] - max(0, d - j + jx)
        return w[ix, jx][i0:i1, j0:j1].ravel()

    def most_common(lst):
        return max(set(lst), key=lst.count)
    arr_ = copy.deepcopy(arr)
    noiseIndices = np.where(arr == n_)
    listOfCoordinates = list(zip(noiseIndices[0], noiseIndices[1]))
    for coOrd in listOfCoordinates:
        # print("noise ind: ", cord[0], cord[1])
        cn = cell_neighbors(arr, coOrd[0], coOrd[1], d)
        cn = np.delete(cn, np.where((cn == bg_) | (cn == n_)))
        arr[coOrd[0], coOrd[1]] = most_common(cn.tolist())
    if plot_:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        arrays = [arr_, arr]
        title = ['noisy', 'corrected']
        fig, axs = plt.subplots(1, 2, figsize=(10, 8), dpi=200, sharex=False)
        for ar, tl, ax in zip(arrays, title, axs.ravel()):
            im = ax.imshow(ar) #, cmap='twilight') #cm)
            ax.set_title(tl, fontsize=20)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, ax=ax)
        plt.show()
    return arr

def displayMR(MR, Title_= 'Demo'):
    plt.imshow(MR, origin='lower', cmap='gray')
    plt.title(Title_)
    plt.colorbar()
    plt.show()

def displayImage(matrix, Title_='demo'):
    if 'msml_list' not in plt.colormaps():
        # colors = [(0.1, 0.1, 0.1), (0.0, 0, 1.0), (0.0, 0.5, 0.5), (0.0, 1.0, 0.0), (0.5, 0.5, 0.0), (1.0, 0, 0.0)]  # Bk -> bl -> G -> r
        colors = [(0.1, 0.1, 0.1),
                  (0.0, 0.0, 1.0),  # (0.0, 0.5, 0.5),
                  (0.0, 1.0, 0.0),  # (0.5, 0.5, 0.0),
                  (1.0, 0.0, 0.0)]
                  # (1.0, 0, 0.0)]  # Bk -> R -> G -> Bl
        color_bin = 256
        mtl.colormaps.register(LinearSegmentedColormap.from_list(name='msml_list', colors=colors, N=color_bin))
    plt.imshow(matrix, origin='lower', cmap='msml_list')
    plt.title(Title_)
    plt.colorbar()
    plt.show()

def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def multi_dil(im, num, element):
    for i in range(num):
        im = dilation(im, element)
    return im
# metric_values = []
# def command_iteration2(method):
#     global metric_values
#     print("metric value: {}".format(method.GetMetricValue()))
#     metric_values.append(method.GetMetricValue())

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
    new_size = [int(sz / float(sf) + 0.5) for sf, sz in zip(shrink_factors, original_size)]
    new_spacing = [((original_sz - 1) * original_spc) / (new_sz - 1)
        for original_sz, original_spc, new_sz in zip(original_size, original_spacing, new_size)]
    return sitk.Resample(smoothed_image,
                        new_size,
                        sitk.Transform(),
                        sitk.sitkNearestNeighbor,
                        image.GetOrigin(),
                        new_spacing,
                        image.GetDirection(),
                        0.0,
                        image.GetPixelID())


def image_pair_generator(fixed_image, moving_image, shrink_factors, smoothing_sigmas):
    """
    Create image pyramid in a memory efficient manner using a generator function.
    The whole pyramid never exists in memory, each level is created when iterating over
    the generator.
    """
    end_level = 0
    start_level = 0
    if shrink_factors is not None:
        end_level = len(shrink_factors)
    for level in range(start_level, end_level):
        f_image = smooth_and_resample(fixed_image, shrink_factors[level], smoothing_sigmas[level])
        m_image = smooth_and_resample(moving_image, shrink_factors[level], smoothing_sigmas[level])
        yield (f_image, m_image)
    yield (fixed_image, moving_image)


def multiscale_demons(registration_algorithm,
                      fixed_image, moving_image, initial_transform=None,
                      shrink_factors=None, smoothing_sigmas=None):
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
    # Create image pyramid.
    fixed_images = [fixed_image]
    moving_images = [moving_image]
    if shrink_factors:
        i = 0
        for shrink_factor, smoothing_sigma in reversed(list(zip(shrink_factors, smoothing_sigmas))):
            fixed_images.append(smooth_and_resample(fixed_images[0], shrink_factor, smoothing_sigma))
            moving_images.append(smooth_and_resample(moving_images[0], shrink_factor, smoothing_sigma))
            print(i+1, ":", fixed_images[i+1].GetSize(), "==", moving_images[i+1].GetSize())
            i += 1
    # Create initial displacement field at lowest resolution.
    # Currently, the pixel type is required to be sitkVectorFloat64 because of a constraint imposed by the Demons filters.
    if initial_transform:
        initial_displacement_field = sitk.TransformToDisplacementField(initial_transform,
                                                                       sitk.sitkVectorFloat64,
                                                                       fixed_images[-1].GetSize(),
                                                                       fixed_images[-1].GetOrigin(),
                                                                       fixed_images[-1].GetSpacing(),
                                                                       fixed_images[-1].GetDirection())
    else:
        print("depth: ", fixed_images[-1].GetDepth())
        initial_displacement_field = sitk.Image(fixed_images[-1].GetWidth(),
                                                fixed_images[-1].GetHeight(),
                                                fixed_images[-1].GetDepth(),
                                                sitk.sitkVectorFloat64)
        # print(">>", fixed_images[-1])
        initial_displacement_field.CopyInformation(fixed_images[-1])

    # print("This happens ...")
    # Run the registration.
    initial_displacement_field = registration_algorithm.Execute(fixed_images[-1],
                                                                moving_images[-1],
                                                                initial_displacement_field)
    # Start at the top of the pyramid and work our way down.
    for f_image, m_image in reversed(list(zip(fixed_images[0:-1], moving_images[0:-1]))):
        initial_displacement_field = sitk.Resample(initial_displacement_field, f_image)
        initial_displacement_field = registration_algorithm.Execute(f_image, m_image, initial_displacement_field)
    return sitk.DisplacementFieldTransform(initial_displacement_field)

def multiscale_demons_filter(registration_algorithm,
                      fixed_image, moving_image, #initial_transform=None,
                      shrink_factors=None, smoothing_sigmas=None):
    """
    :param registration_algorithm: any registration algorithm with an `Execute`
    :param fixed_image:
    :param moving_image:
    :param initial_transform: if necessary in future...
    :param shrink_factors:
    :param smoothing_sigmas:
    :return:
    """
    # Create image pyramid.
    fixed_images = [fixed_image]
    moving_images = [moving_image]
    if shrink_factors:
        for shrink_factor, smoothing_sigma in reversed(list(zip(shrink_factors, smoothing_sigmas))):
            fixed_images.append(smooth_and_resample(fixed_images[0], shrink_factor, smoothing_sigma))
            moving_images.append(smooth_and_resample(moving_images[0], shrink_factor, smoothing_sigma))

    transform_to_displacment_field_filter = sitk.TransformToDisplacementFieldFilter()
    transform_to_displacment_field_filter.SetReferenceImage(fixed_images[-1])
    displacementTx = sitk.DisplacementFieldTransform(transform_to_displacment_field_filter.Execute
                                                     (sitk.Transform(fixed_image.GetDimension(),
                                                                     sitk.sitkIdentity)))
    # Regularization (update field - viscous, total field - elastic).
    displacementTx.SetSmoothingGaussianOnUpdate(varianceForUpdateField=0.0, varianceForTotalField=2.5)

    initial_displacement_field = transform_to_displacment_field_filter.Execute(displacementTx)
    # Start at the top of the pyramid and work our way down.
    level = 0
    for f_image, m_image in reversed(list(zip(fixed_images, moving_images))):
        print("level >>", level+1, f_image.GetSize(), m_image.GetSize())
        initial_displacement_field = sitk.Resample(initial_displacement_field, f_image)
        initial_displacement_field = registration_algorithm.Execute(f_image, m_image, initial_displacement_field)
        level += 1
    return sitk.DisplacementFieldTransform(initial_displacement_field)