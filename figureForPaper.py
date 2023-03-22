# create figures for publication
# to understand more about colors: https://matplotlib.org/stable/tutorials/colors/colors.html
import os
from glob import glob
import h5py
import copy
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib as mtl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.transform import resize
from skimage.exposure import adjust_gamma, rescale_intensity, equalize_hist, equalize_adapthist
from skimage import filters
from skimage.color import label2rgb
from utils import bbox2
# todo: remove the white spaces in the saved figure.
figureDir = r'/media/banikr/banikr/SpinalCordInjury/MALDI/Figures_for_publication'
proteinDir = r'/media/banikr/banikr/SpinalCordInjury/MALDI/210603-Chen_protein_slide_F'
readSegPath = os.path.join(proteinDir, 'segmentation_vae_vae.h5')
proteinPath = os.path.join(proteinDir, 'sum_argrel.h5')  # 'peak_all_regions_2_1_100.h5')
readBICKneePath = os.path.join(proteinDir, 'BICscoreGMMknee.h5')
dataFold = r'/media/banikr/banikr/SpinalCordInjury/Chase_high_res'
niiPath = os.path.join(dataFold, '9.nii')
scMaskPath = os.path.join(dataFold, '9_SC_mask_wo_CSF.nii.gz')

if __name__ != '__main__':
    projectDir = r'/media/banikr/DATA/MALDI/fromPrimatefmri/210603-Chen_protein_slide_F'
    rawDir = os.path.join(projectDir, 'protein_sum_argrel_KL-raw_2023-01-23-16-09-19')
    reconPath = os.path.join(rawDir, 'reconstructed_spectra_latest.h5')
    proteinPath = glob(os.path.join(rawDir, 'sum_argrel.h5'))[0]      # 'peak_all_regions_2_1_100.h5')
    with h5py.File(proteinPath, 'r') as pfile:
        print(pfile.keys())
        spectra = np.array(pfile.get('spectra'), dtype=np.float32)
        peakmzs = np.array(pfile.get('peakmzs'))
        xloc = np.array(pfile.get('xloc'))
        yloc = np.array(pfile.get('yloc'))
    nSpec, nFeat = spectra.shape
    minX, maxX = np.min(xloc), np.max(xloc)
    minY, maxY = np.min(yloc), np.max(yloc)
    meanspec = np.mean(spectra, axis=0)
    nMaxSamp = 10
    maxIntIdx = np.argsort(np.mean(spectra, axis=0))[::-1][0:nMaxSamp]
    print(maxIntIdx)
    fig, ax = plt.subplots(1, figsize=(10, 6), dpi=100)
    ax.plot(peakmzs, meanspec, 'b', alpha=0.5)
    ax.plot(peakmzs[maxIntIdx], meanspec[maxIntIdx], 'rv', alpha=0.5)
    for i, j in zip(peakmzs[maxIntIdx], meanspec[maxIntIdx]):
        ax.annotate('{:.3f}, {:.3f}'.format(i, 100 * j), xy=(i + 0.001, j + 0.001), color='r', fontsize=10)
    # ax.vlines(peakmzs, ymin=0, ymax=spec, colors='#ffc300', alpha=0.2)
    plt.show()

    with h5py.File(reconPath, 'r') as pfile:
        print(pfile.keys())
        recon_spectra = np.array(pfile['spectra'], dtype=np.float32)

    spectra_norm = copy.deepcopy(spectra)
    for s in range(spectra.shape[0]):
        spectrum = spectra[s, :]
        specSum = sum(spectrum)
        if specSum > 0:
            spectrum /= specSum
        spectra_norm[s, :] = spectrum

    if 'msml_list' not in plt.colormaps():
        # colors = [(0.1, 0.1, 0.1), (0.0, 0, 1.0), (0.0, 0.5, 0.5), (0.0, 1.0, 0.0), (0.5, 0.5, 0.0), (1.0, 0, 0.0)]  # Bk -> bl -> G -> r
        colors = [(1.0, 1.0, 1.0),
                  # (0.9569, 0.7686, 0.1882),
                  (0.65, 0.71, 0.91),
                  (0.95, 0.3, 0.22),  # (0.0, 0.5, 0.5),
                  # (0.01, 0.01, 0.95),
                  # (0.01, 0.95, 0.01),  # (0.5, 0.5, 0.0),
                  # (0.65, 0.71, 0.81)
                  # (0.95, 0.01, 0.01)
                  ]
        # (1.0, 0, 0.0)]  # Bk -> R -> G -> Bl
        color_bin = 256
        mtl.colormaps.register(LinearSegmentedColormap.from_list(name='msml_list', colors=colors, N=color_bin))

    for ii, m in enumerate(maxIntIdx):
        im = np.zeros([max(xloc) + 1, max(yloc) + 1])
        recon_im = np.zeros([max(xloc) + 1, max(yloc) + 1])
        # print(im.shape)
        for s in range(nSpec):
            # print("what", xloc[s])
            im[xloc[s], yloc[s]] = spectra_norm[s, m]
            recon_im[xloc[s], yloc[s]] = recon_spectra[s, m]
        img1 = im[minX:maxX, minY:maxY]
        rec1 = recon_im[minX:maxX, minY:maxY]
        fig, axes = plt.subplots(1, 2, figsize=(6, 8), dpi=500)
        ax = axes.ravel()
        cmap_ = 'msml_list'
        ax[0].imshow(img1, cmap=cmap_)
        # ax[0].set_title("original", pad=0.01)
        ax[0].axis('off')
        pcm = ax[0].pcolormesh(img1, cmap=cmap_)
        fig.colorbar(pcm, ax=ax[0], shrink=0.4, pad=-0.2)  # , location={1.0, 0, 0, 0})
        ax[1].imshow(rec1, cmap=cmap_)
        # ax[1].set_title("reconstructed")
        ax[1].axis('off')
        pcm = ax[1].pcolormesh(rec1, cmap=cmap_)
        fig.colorbar(pcm, ax=ax[1], shrink=0.4, pad=-0.2)
        # fig.colorbar()
        # fig.suptitle("Protein ion image at m/z: {:.4f}".format(peakmzs[m]))
        fig.subplots_adjust(top=1.00,
                            bottom=0.0,
                            left=-0.08,
                            right=0.95,
                            hspace=0.0,
                            wspace=0.12)
        # plt.savefig(os.path.join(figureDir, 'ion_image_comparison_{}.png'.format(peakmzs[m])))
        plt.show()
        # break
if __name__ != '__main__':
    with h5py.File(proteinPath, 'r') as pfile:
        print(pfile.keys())
        spectra = np.array(pfile.get('spectra'), dtype=np.float32)
        peakmzs = np.array(pfile.get('peakmzs'))
        xloc = np.array(pfile.get('xloc'))
        yloc = np.array(pfile.get('yloc'))
    nSpec, nFeat = spectra.shape
    minX, maxX = np.min(xloc), np.max(xloc)
    minY, maxY = np.min(yloc), np.max(yloc)
    with h5py.File(readSegPath, 'r') as pfile:  # saves the data
        segImg = np.array(pfile['seg'])
    segImg = segImg[minX:maxX, minY:maxY]
    if 'msml_list' not in plt.colormaps():
        # colors = [(0.1, 0.1, 0.1), (0.0, 0, 1.0), (0.0, 0.5, 0.5), (0.0, 1.0, 0.0), (0.5, 0.5, 0.0), (1.0, 0, 0.0)]  # Bk -> bl -> G -> r
        colors = [(1.0, 1.0, 1.0, 1.0),
                  (0.9569, 0.7686, 0.1882),
                  # (0.65, 0.71, 0.91),
                  (0.95, 0.3, 0.22),  # (0.0, 0.5, 0.5),
                  # (0.01, 0.01, 0.95, 1.0),
                  # (0.01, 0.95, 0.01, 1.0),  # (0.5, 0.5, 0.0),
                  (0.65, 0.71, 0.81),
                  # (0.95, 0.01, 0.01, 1.0)
                  ]
                  # (1.0, 0, 0.0)]  # Bk -> R -> G -> Bl
        color_bin = 6
        mtl.colormaps.register(LinearSegmentedColormap.from_list(name='msml_list', colors=colors, N=color_bin))
    fig = plt.figure(figsize=(1.5, 2), dpi=600)#, layout='constrained')
    # ax = ax.ravel()
    cmap_ = 'msml_list'
    plt.imshow(segImg, cmap=cmap_)
    # ax[0].title("original", pad=0.01)
    pcm = plt.pcolormesh(segImg, cmap=cmap_)
    cb = plt.colorbar(pcm, shrink=0.5, ticks=[0, 1, 2, 3, 4, 5], format='%.0f')
    cb.ax.set_yticklabels(['background', 'GM-1', 'GM-2', 'WM-1', 'WM-2', 'connective\ntissue'], fontsize=5)
    # plt.colorbar()#, shrink=0.5, pad=-0.2)#, location={1.0, 0, 0, 0})
    # # fig.subplots_adjust(top=0.92, bottom=0.0, left=0.02,
    # #                     right=0.95, hspace=0.01, wspace=0.12)
     #)
    # plt.title("segmentation of latent space with 5 GMM")
    plt.axis('off')
    fig.tight_layout(rect=[0.0, 0.0, .95, 1])#pad=0.2)
    # plt.savefig(os.path.join(r'/media/banikr/banikr/SpinalCordInjury/MALDI/Figures_for_publication', 'segmentation_results_5_trans.png'))
    plt.show()
if __name__ != '__main__':
    with h5py.File(readBICKneePath, 'r') as pfile:  # saves the data
        BIC_Scores = np.array(pfile['BICscore'], dtype=np.int)
        n_components = np.array(pfile['n_components'], dtype=np.int)
        Elbow_idx = np.array(pfile['Elbow_idx'])
    # Elbow_idx = 6
    fig = plt.figure(dpi=500)
    plt.plot(n_components, BIC_Scores, '-g', marker='o', markerfacecolor='blue', markeredgecolor='orange',
             markeredgewidth='2', markersize=10, markevery=Elbow_idx, label='knee of the graph')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    # plt.legend(loc='best')
    plt.xlabel('number of components', fontsize=15)
    plt.ylabel('BIC score', fontsize=15)
    plt.title('suggested number of clusters:  ' + np.str(n_components[Elbow_idx][0]), fontsize=20) #  fontweight=10,  #kneedle_point.knee))
    fig.tight_layout(rect=[0.0, 0.0, .95, 1])
    # plt.savefig(os.path.join(r'/media/banikr/banikr/SpinalCordInjury/MALDI/Figures_for_publication',
    #                          'bicscorekneeplot.png'))
    plt.show()
# ion image overlap on MR
if __name__ != '__main__':
    nSliceMR = 2
    nSliceMALDI = 3
    pickMzList = [14115, 14197, 8566, 4961, 7063]
    pickMz = pickMzList[0]
    mrBlock = nib.load(niiPath).get_fdata()
    mrSlice = mrBlock[..., nSliceMR]
    scMask = nib.load(scMaskPath).get_fdata()
    scMaskSlice = scMask[..., nSliceMR]
    rmin, rmax, cmin, cmax = bbox2(scMaskSlice)
    mrSlice_x3 = resize(mrSlice,
                        tuple(3 * x for x in mrSlice.shape),
                        # mode='edge',
                        anti_aliasing=False,  # to preserve label values
                        preserve_range=True,
                        order=0
                        )
    mrSlice_x3_normed = mrSlice_x3 / np.max(mrSlice_x3)
    ionImageMRsize = np.zeros(tuple(3 * x for x in mrSlice.shape))
    ionImageOnMRPath = os.path.join(dataFold, '{}_mr{}_ms{}_ionImage_mz_{}_onMR.npy'.format(
        os.path.basename(os.path.normpath(niiPath).split('.')[0]),
        nSliceMR, nSliceMALDI, pickMz))
    ionImageOnMR = np.load(ionImageOnMRPath)
    ionImageMRsize[3 * rmin: 3 * (rmax + 1), 3 * cmin: 3 * (cmax + 1)] = ionImageOnMR

    if 'msml2' not in plt.colormaps():
        colors = [
            (0.0, 0.0, 0.0, 0.0),
            (0.2, 0.0, 0.8, 1.0),
            (0.85, 0.0, 0.4, 1.0),
            (1.0, 0.0, 0.0, 1.0)
        ]
        color_bin = 256
        mtl.colormaps.register(LinearSegmentedColormap.from_list(name='msml2', colors=colors, N=color_bin))
    fig, ax = plt.subplots(figsize=[5, 4], dpi=1000)
    ax.imshow(mrSlice_x3, origin='lower', cmap='gray', alpha=1.0, zorder=0)
    ax.imshow(ionImageMRsize, origin='lower', cmap='msml2', alpha=1.0, zorder=1,
              # label=
              )
    pcm = ax.pcolormesh(ionImageMRsize, cmap='msml2')
    fig.colorbar(pcm, orientation='vertical', anchor=(0.0, 1.0), ax=ax, shrink=0.5, pad=0.02)

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
    # plt.savefig(os.path.join(figureDir, '{}_mr{}_ms{}_ionImage_mz_{}_onMR.png'.format(os.path.basename(os.path.normpath(niiPath).split('.')[0]),
    #                                                         nSliceMR, nSliceMALDI[0], pickMz)))
    plt.show()
# mri contrast enhancement and segmentation
if __name__ != '__main__':
    mrBlock = nib.load(niiPath).get_fdata()
    scMaskPath = os.path.join(dataFold, '9_SC_mask.nii.gz')
    scMask = nib.load(scMaskPath).get_fdata()
    rmin, rmax, cmin, cmax = bbox2(scMask)
    mrBlock[np.where(scMask == 0)] = 0
    scBlock = mrBlock[rmin:rmax + 1, cmin:cmax + 1, :]
    perc_ = (5, 99.5)
    vmin, vmax = np.percentile(scBlock, q=perc_)
    clipped_data = rescale_intensity(
        scBlock,
        in_range=(vmin, vmax),
        out_range=(0, 255))
    classes_ = 4
    gamma_val = 1.9
    g2rgb = 128/255
    if 'msml_gray' not in plt.colormaps():
        colors = [
            (0.0, 0.0, 0.0, 0.0),
            # (0.2, 0.0, 0.8, 1.0),
            # (0.85, 0.0, 0.4, 1.0),
            (g2rgb, g2rgb, g2rgb, 1.0)
        ]
        color_bin = 256
        mtl.colormaps.register(LinearSegmentedColormap.from_list(name='msml_gray', colors=colors, N=color_bin))
    if 'msml_list' not in plt.colormaps():
        # colors = [(0.1, 0.1, 0.1), (0.0, 0, 1.0), (0.0, 0.5, 0.5), (0.0, 1.0, 0.0), (0.5, 0.5, 0.0), (1.0, 0, 0.0)]  # Bk -> bl -> G -> r
        colors = [(0.0, 0.0, 0.0, 0.0),
                  (0.9569, 0.7686, 0.1882),
                  # (0.65, 0.71, 0.91),
                  (0.95, 0.3, 0.22),  # (0.0, 0.5, 0.5),
                  # (0.01, 0.01, 0.95, 1.0),
                  # (0.01, 0.95, 0.01, 1.0),  # (0.5, 0.5, 0.0),
                  (0.65, 0.0, 0.95),
                  # (0.95, 0.01, 0.01, 1.0)
                  ]
                  # (1.0, 0, 0.0)]  # Bk -> R -> G -> Bl
        color_bin = 4
        mtl.colormaps.register(LinearSegmentedColormap.from_list(name='msml_list', colors=colors, N=color_bin))
    keywords1 = {'cmap': 'msml_gray', 'origin': 'lower'}
    keywords2 = {'cmap': 'msml_list', 'origin': 'lower'}
    fontkeys = {'fontsize': 20, 'fontweight': 'normal'}
    fig, axes = plt.subplots(scBlock.shape[2], 8, figsize=(25, 10), dpi=400, sharex=False)
    for slice in range(scBlock.shape[2]):
        im = axes[slice, 0].imshow(scBlock[..., slice], **keywords1)
        axes[slice, 0].axis('off')
        divider = make_axes_locatable(axes[slice, 0])
        max = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=max, ax=axes[slice, 0])
        if slice == 0: axes[slice, 0].set_title("original", **fontkeys)

        im = axes[slice, 2].imshow(clipped_data[:, :, slice], **keywords1)
        axes[slice, 2].axis('off')
        divider = make_axes_locatable(axes[slice, 2])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, ax=axes[slice, 2])
        if slice == 0: axes[slice, 2].set_title("clipped", **fontkeys)

        gamma_slice = adjust_gamma(clipped_data[:, :, slice], gamma=gamma_val)
        im = axes[slice, 4].imshow(gamma_slice, **keywords1)
        axes[slice, 4].axis('off')
        divider = make_axes_locatable(axes[slice, 4])
        cax = divider.append_axes('right', size='5%', pad=0.15)
        fig.colorbar(im, cax=cax, ax=axes[slice, 4])
        if slice == 0: axes[slice, 4].set_title("gamma-adjusted", **fontkeys)

        eq_img = equalize_hist(gamma_slice) #clipped_data[:, :, slice])
        clahe_img = equalize_adapthist(eq_img, kernel_size=8)
        im = axes[slice, 6].imshow(clahe_img, **keywords1)
        axes[slice, 6].axis('off')
        divider = make_axes_locatable(axes[slice, 6])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, ax=axes[slice, 6])
        if slice == 0: axes[slice, 6].set_title("clahe", **fontkeys)

        thresholds = filters.threshold_multiotsu(scBlock[:, :, slice], classes=classes_)
        regions = np.digitize(scBlock[:, :, slice], bins=thresholds)
        # regions_labeled = label2rgb(regions)
        im = axes[slice, 1].imshow(regions, **keywords2)
        axes[slice, 1].axis('off')
        divider = make_axes_locatable(axes[slice, 1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, ax=axes[slice, 1])

        thresholds = filters.threshold_multiotsu(clipped_data[:, :, slice], classes=classes_)
        regions = np.digitize(clipped_data[:, :, slice], bins=thresholds)
        # regions_labeled = label2rgb(regions)
        im = axes[slice, 3].imshow(regions, **keywords2)
        axes[slice, 3].axis('off')
        divider = make_axes_locatable(axes[slice, 3])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, ax=axes[slice, 3])

        thresholds = filters.threshold_multiotsu(gamma_slice, classes=classes_)
        regions = np.digitize(gamma_slice, bins=thresholds)
        # regions_labeled = label2rgb(regions)
        im = axes[slice, 5].imshow(regions, **keywords2)
        axes[slice, 5].axis('off')
        divider = make_axes_locatable(axes[slice, 5])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, ax=axes[slice, 5])

        thresholds = filters.threshold_multiotsu(clahe_img, classes=classes_)
        regions = np.digitize(clahe_img, bins=thresholds)
        # regions_labeled = label2rgb(regions)
        im = axes[slice, 7].imshow(regions, **keywords2)
        axes[slice, 7].axis('off')
        divider = make_axes_locatable(axes[slice, 7])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, ax=axes[slice, 7])
    # bin_centers_, img_cdf_ = cumulative_distribution(scBlock[..., 4])
    # plt.plot(img_cdf_, bin_centers_)
    # plt.show()
    #
    # bin_centers_, img_cdf_ = cumulative_distribution(clahe_img)
    # plt.plot(img_cdf_, bin_centers_)
    # plt.show()
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
    # fig.suptitle("{}% clipping, gamma:{}, classes:{}".format(perc_, gamma_val, classes_), fontsize=20,
    #              fontweight='bold')
    fig.tight_layout(pad=1.0)
    # plt.savefig(os.path.join(figureDir, 'mr_preprocessing.png'))
    plt.show()