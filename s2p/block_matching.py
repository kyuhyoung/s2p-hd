# Copyright (C) 2015, Carlo de Franchis <carlo.de-franchis@ens-cachan.fr>
# Copyright (C) 2015, Gabriele Facciolo <facciolo@cmla.ens-cachan.fr>
# Copyright (C) 2015, Enric Meinhardt <enric.meinhardt@cmla.ens-cachan.fr>
# Copyright (C) 2015, Julien Michel <julien.michel@cnes.fr>

import os
import tempfile
import numpy as np
import rasterio
from scipy import ndimage

from s2p import common
from s2p.gpu_memory_manager import GPUMemoryManager
from s2p.specklefilter import specklefilter


class MaxDisparityRangeError(Exception):
    pass


def create_rejection_mask(disp, im1, im2, mask):
    """
    Create rejection mask (0 means rejected, 1 means accepted)
    Keep only the points that are matched and present in both input images

    Args:
        disp: path to the input disparity map
        im1, im2: rectified stereo pair
        mask: path to the output rejection mask
    """
#### old plambda version
#    tmp1 = common.tmpfile('.tif')
#    tmp2 = common.tmpfile('.tif')
#    common.run(["plambda", disp, "x 0 join", "-o", tmp1])
#    common.run(["backflow", tmp1, im2, tmp2])
#    common.run(["plambda", disp, im1, tmp2, "x isfinite y isfinite z isfinite and and vmul", "-o", mask])
####

    im1 = common.rio_read_as_array_with_nans(im1)
    im2 = common.rio_read_as_array_with_nans(im2)
    disp= common.rio_read_as_array_with_nans(disp)

    # ensure 2D (luminance conversion if multi-band)
    im1 = common.to_grayscale(im1)
    im2 = common.to_grayscale(im2)
    if disp.ndim == 3:
        disp = disp[0]

    h, w = disp.shape[:2]
    disp = np.stack( (np.zeros_like(disp), disp), 2)

    disp[:,:,0] += np.arange(h)[:,np.newaxis]
    disp[:,:,1] += np.arange(w)
    m = ndimage.map_coordinates(im2, disp.transpose((2,0,1)) ,order=1, mode='constant', cval=np.nan)
    #m= cv2.remap(im2, disp, None, cv2.INTER_LINEAR) #, cv2.BORDER_CONSTANT, np.nan)   # cv2 alternative
    m = ( np.isfinite(im1) * np.isfinite(m) * np.isfinite(disp[:,:,0]) ).astype(np.uint8)
    common.rasterio_write(mask, m )



def leftright(offL, offR, maxdiff=1):
    '''
    Filters the disparity maps applying the left-right consistency test
        | offR(round(x - offL(x))) + offR(x)| <= maxdiff

    Args:
        offL, offR: numpy arrays containing the Left and Right disparity maps
        maxdiff: threshold for the uniqueness constraint

    Returns:
        numpy array containing the offL disparity map,
        where the rejected pixels are set to np.inf
    '''
    sh = offL.shape
    mnan = np.isnan(offL)
    X, Y = np.meshgrid(range(sh[1]), range(sh[0]))
    offLa = np.nan_to_num(offL,0)
    X = np.round(np.clip(X + offLa, 0, sh[1]-1)).astype(int)
    m = (np.abs(offLa + offR[Y,X] ) > maxdiff )
    out = offL.copy()
    out[m] = np.nan
    out[mnan] = np.nan
    return out






def compute_disparity_map(cfg, im1, im2, disp, mask, algo, disp_min=None,
                          disp_max=None, timeout=600, max_disp_range=None,
                          extra_params='',
                          *,
                          gpu_mem_manager: GPUMemoryManager):
    """
    Runs a block-matching binary on a pair of stereo-rectified images.

    Args:
        im1, im2: rectified stereo pair
        disp: path to the output diparity map
        mask: path to the output rejection mask
        algo: string used to indicate the desired binary. Currently it can be
            one among 'hirschmuller02', 'hirschmuller08',
            'hirschmuller08_laplacian', 'hirschmuller08_cauchy', 'sgbm',
            'msmw', 'tvl1', 'mgm', 'mgm_multi' and 'stereosgm_gpu'
        disp_min: smallest disparity to consider
        disp_max: biggest disparity to consider
        timeout: time in seconds after which the disparity command will
            raise an error if it hasn't returned.
            Only applies to `mgm*` algorithms.
        extra_params: optional string with algorithm-dependent parameters

    Raises:
        MaxDisparityRangeError: if max_disp_range is defined,
            and if the [disp_min, disp_max] range is greater
            than max_disp_range, to avoid endless computation.
    """
    # limit disparity bounds
    if disp_min is not None and disp_max is not None:
        with rasterio.open(im1, "r") as f:
            width = f.width
        if disp_max - disp_min > width:
            center = 0.5 * (disp_min + disp_max)
            disp_min = int(center - 0.5 * width)
            disp_max = int(center + 0.5 * width)

    # round disparity bounds
    if disp_min is not None:
        disp_min = int(np.floor(disp_min))
    if disp_max is not None:
        disp_max = int(np.ceil(disp_max))

    if (
        max_disp_range is not None
        and disp_max - disp_min > max_disp_range
    ):
        raise MaxDisparityRangeError(
            'Disparity range [{}, {}] greater than {}'.format(
                disp_min, disp_max, max_disp_range
            )
        )

    # define environment variables
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(cfg['omp_num_threads'])

    # call the block_matching binary
    if algo == 'hirschmuller02':
        bm_binary = 'subpix.sh'
        common.run('{0} {1} {2} {3} {4} {5} {6} {7}'.format(bm_binary, im1, im2, disp, mask, disp_min,
                                                            disp_max, extra_params))
        # extra_params: LoG(0) regionRadius(3)
        #    LoG: Laplacian of Gaussian preprocess 1:enabled 0:disabled
        #    regionRadius: radius of the window

    if algo == 'hirschmuller08':
        bm_binary = 'callSGBM.sh'
        common.run('{0} {1} {2} {3} {4} {5} {6} {7}'.format(bm_binary, im1, im2, disp, mask, disp_min,
                                                            disp_max, extra_params))
        # extra_params: regionRadius(3) P1(default) P2(default) LRdiff(1)
        #    regionRadius: radius of the window
        #    P1, P2 : regularization parameters
        #    LRdiff: maximum difference between left and right disparity maps

    if algo == 'hirschmuller08_laplacian':
        bm_binary = 'callSGBM_lap.sh'
        common.run('{0} {1} {2} {3} {4} {5} {6} {7}'.format(bm_binary, im1, im2, disp, mask, disp_min,
                                                            disp_max, extra_params))
    if algo == 'hirschmuller08_cauchy':
        bm_binary = 'callSGBM_cauchy.sh'
        common.run('{0} {1} {2} {3} {4} {5} {6} {7}'.format(bm_binary, im1, im2, disp, mask, disp_min,
                                                            disp_max, extra_params))
    if algo == 'sgbm':
        # opencv sgbm function implements a modified version of Hirschmuller's
        # Semi-Global Matching (SGM) algorithm described in "Stereo Processing
        # by Semiglobal Matching and Mutual Information", PAMI, 2008

        p1 = 8  # penalizes disparity changes of 1 between neighbor pixels
        p2 = 32  # penalizes disparity changes of more than 1
        # it is required that p2 > p1. The larger p1, p2, the smoother the disparity

        win = 3  # matched block size. It must be a positive odd number
        lr = 1  # maximum difference allowed in the left-right disparity check
        cost = tempfile.NamedTemporaryFile()
        common.run('sgbm {} {} {} {} {} {} {} {} {} {}'.format(im1, im2,
                                                               disp, cost.name,
                                                               disp_min,
                                                               disp_max,
                                                               win, p1, p2, lr))
        cost.close()

        create_rejection_mask(disp, im1, im2, mask)

    if algo == 'tvl1':
        tvl1 = 'callTVL1.sh'
        common.run('{0} {1} {2} {3} {4}'.format(tvl1, im1, im2, disp, mask),
                   env)

    if algo == 'msmw':
        bm_binary = 'iip_stereo_correlation_multi_win2'
        common.run('{0} -i 1 -n 4 -p 4 -W 5 -x 9 -y 9 -r 1 -d 1 -t -1 -s 0 -b 0 -o 0.25 -f 0 -P 32 -m {1} -M {2} {3} {4} {5} {6}'.format(bm_binary, disp_min, disp_max, im1, im2, disp, mask))

    if algo == 'msmw2':
        bm_binary = 'iip_stereo_correlation_multi_win2_newversion'
        common.run('{0} -i 1 -n 4 -p 4 -W 5 -x 9 -y 9 -r 1 -d 1 -t -1 -s 0 -b 0 -o -0.25 -f 0 -P 32 -D 0 -O 25 -c 0 -m {1} -M {2} {3} {4} {5} {6}'.format(
                bm_binary, disp_min, disp_max, im1, im2, disp, mask), env)

    if algo == 'msmw3':
        bm_binary = 'msmw'
        common.run('{0} -m {1} -M {2} -il {3} -ir {4} -dl {5} -kl {6}'.format(
                bm_binary, disp_min, disp_max, im1, im2, disp, mask))


    if algo == 'stereosgm_gpu':
        census_win = int(cfg['census_ncc_win'])
        specklefilter_surface = int(cfg['stereo_speckle_filter'])
        nb_dir = int(cfg['mgm_nb_directions'])
        # it is required that p2 > p1. The larger p1, p2, the smoother the disparity
        regularity_multiplier = cfg['stereo_regularity_multiplier']

        from s2p import stereosgm_gpu
        i1 = common.rio_read_as_array_with_nans(im1)
        i2 = common.rio_read_as_array_with_nans(im2)

        vram_required = np.prod(np.max([i1.shape, i2.shape], axis=0)) * 514 * 8 / 1024 / 1024

        # code of the census window for the sgm library
        # 0: 5x5, 1: 7x5, 2: 7x7, 3: 9x7
        census_code = 0 if census_win <=5 else 2 if census_win <=7 else 3

        # adjust the regularizer strenght to the size of the census window
        regularity_multiplier *= 0.4 if census_code == 0 else 0.55 if census_code == 1 else 0.77 if census_code == 2 else 1
        P1 = int(10*regularity_multiplier)   # penalizes disparity changes of 1 between neighbor pixels
        P2 = int(40*regularity_multiplier)  # penalizes disparity changes of more than 1

        # 2 scale disparity range estimation
        #  this doesnt really work perfectly
        i1o = ndimage.zoom(np.nan_to_num(i1), .5, mode='reflect')
        i2o = ndimage.zoom(np.nan_to_num(i2), .5, mode='reflect')
        dispmino = (disp_min + disp_max)//2 - 256

        with gpu_mem_manager.request(vram_required):
            resulto = stereosgm_gpu.run(i1o, i2o, nb_dir=nb_dir, disp_min=dispmino, P1=P1, P2=P2)
            # debug
            # common.rasterio_write(disp+'o.tif', resulto)
            disp_min, disp_max, disp_med = np.nanquantile(resulto[20:-20,20:-20], [0.001, 0.999, 0.5])

            result  = stereosgm_gpu.run(i1, i2, nb_dir=nb_dir, disp_min=int( -disp_med*2)-256,P1=P1, P2=P2, census_transform_size=census_code)

        #resultR = stereosgm_gpu.run(i2, i1, nb_dir=nb_dir, disp_min=int( -disp_med*2)-256,P1=P1, P2=P2, census_transform_size=census_code)
        #result = leftright(result, resultR, maxdiff=1)

        # apply spekle filter
        result = specklefilter(result,specklefilter_surface, 5)

        result[0:2,:] = np.nan
        result[-2:,:] = np.nan
        result[:,0:2] = np.nan
        result[:,-2:] = np.nan

        common.rasterio_write(disp, result)

        dispgpu= '{}_gpu.tif'.format(os.path.splitext(disp)[0])
        common.rasterio_write(dispgpu, result)


        ##### POSTPROCESS WITH SUBPIXEL MGM
        if cfg['postprocess_stereosgm_gpu']: 

            env['REMOVESMALLCC']  = str(cfg['stereo_speckle_filter'])
            env['MINDIFF']        = str(cfg['mgm_mindiff_control'])
            env['TESTLRRL']       = str(cfg['mgm_leftright_control'])
            env['TESTLRRL_TAU']   = str(cfg['mgm_leftright_threshold'])
            env['CENSUS_NCC_WIN'] = str(cfg['census_ncc_win'])
            env['SUBPIX'] = '2'
            # it is required that p2 > p1. The larger p1, p2, the smoother the disparity
            regularity_multiplier = cfg['stereo_regularity_multiplier']
    
            nb_dir = cfg['mgm_nb_directions']
    
            P1 = 8*regularity_multiplier   # penalizes disparity changes of 1 between neighbor pixels
            P2 = 32*regularity_multiplier  # penalizes disparity changes of more than 1
            #conf = '{}_confidence.tif'.format(os.path.splitext(disp)[0])   << TODO: currently not working
    
            # feed dmin and dmax 
            disp_min= '{}_dmin.tif'.format(os.path.splitext(disp)[0])
            disp_max= '{}_dmax.tif'.format(os.path.splitext(disp)[0])
            common.rasterio_write(disp_min, result)
            common.rasterio_write(disp_max, result)
    
            common.run(
                '{executable} '
                '-m {disp_min} -M {disp_max} '
                '-r -1 -R 1 '
                '-S -1 '    ## <<< this is the line that makes MGM run only as a refinement.
                '-s vfit '
                '-t census '
                '-O {nb_dir} '
                '-P1 {P1} -P2 {P2} '
                #'-confidence_consensusL {conf} '
                '{im1} {im2} {disp}'.format(
                    executable='mgm_multi',
                    disp_min=disp_min,
                    disp_max=disp_max,
                    nb_dir=nb_dir,
                    P1=P1,
                    P2=P2,
                    #conf=conf,
                    im1=im1,
                    im2=im2,
                    disp=disp,
                ),
                env=env,
                timeout=timeout,
            )
    
    
    
        create_rejection_mask(disp, im1, im2, mask)



    if algo == 'mgm':
        env['MEDIAN'] = '1'
        env['CENSUS_NCC_WIN'] = str(cfg['census_ncc_win'])
        env['TSGM'] = '3'
        env['TESTLRRL']       = str(cfg['mgm_leftright_control'])
        env['TESTLRRL_TAU']   = str(cfg['mgm_leftright_threshold'])
        env['MINDIFF']        = str(cfg['mgm_mindiff_control'])

        nb_dir = cfg['mgm_nb_directions']

        conf = '{}_confidence.tif'.format(os.path.splitext(disp)[0])

        common.run(
            '{executable} '
            '-r {disp_min} -R {disp_max} '
            '-s vfit '
            '-t census '
            '-O {nb_dir} '
            '-confidence_consensusL {conf} '
            '{im1} {im2} {disp}'.format(
                executable='mgm',
                disp_min=disp_min,
                disp_max=disp_max,
                nb_dir=nb_dir,
                conf=conf,
                im1=im1,
                im2=im2,
                disp=disp,
            ),
            env=env,
            timeout=timeout,
        )

        create_rejection_mask(disp, im1, im2, mask)


    if algo == 'mgm_multi':
        env['REMOVESMALLCC']  = str(cfg['stereo_speckle_filter'])
        env['MINDIFF']        = str(cfg['mgm_mindiff_control'])
        env['TESTLRRL']       = str(cfg['mgm_leftright_control'])
        env['TESTLRRL_TAU']   = str(cfg['mgm_leftright_threshold'])
        env['CENSUS_NCC_WIN'] = str(cfg['census_ncc_win'])
        env['SUBPIX'] = '2'
        # it is required that p2 > p1. The larger p1, p2, the smoother the disparity
        regularity_multiplier = cfg['stereo_regularity_multiplier']

        nb_dir = cfg['mgm_nb_directions']

        P1 = 8*regularity_multiplier   # penalizes disparity changes of 1 between neighbor pixels
        P2 = 32*regularity_multiplier  # penalizes disparity changes of more than 1
        conf = '{}_confidence.tif'.format(os.path.splitext(disp)[0])

        common.run(
            '{executable} '
            '-r {disp_min} -R {disp_max} '
            '-S 6 '
            '-s vfit '
            '-t census '
            '-O {nb_dir} '
            '-P1 {P1} -P2 {P2} '
            '-confidence_consensusL {conf} '
            '{im1} {im2} {disp}'.format(
                executable='mgm_multi',
                disp_min=disp_min,
                disp_max=disp_max,
                nb_dir=nb_dir,
                P1=P1,
                P2=P2,
                conf=conf,
                im1=im1,
                im2=im2,
                disp=disp,
            ),
            env=env,
            timeout=timeout,
        )

        create_rejection_mask(disp, im1, im2, mask)
