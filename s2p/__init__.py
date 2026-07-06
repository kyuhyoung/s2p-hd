#!/usr/bin/env python

# s2p - Satellite Stereo Pipeline
# Copyright (C) 2015, Carlo de Franchis <carlo.de-franchis@polytechnique.org>
# Copyright (C) 2015, Gabriele Facciolo <facciolo@cmla.ens-cachan.fr>
# Copyright (C) 2015, Enric Meinhardt <enric.meinhardt@cmla.ens-cachan.fr>
# Copyright (C) 2015, Julien Michel <julien.michel@cnes.fr>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import sys
import os.path
import json
import multiprocessing
import tempfile
import logging
from typing import List
import subprocess

import numpy as np
import rasterio
import rasterio.merge
from plyflatten import plyflatten_from_plyfiles_list

from s2p import common
from s2p import parallel
from s2p import geographiclib
from s2p import initialization
from s2p import pointing_accuracy
from s2p import rectification
from s2p import block_matching
from s2p import dl_stereo
from s2p import masking
from s2p import ply
from s2p import triangulation
from s2p import fusion
from s2p import visualisation
from s2p import config
from s2p import homography
from s2p.tile import Tile
from .gpu_memory_manager import GPUMemoryManager


logger = logging.getLogger(__name__)


def pointing_correction(cfg, tile: Tile, i) -> bool:
    """
    Compute the translation that corrects the pointing error on a pair of tiles.

    Args:
        tile: Tile containing the information needed to process the tile
        i: index of the processed pair
    """
    x, y, w, h = tile.coordinates
    out_dir = os.path.join(tile.dir, 'pair_{}'.format(i))
    img1 = cfg['images'][0]['img']
    rpc1 = cfg['images'][0]['rpcm']
    img2 = cfg['images'][i]['img']
    rpc2 = cfg['images'][i]['rpcm']

    # correct pointing error
    logger.info('correcting pointing on tile {} {} pair {}...'.format(x, y, i))
    method = 'relative' if cfg['relative_sift_match_thresh'] is True else 'absolute'
    try:
        A, m = pointing_accuracy.compute_correction(
            cfg, img1, img2, rpc1, rpc2, x, y, w, h, method,
            cfg['sift_match_thresh'], cfg['max_pointing_error'],
            cfg['n_gcp_per_axis']
        )
        if A is not None:  # A is the correction matrix
            np.savetxt(os.path.join(out_dir, 'pointing.txt'), A, fmt='%6.3f')
        if m is not None:  # m is the list of sift matches
            np.savetxt(os.path.join(out_dir, 'sift_matches.txt'), m, fmt='%9.3f')
            np.savetxt(os.path.join(out_dir, 'center_keypts_sec.txt'),
                       np.mean(m[:, 2:], 0), fmt='%9.3f')
            if cfg['debug']:
                visualisation.plot_matches(cfg, img1, img2, rpc1, rpc2, m,
                                           os.path.join(out_dir,
                                                        'sift_matches_pointing.png'),
                                           x, y, w, h)
        return True  ## success
    except Exception:
        # pointing accuracy can fail because one of the images is empty
        # in this case, we return success = False and the tile will be removed 
        # from the pipeline 
        logger.error('pointing_accuracy.compute_correction has failed: tile: {} {}'.format(*tile.coordinates[0:2]))

        return False ## not success


def global_pointing_correction(cfg, tiles: List[Tile]) -> None:
    """
    Compute the global pointing corrections for each pair of images.

    Args:
        tiles: list of tiles
    """
    for i in range(1, len(cfg['images'])):
        out = os.path.join(cfg['out_dir'], 'global_pointing_pair_%d.txt' % i)
        l = [os.path.join(t.dir, 'pair_%d' % i) for t in tiles]
        np.savetxt(out, pointing_accuracy.global_from_local(l),
                   fmt='%12.6f')
        if cfg['clean_intermediate']:
            for d in l:
                common.remove(os.path.join(d, 'center_keypts_sec.txt'))


# evaluate the epipolar line between two images at a value of h
def epipolar_correspondence(rpc_A, rpc_B, x, y, h):
    lon, lat = rpc_A.localization(x, y, h)
    return rpc_B.projection(lon, lat, h)


def triangulation_iterative(rpc1, rpc2, x1, y1, x2, y2, A=None):
    """
    Triangulate a match between two images.

    Arguments:
        rpc1, rpc2: calibration data for each image
        x1, y1: pixel coordinates in the domain of the first image
        x2, y2: pixel coordinates in the domain of the second image
        A: pointing correction matrix

    Return value: a 4-tuple (lon, lat, h, e)
        lon, lat, h, e: coordinates of the triangulated point, reprojection error
    """

    # apply the pointing correction matrix
    if A is not None:
        tempout = homography.points_apply_homography((A), np.vstack([x2 , y2]).transpose())
        x2 = tempout[:,0]
        y2 = tempout[:,1]

    # initial guess for h
    h = rpc1.alt_offset
    hstep = 1
    err = 0

    # iteratively improve h to minimize the error
    for _ in range(10):
        # two points on the epipolar curve of (x1, y1)
        # are used to approximate it by a straight line
        px, py = epipolar_correspondence(rpc1, rpc2, x1, y1, h)
        qx, qy = epipolar_correspondence(rpc1, rpc2, x1, y1, h + hstep)

        # displacement vectors between these two points and with the target
        ax, ay = qx-px, qy-py
        bx, by = x2-px, y2-py

        # projection of the target into the straight line
        l = (ax*bx + ay*by) / (ax*ax + ay*ay)
        rx, ry = px+l*ax, py+l*ay

        # error of this projection
        err = np.hypot(rx - x2, ry - y2)

        # new value for h
        h = h + l * hstep

        # stop if l becomes too small (max 2 or 3 iterations are performed in practice)
        if np.all(np.fabs(l) < 1e-3):
            break

    lon, lat = rpc1.localization(x1, y1, h)
    return lon, lat, h, err


def refine_matches(rpcA, rpcB, matches, A, max_altitude_span, altitude_margin):

    if (matches is None) or (len(matches) == 0):
        return matches

    lon, lat, alt, err = triangulation_iterative(rpcA, rpcB, matches[:,0] , matches[:,1] , matches[:,2], matches[:,3],(A))

    mmin, mmed, mmax = np.quantile(alt, [.01, .5, .99])
    removed_matches = (alt <= mmed + altitude_margin) & (alt >= mmed - altitude_margin)

    if max(alt) - min(alt) > max_altitude_span:
        logger.info(f"maximum altitude range moved from {max(alt) - min(alt)} to {mmax-mmin}")
        return matches[removed_matches]

    return matches

def rectification_pair(cfg, tile: Tile, i: int) -> bool:
    """
    Rectify a pair of images on a given tile.

    Args:
        tile: Tile containing the information needed to process a tile.
        i: index of the processed pair
    """
    out_dir = os.path.join(tile.dir, 'pair_{}'.format(i))
    x, y, w, h = tile.coordinates
    img1 = cfg['images'][0]['img']
    rpc1 = cfg['images'][0]['rpcm']
    img2 = cfg['images'][i]['img']
    rpc2 = cfg['images'][i]['rpcm']
    pointing = os.path.join(cfg['out_dir'],
                            'global_pointing_pair_{}.txt'.format(i))

    logger.info('rectifying tile {} {} pair {}...'.format(x, y, i))
    try:
        A = np.loadtxt(os.path.join(out_dir, 'pointing.txt'))
    except IOError:
        A = np.loadtxt(pointing)
    try:
        m = np.loadtxt(os.path.join(out_dir, 'sift_matches.txt'))
    except IOError:
        m = None

    cur_dir = os.path.join(tile.dir, 'pair_{}'.format(i))
    for n in tile.neighborhood_dirs:
        nei_dir = os.path.join(tile.dir, n, 'pair_{}'.format(i))
        if os.path.exists(nei_dir) and not os.path.samefile(cur_dir, nei_dir):
            sift_from_neighborhood = os.path.join(nei_dir, 'sift_matches.txt')
            try:
                m_n = np.loadtxt(sift_from_neighborhood)
                # added sifts in the ellipse of semi axes : (3*w/4, 3*h/4)
    #            m_n = m_n[np.where(np.linalg.norm([(m_n[:, 0] - (x + w/2)) / w,
    #                                               (m_n[:, 1] - (y + h/2)) / h],
    #                                              axis=0) < 8/4)]
                if m is None:
                    m = m_n
                else:
                    m = np.concatenate((m, m_n))
            except IOError:
                logger.warning('%s does not exist' % sift_from_neighborhood)

    # remove sift matches that triangulate to points that are extreme
    m = refine_matches(rpc1, rpc2, m, A, cfg['max_altitude_span'], cfg['altitude_margin'])

    rect1 = os.path.join(out_dir, 'rectified_ref.tif')
    rect2 = os.path.join(out_dir, 'rectified_sec.tif')
    H1, H2, disp_min, disp_max, success = rectification.rectify_pair(cfg, img1, img2,
                                                                     rpc1, rpc2,
                                                                     x, y, w, h,
                                                                     rect1, rect2, A, m,
                                                                     method=cfg['rectification_method'],
                                                                     hmargin=cfg['horizontal_margin'],
                                                                     vmargin=cfg['vertical_margin'],
                                                                     pair_idx=i)

    if success:
        np.savetxt(os.path.join(out_dir, 'H_ref.txt'), H1, fmt='%12.6f')
        np.savetxt(os.path.join(out_dir, 'H_sec.txt'), H2, fmt='%12.6f')
        np.savetxt(os.path.join(out_dir, 'disp_min_max.txt'), [disp_min, disp_max],
                   fmt='%3.1f')

    return success


def disparity_range_check(cfg, tile: Tile, i: int):
    """
    Reason about the estimated disparity ranges for all the tiles and update them if needed

    Args:
        tile: dictionary containing the information needed to process a tile.
        i: index of the processed pair
    Returns:
        True if the tile passes the test False otherwise
    """
    out_dir = os.path.join(tile.dir, 'pair_{}'.format(i))
    x, y, w, h = tile.coordinates
    img1 = cfg['images'][0]['img']
    rpc1 = cfg['images'][0]['rpcm']
    img2 = cfg['images'][i]['img']
    rpc2 = cfg['images'][i]['rpcm']
    pointing = os.path.join(cfg['out_dir'],
                            'global_pointing_pair_{}.txt'.format(i))

    disp_min, disp_max =  np.loadtxt(os.path.join(out_dir, 'disp_min_max.txt'))

    try:
        A = np.loadtxt(os.path.join(out_dir, 'pointing.txt'))
    except IOError:
        A = np.loadtxt(pointing)
    try:
        m = np.loadtxt(os.path.join(out_dir, 'sift_matches.txt'))
    except IOError:
        m = None

    # TODO: reason about the disparity range of the current tile based on the range of neighboring tiles
    if disp_max - disp_min > 100:
        logger.info('checking tile {} {} pair {}... {} {}'.format(x, y, i, disp_min, disp_max))


    for n in tile.neighborhood_dirs:
        nei_dir = os.path.join(tile.dir, n, 'pair_{}'.format(i))
        if os.path.exists(nei_dir) and not os.path.samefile(out_dir, nei_dir):
            sift_from_neighborhood = os.path.join(nei_dir, 'sift_matches.txt')
            dmin_dmax_from_neighborhood = os.path.join(nei_dir, 'disp_min_max.txt')
            # TODO continue this

    # This is a very simple heuristic test. If the disparity range is > 512 something is wrong with the tile
    # This test is rendered useless by the addition of the refine_matches function
    if disp_max-disp_min > w/2:
        return True
    else:
        return True


def stereo_matching(cfg, tile: Tile, i: int, gpu_mem_manager: GPUMemoryManager) -> None:
    """
    Compute the disparity of a pair of images on a given tile.

    Args:
        tile: Tile containing the information needed to process a tile.
        i: index of the processed pair
    """
    out_dir = os.path.join(tile.dir, 'pair_{}'.format(i))
    x, y = tile.coordinates[:2]

    logger.info('estimating disparity on tile {} {} pair {}...'.format(x, y, i))
    rect1 = os.path.join(out_dir, 'rectified_ref.tif')
    rect2 = os.path.join(out_dir, 'rectified_sec.tif')
    disp = os.path.join(out_dir, 'rectified_disp.tif')
    mask = os.path.join(out_dir, 'rectified_mask.png')
    disp_min, disp_max = np.loadtxt(os.path.join(out_dir, 'disp_min_max.txt'))

    try:
        if cfg['matching_algorithm'] == 'dl_stereo':
            # Deep learning stereo matcher
            model = dl_stereo.load_model(cfg)
            dl_stereo.compute_disparity_map(cfg, rect1, rect2, disp, mask,
                                            model, gpu_mem_manager=gpu_mem_manager)
        else:
            # Classical block matching (SGM/MGM etc.)
            block_matching.compute_disparity_map(cfg, rect1, rect2, disp, mask,
                                                 cfg['matching_algorithm'], disp_min,
                                                 disp_max, timeout=cfg['mgm_timeout'],
                                                 max_disp_range=cfg['max_disp_range'],
                                                 gpu_mem_manager=gpu_mem_manager)

        # add margin around masked pixels
        masking.erosion(mask, mask, cfg['msk_erosion'])
    except Exception:
        # in case of timeout we should take note
        # TODO: take note of the failed block matching
        logger.exception('stereo matching has failed:')

    if cfg['clean_intermediate']:
        if len(cfg['images']) > 2:
            common.remove(rect1)
        common.remove(rect2)
#        common.remove(os.path.join(out_dir, 'disp_min_max.txt'))


def disparity_to_height(cfg, tile: Tile, i: int) -> None:
    """
    Compute a height map from the disparity map of a pair of image tiles.

    Args:
        tile: Tile containing the information needed to process a tile.
        i: index of the processed pair.
    """
    out_dir = os.path.join(tile.dir, 'pair_{}'.format(i))
    x, y, w, h = tile.coordinates

    logger.info('triangulating tile {} {} pair {}...'.format(x, y, i))
    rpc1 = cfg['images'][0]['rpcm']
    rpc2 = cfg['images'][i]['rpcm']
    H_ref = np.loadtxt(os.path.join(out_dir, 'H_ref.txt'))
    H_sec = np.loadtxt(os.path.join(out_dir, 'H_sec.txt'))
    disp = os.path.join(out_dir, 'rectified_disp.tif')
    mask = os.path.join(out_dir, 'rectified_mask.png')
    mask_orig = os.path.join(tile.dir, 'mask.tif')
    pointing = os.path.join(cfg['out_dir'],
                            'global_pointing_pair_{}.txt'.format(i))

    with rasterio.open(disp, 'r') as f:
        disp_img = f.read().squeeze()
    with rasterio.open(mask, 'r') as f:
        mask_rect_img = f.read().squeeze()
    with rasterio.open(mask_orig, 'r') as f:
        mask_orig_img = f.read().squeeze()
    # DL-stereo overlap blending (3-image path): compute the height map over
    # (tile + margin) so adjacent tiles OVERLAP by 2*margin. plys_to_dsm then
    # averages that overlap, so the tile boundary is filled by both tiles'
    # reliable interiors instead of a hard join over their (border-trimmed)
    # edges -- which is what produced localized boundary bumps. Mirrors the
    # 2-image disparity_to_ply path.
    hm = cfg.get('horizontal_margin', 0) if cfg.get('dl_overlap_blend', False) else 0
    vm = cfg.get('vertical_margin', 0) if cfg.get('dl_overlap_blend', False) else 0
    if hm or vm:
        mask_orig_img = np.pad(mask_orig_img, ((vm, vm), (hm, hm)),
                               constant_values=1)
    height_map = triangulation.height_map(x - hm, y - vm, w + 2 * hm, h + 2 * vm,
                                          rpc1, rpc2, H_ref, H_sec,
                                          disp_img, mask_rect_img,
                                          mask_orig_img,
                                          A=np.loadtxt(pointing))

    # write height map to a file
    common.rasterio_write(os.path.join(out_dir, 'height_map.tif'), height_map)

    if cfg['clean_intermediate']:
        common.remove(H_ref)
        common.remove(H_sec)
        common.remove(disp)
        common.remove(mask)


def disparity_to_ply(cfg, tile: Tile) -> None:
    """
    Compute a point cloud from the disparity map of a pair of image tiles.

    This function is called by s2p.main only if there are two input images (not
    three).

    Args:
        tile: Tile containing the information needed to process a tile.
    """
    out_dir = tile.dir
    ply_file = os.path.join(out_dir, 'cloud.ply')
    x, y, w, h = tile.coordinates
    rpc1 = cfg['images'][0]['rpcm']
    rpc2 = cfg['images'][1]['rpcm']

    logger.info('triangulating tile {} {}...'.format(x, y))
    H_ref = os.path.join(out_dir, 'pair_1', 'H_ref.txt')
    H_sec = os.path.join(out_dir, 'pair_1', 'H_sec.txt')
    pointing = os.path.join(cfg['out_dir'], 'global_pointing_pair_1.txt')
    disp = os.path.join(out_dir, 'pair_1', 'rectified_disp.tif')
    extra = os.path.join(out_dir, 'pair_1', 'rectified_disp_confidence.tif')
    if not os.path.exists(extra):    # confidence file not always generated
        extra = ''
    mask_rect = os.path.join(out_dir, 'pair_1', 'rectified_mask.png')
    mask_orig = os.path.join(out_dir, 'mask.tif')

    # first check if disp exists for this tile
    if os.path.exists(disp) is False:
        #TODO: take note of the missing tile and move to the next
        logger.error(f'input file: {disp}')
        return

    # prepare the image needed to colorize point cloud
    if cfg['images'][0]['clr']:
        # we want colors image and rectified_ref.tif to have the same size
        with rasterio.open(os.path.join(out_dir, 'pair_1', 'rectified_ref.tif')) as f:
            ww, hh = f.width, f.height

        colors_path = tempfile.NamedTemporaryFile()
        common.image_apply_homography(colors_path.name, cfg['images'][0]['clr'],
                                      np.loadtxt(H_ref), ww, hh)
        with rasterio.open(colors_path.name, "r") as f:
            colors = f.read()
        colors_path.close()

    else:
        with rasterio.open(os.path.join(out_dir, 'pair_1', 'rectified_ref.tif')) as f:
            img = f.read()
        colors = common.linear_stretching_and_quantization_8bit(img)

    # compute the point cloud
    with rasterio.open(disp, 'r') as f:
        disp_img = f.read().squeeze()
    with rasterio.open(mask_rect, 'r') as f:
        mask_rect_img = f.read().squeeze()
    with rasterio.open(mask_orig, 'r') as f:
        mask_orig_img = f.read().squeeze()

    # DL-stereo overlap blending: emit points over the rectified area that
    # corresponds to (tile + margin) so that adjacent tiles overlap by 2*margin.
    # plyflatten then Gaussian-averages the overlap in plys_to_dsm, smoothing
    # the tile-boundary seam. Per-tile H is preserved.
    img_bbx = (x, x+w, y, y+h)
    mask_for_c = mask_orig_img
    if cfg.get('dl_overlap_blend', False):
        hm = cfg.get('horizontal_margin', 0)
        vm = cfg.get('vertical_margin', 0)
        if hm > 0 or vm > 0:
            img_bbx = (x-hm, x+w+hm, y-vm, y+h+vm)
            mask_for_c = np.pad(mask_orig_img,
                                ((vm, vm), (hm, hm)),
                                constant_values=1)

    out_crs = geographiclib.pyproj_crs(cfg['out_crs'])
    xyz_array, err = triangulation.disp_to_xyz(rpc1, rpc2,
                                               np.loadtxt(H_ref), np.loadtxt(H_sec),
                                               disp_img, mask_rect_img,
                                               img_bbx=img_bbx,
                                               mask_orig=mask_for_c,
                                               A=np.loadtxt(pointing),
                                               out_crs=out_crs)

    # 3D filtering
    gsd_radius = cfg['3d_filtering_radius_gsd']
    fillfactor = cfg['3d_filtering_fill_factor']
    valid_in = np.sum(np.all(np.isfinite(xyz_array.reshape(-1, 3)), axis=1))
    if gsd_radius  and  fillfactor:
        r = gsd_radius * cfg['gsd']    # compute radius in meters
        n = int(fillfactor * 2*3.14*gsd_radius**2)  # fraction of the disk 
        triangulation.filter_xyz(xyz_array, r, n, cfg['gsd'])

    # check result
    valid_out = np.sum(np.all(np.isfinite(xyz_array.reshape(-1, 3)), axis=1))
    if valid_out < valid_in//10:
        logger.warning("triangulation.filter_xyz with params {} has conserved only {} out of {}".format((r, n, cfg['gsd']), valid_out, valid_in))

    proj_com = "CRS {}".format(cfg['out_crs'])
    try:
        triangulation.write_to_ply(ply_file, xyz_array, colors, proj_com, confidence=extra)
    except Exception:
        logger.error('triangulation.write_to_ply has failed: tile: {} {}'.format(*tile.coordinates[0:2]))


    if cfg['clean_intermediate']:
        common.remove(H_ref)
        common.remove(H_sec)
        common.remove(disp)
        common.remove(mask_rect)
        common.remove(mask_orig)
        common.remove(os.path.join(out_dir, 'pair_1', 'rectified_ref.tif'))


def mean_heights(cfg, tile: Tile) -> None:
    n = len(cfg['images']) - 1
    # Read each pair's height map at its actual size. With dl_overlap_blend the
    # per-pair height maps cover (tile + margin), so we must not assume the bare
    # (w, h) tile shape here.
    paths = [os.path.join(tile.dir, 'pair_{}'.format(i + 1), 'height_map.tif')
             for i in range(n)]
    shp = None
    for p in paths:
        if os.path.exists(p):
            with rasterio.open(p) as f:
                shp = (f.height, f.width)
            break
    if shp is None:        # no pair produced a height map for this tile
        return
    maps = np.full((shp[0], shp[1], n), np.nan)
    for i, p in enumerate(paths):
        try:
            with rasterio.open(p, 'r') as f:
                maps[:, :, i] = f.read(1)
        except RuntimeError:  # the file is not there
            pass

    validity_mask = maps.sum(axis=2)  # sum to propagate nan values
    validity_mask += 1 - validity_mask  # 1 on valid pixels, and nan on invalid

    # save the n mean height values to a txt file in the tile directory.
    # nanmedian, NOT nanmean: these run on the RAW pair height maps (before
    # cargarse_basura), where water/forest garbage differs per pair. A mean
    # lets that garbage skew each pair's offset differently (Daejeon lower
    # full: 10.9 m spurious pair offset vs 0.55 m true), and merge_n then
    # registers the pairs apart so average_if_close discards most pixels.
    np.savetxt(os.path.join(tile.dir, 'local_mean_heights.txt'),
               [np.nanmedian(validity_mask * maps[:, :, i]) for i in range(n)])


def global_mean_heights(cfg, tiles: List[Tile]) -> None:
    # Tiles whose matching produced no valid points never wrote
    # local_mean_heights.txt; skip them instead of crashing (common when the
    # ROI/full image extends past the stereo overlap, e.g. edge/water tiles).
    local_mean_heights = []
    for t in tiles:
        p = os.path.join(t.dir, 'local_mean_heights.txt')
        if os.path.exists(p):
            local_mean_heights.append(np.loadtxt(p))
    if not local_mean_heights:
        raise RuntimeError("no tile produced local_mean_heights.txt (no valid stereo)")
    # median across tiles for the same robustness reason as the local step:
    # tiles where one pair matched garbage (water/forest) must not drag the
    # per-pair global offset away from the other pair's.
    global_mean_heights = np.nanmedian(local_mean_heights, axis=0)
    for i in range(len(cfg['images']) - 1):
        np.savetxt(os.path.join(cfg['out_dir'],
                                'global_mean_height_pair_{}.txt'.format(i+1)),
                   [global_mean_heights[i]])


def align_tile_heights(cfg, tiles: List[Tile]) -> None:
    """
    Inter-tile height registration.

    s2p tiles the ROI and reconstructs each tile independently; the 5b/5c
    "pairwise height offset" steps only register the image PAIRS within a tile,
    never adjacent TILES to each other. With per-tile rectification each tile
    therefore has its own small absolute-height bias, which appears as a step
    (seam) at every tile boundary in the merged DSM.

    Root fix: measure the median height step across each shared tile edge
    (difference of the two adjacent boundary lines, which cancels the terrain
    along the edge), then solve a single global least-squares for one constant
    offset per tile (gauge: mean offset = 0) and subtract it from each tile's
    height map. Rectification is untouched, so per-tile sharpness is preserved
    while the boundaries become continuous.
    """
    coords = [t.coordinates for t in tiles]            # (x, y, w, h) per tile
    pos = {(c[0], c[1]): i for i, c in enumerate(coords)}
    paths = [os.path.join(t.dir, 'height_map.tif') for t in tiles]

    arrs = []
    for p in paths:
        if os.path.exists(p):
            with rasterio.open(p) as f:
                arrs.append(f.read(1).astype(np.float64))
        else:
            arrs.append(None)

    # With dl_overlap_blend the height maps cover (tile + margin), so adjacent
    # tiles overlap by 2*margin and we measure the offset over that shared
    # region (identical ground points). Without overlap we fall back to the two
    # adjacent boundary lines.
    hm = cfg.get('horizontal_margin', 0) if cfg.get('dl_overlap_blend', False) else 0
    vm = cfg.get('vertical_margin', 0) if cfg.get('dl_overlap_blend', False) else 0

    # PLANAR per-tile correction. A single constant offset cannot remove a
    # seam whose step VARIES along the shared edge (a relative tilt between
    # tiles, which cross-date illumination/matching differences produce). So
    # each tile gets a plane  p_i(u,v) = a_i + b_i*u + c_i*v  (u,v normalized
    # to [-0.5, 0.5] over the tile). We sample several points along every
    # shared edge, require the two tiles to agree there after subtracting their
    # planes, and solve one global least-squares for all (a,b,c) with a gauge
    # (mean a = mean b = mean c = 0, i.e. no global plane is invented).
    n = len(tiles)
    NP = 3            # params per tile: offset, tilt-u, tilt-v
    K = 16           # samples along each shared edge
    BAND = 3         # half-width of the median window at each sample

    def uv(shape, col, row):
        H_, W_ = shape
        return (col + 0.5) / W_ - 0.5, (row + 0.5) / H_ - 0.5

    rows, ds = [], []
    nedge = 0
    for i, (x, y, w, h) in enumerate(coords):
        A = arrs[i]
        if A is None:
            continue
        Ha, Wa = A.shape
        # right neighbour at (x+w, y): A's right edge vs B's left edge
        j = pos.get((x + w, y))
        if j is not None and arrs[j] is not None:
            B = arrs[j]; Hb, Wb = B.shape; mrow = min(Ha, Hb); nedge += 1
            for k in range(K):
                r = int((k + 0.5) / K * mrow)
                av = np.nanmedian(A[max(0, r - BAND):r + BAND + 1, -2 * max(hm, 1):]) if hm else \
                     np.nanmedian(A[max(0, r - BAND):r + BAND + 1, -BAND:])
                bv = np.nanmedian(B[max(0, r - BAND):r + BAND + 1, :2 * hm]) if hm else \
                     np.nanmedian(B[max(0, r - BAND):r + BAND + 1, :BAND])
                if not (np.isfinite(av) and np.isfinite(bv)):
                    continue
                uA, vA = uv((Ha, Wa), Wa - 1, r); uB, vB = uv((Hb, Wb), 0, r)
                eq = np.zeros(NP * n)
                eq[NP*j] += 1; eq[NP*j+1] += uB; eq[NP*j+2] += vB
                eq[NP*i] -= 1; eq[NP*i+1] -= uA; eq[NP*i+2] -= vA
                rows.append(eq); ds.append(float(bv - av))   # planeB-planeA = B-A
        # bottom neighbour at (x, y+h): A's bottom edge vs B's top edge
        j = pos.get((x, y + h))
        if j is not None and arrs[j] is not None:
            B = arrs[j]; Hb, Wb = B.shape; mcol = min(Wa, Wb); nedge += 1
            for k in range(K):
                c = int((k + 0.5) / K * mcol)
                av = np.nanmedian(A[-2 * vm:, max(0, c - BAND):c + BAND + 1]) if vm else \
                     np.nanmedian(A[-BAND:, max(0, c - BAND):c + BAND + 1])
                bv = np.nanmedian(B[:2 * vm, max(0, c - BAND):c + BAND + 1]) if vm else \
                     np.nanmedian(B[:BAND, max(0, c - BAND):c + BAND + 1])
                if not (np.isfinite(av) and np.isfinite(bv)):
                    continue
                uA, vA = uv((Ha, Wa), c, Ha - 1); uB, vB = uv((Hb, Wb), c, 0)
                eq = np.zeros(NP * n)
                eq[NP*j] += 1; eq[NP*j+1] += uB; eq[NP*j+2] += vB
                eq[NP*i] -= 1; eq[NP*i+1] -= uA; eq[NP*i+2] -= vA
                rows.append(eq); ds.append(float(bv - av))

    if rows:
        for pp in range(NP):                              # gauge: mean of each param = 0
            g = np.zeros(NP * n); g[pp::NP] = 1.0
            rows.append(g); ds.append(0.0)
        sol, *_ = np.linalg.lstsq(np.asarray(rows), np.asarray(ds), rcond=None)
        params = sol.reshape(n, NP)
    else:
        params = np.zeros((n, NP))

    logger.info('inter-tile PLANAR alignment: %d edges, %d samples | '
                'offset std %.3f m, tilt-u std %.3f, tilt-v std %.3f',
                nedge, len(ds) - NP if rows else 0,
                float(np.std(params[:, 0])), float(np.std(params[:, 1])),
                float(np.std(params[:, 2])))

    # subtract each tile's correction plane from its height map.
    for i, p in enumerate(paths):
        if arrs[i] is None or not np.all(np.isfinite(params[i])) or not np.any(params[i]):
            continue
        a_, b_, c_ = params[i]
        with rasterio.open(p) as f:
            prof = f.profile; data = f.read(1)
        H_, W_ = data.shape
        u = (np.arange(W_) + 0.5) / W_ - 0.5
        v = (np.arange(H_) + 0.5) / H_ - 0.5
        plane = (a_ + b_ * u[None, :] + c_ * v[:, None]).astype(np.float32)
        with rasterio.open(p, 'w', **prof) as f:
            f.write(data - plane, 1)


def heights_fusion(cfg, tile: Tile) -> None:
    """
    Merge the height maps computed for each image pair and generate a ply cloud.

    Args:
        tile: Tile that provides all you need to process a tile
    """
    tile_dir = tile.dir
    height_maps = [os.path.join(tile_dir, 'pair_%d' % (i + 1), 'height_map.tif')
                   for i in range(len(cfg['images']) - 1)]

    # empty tile (no pair produced a height map, e.g. outside stereo overlap):
    # nothing to fuse, skip so downstream (which already tolerates a missing
    # tile height_map.tif / cloud.ply) just leaves a gap here.
    height_maps = [h for h in height_maps if os.path.exists(h)]
    if not height_maps:
        return

    # remove spurious matches
    if cfg['cargarse_basura']:
        for img in height_maps:
            common.cargarse_basura(img, img)

    # load global mean heights
    if cfg.get('fusion_vertical_registration', False):
        global_mean_heights = []
        for i in range(len(cfg['images']) - 1):
            x = np.loadtxt(os.path.join(cfg['out_dir'],
                                        'global_mean_height_pair_{}.txt'.format(i+1)))
            global_mean_heights.append(x)
    else:
        # BA'd inputs: the true per-pair vertical bias is sub-metre (Daejeon
        # urban pixelwise: 0.13-0.55 m), while ESTIMATING it from per-pair
        # marginal stats gets poisoned by canopy/water mask differences
        # (Daejeon lower full: 6-11 m spurious offset, which shifts the
        # average_if_close band off the data and discards most good pixels).
        # Trust the BA and skip vertical registration.
        global_mean_heights = [0.0] * (len(cfg['images']) - 1)

    # merge the height maps (applying mean offset to register)
    fusion.merge_n(os.path.join(tile_dir, 'height_map.tif'), height_maps,
                   global_mean_heights, averaging=cfg['fusion_operator'],
                   threshold=cfg['fusion_thresh'], debug=cfg['debug'])

    if cfg['clean_intermediate']:
        for f in height_maps:
            common.remove(f)


def heights_to_ply(cfg, tile: Tile) -> None:
    """
    Generate a ply cloud.

    Args:
        tile: a Tile that provides all you need to process a tile
    """
    # compute a ply from the merged height map
    out_dir = tile.dir
    x, y, w, h = tile.coordinates
    plyfile = os.path.join(out_dir, 'cloud.ply')
    height_map = os.path.join(out_dir, 'height_map.tif')

    # The merged + inter-tile-aligned height map is normally produced by the
    # separate 5d/5e passes. Fall back to fusing here if it is missing (e.g.
    # heights_to_ply invoked standalone).
    if not os.path.exists(height_map):
        heights_fusion(cfg, tile)

    # empty tile: heights_fusion produced no merged height_map.tif; no cloud.
    # downstream plys_to_dsm already tolerates a missing cloud.ply.
    if not os.path.exists(height_map):
        return

    # Overlap blending: the merged height map covers (tile + margin), so emit
    # the cloud over the same (x-hm .. x+w+hm, y-vm .. y+h+vm) window. Adjacent
    # tiles then overlap by 2*margin and plys_to_dsm averages it (seamless).
    hm = cfg.get('horizontal_margin', 0) if cfg.get('dl_overlap_blend', False) else 0
    vm = cfg.get('vertical_margin', 0) if cfg.get('dl_overlap_blend', False) else 0
    cx, cy, cw, ch = x - hm, y - vm, w + 2 * hm, h + 2 * vm

    if cfg['images'][0]['clr']:
        with rasterio.open(cfg['images'][0]['clr'], "r") as f:
            colors = f.read(window=((cy, cy + ch), (cx, cx + cw)),
                            boundless=True, fill_value=0)
    else:
        with rasterio.open(cfg['images'][0]['img'], "r") as f:
            colors = f.read(window=((cy, cy + ch), (cx, cx + cw)),
                            boundless=True, fill_value=0)

        colors = common.linear_stretching_and_quantization_8bit(colors)

    out_crs = geographiclib.pyproj_crs(cfg['out_crs'])
    xyz_array = triangulation.height_map_to_xyz(height_map,
                                                cfg['images'][0]['rpcm'], cx, cy,
                                                out_crs)

    # 3D filtering
    gsd_radius = cfg['3d_filtering_radius_gsd']
    fillfactor = cfg['3d_filtering_fill_factor']
    if gsd_radius  and  fillfactor:
        r = gsd_radius * cfg['gsd']    # compute radius in meters
        n = int(fillfactor * 2*3.14*gsd_radius**2)  # fraction of the disk 
        triangulation.filter_xyz(xyz_array, r, n, cfg['gsd'])


    proj_com = "CRS {}".format(cfg['out_crs'])
    triangulation.write_to_ply(plyfile, xyz_array, colors, proj_com)

    if cfg['clean_intermediate']:
        common.remove(height_map)
        common.remove(os.path.join(out_dir, 'mask.tif'))


def plys_to_dsm(cfg, tile: Tile) -> None:
    """
    Generates DSM from plyfiles (cloud.ply)

    Args:
        tile: a dictionary that provides all you need to process a tile
    """

    ply_name = 'cloud.ply'

    out_dsm = os.path.join(tile.dir, 'dsm.tif')
    out_conf = os.path.join(tile.dir, 'confidence.tif')
    out_dsm_filtered = os.path.join(tile.dir, 'dsm-filtered.tif')
    r = cfg['dsm_resolution']

    in_ply = os.path.join(tile.dir, ply_name)
    # first check if ply exists (it might not exist because of a failed blockmatching)
    if not os.path.exists(in_ply):
        # TODO: take note of the missing part of the DSM
        logger.error(f'missing input file: {in_ply}')
        return

    # compute the point cloud x, y bounds
    points, _ = ply.read_3d_point_cloud_from_ply(in_ply)
    if len(points) == 0:
        # TODO: take note of the missing part of the DSM
        logger.error(f'plys_to_dsm no points in file: {in_ply}')
        return

    xmin, ymin, *_ = np.min(points, axis=0)
    xmax, ymax, *_ = np.max(points, axis=0)

    # compute xoff, yoff, xsize, ysize on a grid of unit r
    xoff = np.floor(xmin / r) * r
    xsize = int(1 + np.floor((xmax - xoff) / r))

    yoff = np.ceil(ymax / r) * r
    ysize = int(1 - np.floor((ymin - yoff) / r))

    roi = xoff, yoff, xsize, ysize

    # since some tiles might have failed we test for the neighborhood tiles before feeding them to merge
    clouds = []
    for n_dir in tile.neighborhood_dirs:
        nply = os.path.join(tile.dir, n_dir, ply_name)
        if os.path.exists(nply):
            clouds.append(nply)

    # this option controls the type of aggregation
    # TODO: this interface is VERY VERY ugly AND FRAGILE and will be reworked within a new plyflatten
    use_max_aggregation = cfg['dsm_aggregation_with_max']
    # NOTE: keep MAX aggregation WITHIN each tile (rooftops stay crisp) even
    # under dl_overlap_blend. The inter-tile blending is now done by the
    # distance-feathered global merge (merge_tiles_feather), which weights each
    # tile by distance to its border so the reliable interior wins over the
    # unreliable edge -- no equal-average corner artifacts.
    raster, profile = plyflatten_from_plyfiles_list(clouds,
                                                    resolution=r,
                                                    roi=roi,
                                                    radius=cfg['dsm_radius'],
                                                    sigma=cfg['dsm_sigma'],
                                                    amax=use_max_aggregation
                                                    )

    # save output image with utm georeferencing
    if use_max_aggregation:
        # the raster channel where the max is stored is #5 or #4 depending on the presence of the confidence
        if (raster.shape[-1] % 5) == 0:
            dsm = raster[:, :, 5]
        else: 
            dsm = raster[:, :, 4]
    else:
        # the average raster is stored in #0
        dsm = raster[:, :, 0]

    common.rasterio_write(out_dsm, dsm, profile=profile)

    # export confidence (optional)
    # note that the plys are assumed to contain the fields:
    # [x(float32), y(float32), z(float32), r(uint8), g(uint8), b(uint8), confidence(optional, float32)]
    # so the raster has 4 or 5 columns: [z, r, g, b, confidence (optional)]
    if raster.shape[-1] == 5:
        common.rasterio_write(out_conf, raster[:, :, 4], profile=profile)



    # fill the small gaps in the dsm
    if maxsize := cfg['fill_dsm_holes_smaller_than']:
        import s2p.demtk
        from s2p.specklefilter import specklefilter

        # compute the mask where the interpolation will not be applied
        # (masked_nans is a mask of large connected components)
        z = np.isnan(dsm).astype(np.float32)
        z[z == 0] = np.nan
        masked_nans = specklefilter(z, maxsize, 0) == 1

        # apply the interpolation after removing the masked areas
        dsm[masked_nans] = -1000
        filtered = s2p.demtk.descending_neumann_interpolation(dsm).astype(np.float32)
        filtered[masked_nans] = np.nan

        common.rasterio_write(out_dsm_filtered, filtered, profile=profile)


def merge_tiles_rasterio(paths, bounds, res, dst_path, creation_options, method):
    """
    Merge a list of raster tiles into a single raster.

    Parameters:
        paths (list of str): List of file paths to the input raster tiles to be merged.
        bounds (tuple or None): Bounding box (left, bottom, right, top) to limit the merge.
                                If None, bounds are inferred from the input tiles.
        res (float or tuple): Output resolution (pixel size). Can be a single float or (xres, yres).
        dst_path (str): Save path to the output merged raster file.
        creation_options (dict): GDAL creation options (e.g., compression, tiling, block size).
        method (str): Merge method to use when overlapping pixels exist.

    Returns:
        None. The merged raster is written to `dst_path`.
    """

    rasterio.merge.merge(paths,
                         bounds=bounds,
                         res=res,
                         nodata=np.nan,
                         indexes=[1],
                         dst_path=dst_path,
                         dst_kwds=creation_options,
                         method=method)
    return


def merge_with_gdalwarp(input_files, output_file, nb_workers, nodata=np.nan):
    """
    Merge multiple raster files using gdalwarp with maximum value resampling.

    Parameters:
        input_files (list of str): List of file paths to raster files to be merged.
        output_file (str): Save path to the output merged raster file.
        nb_workers (int): Number of threads to use for processing.
        nodata (float, optional): NoData value to consider during merge (default: np.nan).

    Returns:
        None. The result is saved to `output_file`.
    """
    os.environ['GDAL_NUM_THREADS'] = str(nb_workers) # allow GDAL more than one thread to allow gdalwarp to multiprocess

    if np.isnan(nodata):
        nodata_str = "nan"
    else:
        nodata_str = str(nodata)

    nb_workers = min(nb_workers, len(input_files))

    cmd = [
        "gdalwarp",
        "-q",
        "-wo", f"NUM_THREADS={nb_workers}",
        "-r", "max",
        "-wo", "UNIFIED_SRC_NODATA=YES",
        "-srcnodata", nodata_str,
        "-dstnodata", nodata_str,
        *input_files,
        output_file
    ]
    subprocess.run(cmd, check=True)
    os.environ['GDAL_NUM_THREADS'] = "1"


def merge_tiles_mp(nb_workers, global_dst_path, save_folder,
                   paths, bounds, res, creation_options, method,
                   remove_merged=True):
    """
    Merge a large number of DSM tiles in parallel using Rasterio and GDAL.

    This function splits the input tile list into subsets, merges each subset in parallel
    using Rasterio, and finally merges the intermediate results using `gdalwarp` with
    multi-threading.

    Parameters:
        nb_workers (int): Number of worker processes to run in parallel.
        global_dst_path (str): Path to save the final merged DSM.
        save_folder (str): Directory where intermediate merged tiles are saved.
        paths (list of str): List of paths to input DSM tile files.
        bounds (tuple or None): Bounding box for the merge (left, bottom, right, top).
                                If None, bounds are inferred from the input tiles.
        res (float or tuple): Output resolution (pixel size). Can be a single float or (xres, yres).
        creation_options (dict): GDAL creation options for output files.
        method (str): Resampling method used in rasterio.merge.
        remove_merged (bool, optional): If True (default), deletes intermediate merged tiles
                                        after the final merge.

    Returns:
        None. The final merged DSM is saved to `global_dst_path`.
    """

    tiles_per_process = int(np.ceil(len(paths) / nb_workers))

    if tiles_per_process <= 4:
        # if there are only few tiles, it is faster to run everything at once
        # merge_with_gdalwarp(paths, global_dst_path, nb_workers, nodata=np.nan)
        merge_tiles_rasterio(paths, bounds, res, global_dst_path, creation_options, method)
        return

    context = multiprocessing.get_context("spawn")
    pool = context.Pool(nb_workers)

    list_paths_to_merge = [
        paths[i*tiles_per_process: (i+1)*tiles_per_process] for i in range(nb_workers)
                                                            if i*tiles_per_process < len(paths) # only consider non-empty subsets of tiles
    ]
    dst_paths = [
        os.path.join(save_folder, f"merge_{i}.tif") for i in range(len(list_paths_to_merge))
    ]

    results = []

    for (paths_to_merge, dst_path) in zip(list_paths_to_merge, dst_paths):
        results.append(pool.apply_async(
            merge_tiles_rasterio, args=(paths_to_merge, bounds, res, dst_path, creation_options, method)
        ))

    pool.close()
    pool.join()

    merge_with_gdalwarp(dst_paths, global_dst_path, nb_workers, nodata=np.nan)

    if remove_merged:
        for dst_path in dst_paths:
            os.remove(dst_path)

    return


def merge_tiles_feather(paths, bounds, res, dst_path, feather_radius=48):
    """
    Mosaic tile DSMs with distance-to-edge FEATHERING instead of a hard
    max/first pick.

    Each per-tile DSM is unreliable near its border (neural aperture /
    dl_border_trim), and the default 'max' merge picks whichever tile is
    higher in the small inter-tile overlap -- so a slightly-too-high border
    pixel wins and leaves a 1-px boundary seam. Here every pixel is weighted by
    its distance to that tile's rectangular border (capped at feather_radius);
    overlapping tiles are blended by a weighted average. The unreliable border
    is down-weighted, the full-weight interior keeps its sharpness, and the
    boundary becomes continuous.
    """
    from rasterio.transform import from_origin
    if bounds is not None:
        left, bottom, right, top = bounds
    else:
        ls, bs, rs, ts = [], [], [], []
        for p in paths:
            with rasterio.open(p) as s:
                b = s.bounds
                ls.append(b.left); bs.append(b.bottom); rs.append(b.right); ts.append(b.top)
        left, bottom, right, top = min(ls), min(bs), max(rs), max(ts)

    W = int(round((right - left) / res))
    H = int(round((top - bottom) / res))
    with rasterio.open(paths[0]) as s:
        crs = s.crs
        profile = s.profile.copy()
    transform = from_origin(left, top, res, res)

    accum_wz = np.zeros((H, W), dtype=np.float64)
    accum_w = np.zeros((H, W), dtype=np.float64)

    for p in paths:
        with rasterio.open(p) as s:
            z = s.read(1).astype(np.float64)
            nd = s.nodata
            b = s.bounds
        th, tw = z.shape
        valid = np.isfinite(z)
        if nd is not None and not np.isnan(nd):
            valid &= (z != nd)
        # distance to the rectangular tile border (NOT to interior holes, so the
        # interior keeps full weight); +1 so a single-tile pixel still counts.
        ii = np.minimum(np.arange(th), th - 1 - np.arange(th))
        jj = np.minimum(np.arange(tw), tw - 1 - np.arange(tw))
        w = np.minimum(np.minimum(ii[:, None], jj[None, :]), feather_radius).astype(np.float64) + 1.0
        w[~valid] = 0.0
        zz = np.where(valid, z, 0.0)

        col0 = int(round((b.left - left) / res))
        row0 = int(round((top - b.top) / res))
        r0, c0 = max(0, row0), max(0, col0)
        r1, c1 = min(H, row0 + th), min(W, col0 + tw)
        if r1 <= r0 or c1 <= c0:
            continue
        tr0, tc0 = r0 - row0, c0 - col0
        accum_wz[r0:r1, c0:c1] += (w * zz)[tr0:tr0 + (r1 - r0), tc0:tc0 + (c1 - c0)]
        accum_w[r0:r1, c0:c1] += w[tr0:tr0 + (r1 - r0), tc0:tc0 + (c1 - c0)]

    out = np.full((H, W), np.nan, dtype=np.float32)
    m = accum_w > 0
    out[m] = (accum_wz[m] / accum_w[m]).astype(np.float32)

    profile.update(driver="GTiff", height=H, width=W, transform=transform,
                   crs=crs, count=1, dtype="float32", nodata=np.nan)
    with rasterio.open(dst_path, "w", **profile) as d:
        d.write(out, 1)


def global_dsm(cfg, tiles: List[Tile]) -> None:
    """
    Merge tilewise DSMs and confidence maps in a global DSM and confidence map.
    """
    bounds = None
    if "roi_geojson" in cfg:
        ll_poly = geographiclib.read_lon_lat_poly_from_geojson(cfg["roi_geojson"])
        pyproj_crs = geographiclib.pyproj_crs(cfg["out_crs"])
        bounds = geographiclib.crs_bbx(ll_poly, pyproj_crs,
                                       align=cfg["dsm_resolution"])

    creation_options = {"tiled": True,
                        "zlevel": 2,
                        "blockxsize": 256,
                        "blockysize": 256,
                        "compress": "deflate",
                        "BIGTIFF": "IF_SAFER",
                        "COPY_SRC_OVERVIEWS": "YES",
                        "predictor": 2}

    dsms = []
    dsms_filtered = []
    confidence_maps = []

    for t in tiles:
        d = os.path.join(t.dir, "dsm.tif")
        if os.path.exists(d):
            dsms.append(d)

        f = os.path.join(t.dir, "dsm-filtered.tif")
        if os.path.exists(f):
            dsms_filtered.append(f)

        c = os.path.join(t.dir, "confidence.tif")
        if os.path.exists(c):
            confidence_maps.append(c)

    nb_workers = cfg['max_processes'] or multiprocessing.cpu_count()
    save_folder = os.path.join(cfg["out_dir"], "tile_merging")
    os.makedirs(save_folder, exist_ok=True)

    feather = cfg["dsm_merging_method"] == "feather"

    if dsms:
        global_dst_path_dsm = os.path.join(cfg["out_dir"], "dsm.tif")
        if feather:
            merge_tiles_feather(dsms, bounds, cfg["dsm_resolution"], global_dst_path_dsm)
        else:
            merge_tiles_mp(nb_workers, global_dst_path_dsm, save_folder, dsms, bounds,
                           res=cfg["dsm_resolution"], creation_options=creation_options,
                           method=cfg["dsm_merging_method"], remove_merged=True)

    if dsms_filtered:
        global_dst_path_dsm_filtered = os.path.join(cfg["out_dir"], "dsm-filtered.tif")
        if feather:
            merge_tiles_feather(dsms_filtered, bounds, cfg["dsm_resolution"],
                                global_dst_path_dsm_filtered)
        else:
            merge_tiles_mp(nb_workers, global_dst_path_dsm_filtered, save_folder,
                           dsms_filtered, bounds, res=cfg["dsm_resolution"],
                           creation_options=creation_options,
                           method=cfg["dsm_merging_method"], remove_merged=True)

    if confidence_maps:
        global_dst_path_dsm_confidence = os.path.join(cfg["out_dir"], "confidence.tif")
        merge_tiles_mp(nb_workers,
                       global_dst_path_dsm_confidence,
                       save_folder,
                       confidence_maps,
                       bounds,
                       res=cfg["dsm_resolution"],
                       creation_options=creation_options,
                       method=("max" if feather else cfg["dsm_merging_method"]),
                       remove_merged=True)

    os.rmdir(save_folder)


def main(user_cfg, start_from=0):
    """
    Launch the s2p pipeline with the parameters given in a json file.

    Args:
        user_cfg: user config dictionary
        start_from: the step to start from (default: 0)
    """
    common.reset_elapsed_time()

    # Reset the process-level DL-stereo flip decision so that back-to-back
    # s2p runs in the same process (e.g. run_all_models.sh) don't inherit
    # a stale flip from a previous config.
    from s2p.rectification import _reset_dl_global_flip_decision
    _reset_dl_global_flip_decision()

    # setup logger to stderr
    # (loggers per tiles are set in parallel.py)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    f = logging.Formatter('%(message)s')
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(f)
    root.addHandler(h)

    # s2p is already using (processed-based) parallelism when needed
    os.environ['GDAL_NUM_THREADS'] = "1"
    os.environ['OMP_NUM_THREADS'] = "1"

    cfg = config.get_default_config()
    initialization.build_cfg(cfg, user_cfg)
    initialization.make_dirs(cfg)

    # multiprocessing setup
    nb_workers = cfg['max_processes'] or multiprocessing.cpu_count()  # nb of available cores

    tw, th = initialization.adjust_tile_size(cfg)
    tiles_txt = os.path.join(cfg['out_dir'], 'tiles.txt')
    if start_from <= 1:
        tiles = initialization.tiles_full_info(cfg, tw, th, tiles_txt, create_masks=True)
    else: # skip mask creation if already done
        tiles = initialization.tiles_full_info(cfg, tw, th, tiles_txt, create_masks=False)
    if not tiles:
        logger.error('the ROI is not seen in two images or is totally masked.')
        sys.exit(1)

    if start_from > 0:
        assert os.path.exists(tiles_txt), "start_from set to {} but tiles.txt is not found in '{}'. Make sure this is" \
                                          " the output directory of a previous run.".format(start_from, cfg['out_dir'])
    else:
        # initialisation: write the list of tilewise json files to outdir/tiles.txt
        with open(tiles_txt, 'w') as f:
            for t in tiles:
                f.write(t.json)
                f.write('\n')

    n = len(cfg['images'])
    tiles_pairs = [(cfg, t, i) for i in range(1, n) for t in tiles]
    tiles_with_cfg = [(cfg, t) for t in tiles]
    timeout = cfg['timeout']

    # local-pointing step:
    if start_from <= 1:
        logger.info('1) correcting pointing locally...')
        successes = parallel.launch_calls(cfg, pointing_correction, tiles_pairs, nb_workers,
                              timeout=timeout)

        # update the tiles removing the discarded tiles
        tiles_pairs = [x for x, b in zip(tiles_pairs, successes) if b]

    # global-pointing step:
    if start_from <= 2:
        logger.info('2) correcting pointing globally...')
        global_pointing_correction(cfg, tiles)
        common.print_elapsed_time()

    # rectification step:
    if start_from <= 3:
        logger.info('3) rectifying tiles...')
        # If enabled, compute a SINGLE global rectification homography for
        # the whole ROI from RPC virtual matches and stash it in cfg. All
        # tiles will reuse the same H1/H2 base, so adjacent tiles have
        # byte-identical rectification conventions and the per-tile
        # boundary seams disappear.
        if cfg.get('dl_global_rectification', False):
            from s2p import rpc_utils
            roi = cfg['roi']
            rpc1 = cfg['images'][0]['rpcm']
            # Compute one global rectification H PER PAIR. Each pair (ref vs
            # image i) has its own epipolar geometry, so a single shared H is
            # only valid for the pair it was fit on; reusing pair 1's H for the
            # other pairs produces garbage heights there, which fusion then
            # discards (near-empty DSM). Store keyed by pair index.
            cfg['_global_H1'] = {}
            cfg['_global_H2'] = {}
            cfg['_dl_global_flip'] = {}
            for i in range(1, len(cfg['images'])):
                rpc2 = cfg['images'][i]['rpcm']
                g_matches = rpc_utils.matches_from_rpc(cfg, rpc1, rpc2,
                                                       roi['x'], roi['y'], roi['w'], roi['h'],
                                                       cfg['n_gcp_per_axis'])
                H1g, H2g, _Fg = rectification.rectification_homographies(
                    g_matches, roi['x'], roi['y'], roi['w'], roi['h'])

                # Global unipolarity + flip for DL stereo, decided per pair so
                # every tile of this pair shares one H (tile-boundary seams
                # vanish) with the correct flip convention.
                flip_i = False
                if cfg.get('matching_algorithm') == 'dl_stereo':
                    t_margin = cfg.get('dl_unipolarity_margin', 50)
                    H2g_neg = rectification.register_horizontally_translation(
                        g_matches, H1g, H2g, flag='negative')
                    H2g_neg = np.dot(common.matrix_translation(-t_margin, 0), H2g_neg)
                    mean_alt = np.mean(rpc_utils.altitude_range(cfg, rpc1,
                                                                roi['x'], roi['y'],
                                                                roi['w'], roi['h']))
                    grows = rectification.disparity_grows_with_altitude(
                        H1g, H2g_neg, rpc1, rpc2,
                        roi['x'] + roi['w'] // 2, roi['y'] + roi['h'] // 2, mean_alt)
                    flip_i = bool(not grows)
                    if grows:
                        H2g = H2g_neg
                    else:
                        H2g = rectification.register_horizontally_translation(
                            g_matches, H1g, H2g, flag='positive')
                        H2g = np.dot(common.matrix_translation(t_margin, 0), H2g)

                cfg['_global_H1'][i] = H1g
                cfg['_global_H2'][i] = H2g
                cfg['_dl_global_flip'][i] = flip_i
                logger.info('dl_global_rectification: pair %d global H from %d '
                            'RPC matches on ROI %dx%d, flip=%s',
                            i, len(g_matches), roi['w'], roi['h'], flip_i)
        elif (cfg.get('matching_algorithm') == 'dl_stereo'
              and cfg.get('dl_flip_mode', 'auto') == 'auto'):
            # Per-tile (non-global) DL path: decide the unipolarity FLIP once
            # per pair over the whole ROI and share it via cfg. rectify_pair
            # runs in spawned workers whose module-global flip cache resets, so
            # otherwise each worker's first tile decides on its own; in
            # borderline-geometry regions (e.g. ROI far from a scene's nadir)
            # different workers pick opposite flips, yielding garbage altitudes
            # in some tiles that average_if_close then discards -> near-empty
            # DSM. One ROI-wide decision per pair keeps every worker consistent.
            from s2p import rpc_utils
            roi = cfg['roi']
            rpc1 = cfg['images'][0]['rpcm']
            t_margin = cfg.get('dl_unipolarity_margin', 50)
            cfg['_dl_flip_decision'] = {}
            for i in range(1, len(cfg['images'])):
                rpc2 = cfg['images'][i]['rpcm']
                g_matches = rpc_utils.matches_from_rpc(cfg, rpc1, rpc2,
                                                       roi['x'], roi['y'], roi['w'], roi['h'],
                                                       cfg['n_gcp_per_axis'])
                H1g, H2g, _Fg = rectification.rectification_homographies(
                    g_matches, roi['x'], roi['y'], roi['w'], roi['h'])
                H2g_neg = rectification.register_horizontally_translation(
                    g_matches, H1g, H2g, flag='negative')
                H2g_neg = np.dot(common.matrix_translation(-t_margin, 0), H2g_neg)
                mean_alt = np.mean(rpc_utils.altitude_range(cfg, rpc1, roi['x'],
                                                            roi['y'], roi['w'], roi['h']))
                grows = rectification.disparity_grows_with_altitude(
                    H1g, H2g_neg, rpc1, rpc2,
                    roi['x'] + roi['w'] // 2, roi['y'] + roi['h'] // 2, mean_alt)
                cfg['_dl_flip_decision'][i] = bool(not grows)
                logger.info('dl flip (ROI-wide, pair %d): flip=%s', i, bool(not grows))
        successes = parallel.launch_calls(cfg, rectification_pair, tiles_pairs, nb_workers,
                              timeout=timeout)

        # update the tiles removing the discarded tiles
        tiles_pairs = [x for x, b in zip(tiles_pairs, successes) if b]

    # disparity range reasoning step: (WIP)
    if start_from <= 4:
        logger.info('4) reason about the disparity ranges... (WIP)')
        # extra step checking the disparity range
        # verity if the disparity range of a tile is not too different from its neighbors
        tiles_usefulnesses = parallel.launch_calls(cfg, disparity_range_check, tiles_pairs, nb_workers,
                              timeout=timeout)
        # some feedback
        for x, b in zip(tiles_pairs, tiles_usefulnesses):
            if not b: logger.info('  removed tile: %s', x[1].dir)

        # update the tiles removing the discarded tiles
        tiles_pairs = [x for x, b in zip(tiles_pairs, tiles_usefulnesses) if b]


    # Resumed runs (start_from >= 4) rebuild tiles_pairs from tiles.txt, which
    # still lists pairs that earlier steps discarded (failed pointing /
    # rectification / disp-range check) — their products are missing on disk.
    # A fresh run drops them via the per-step `successes` filters above, which
    # a resume skips, so re-derive the same filter from what actually exists.
    if start_from >= 4:
        def _pair_products_exist(tp):
            _, t, i = tp
            d = os.path.join(t.dir, 'pair_{}'.format(i))
            need = ['H_ref.txt', 'H_sec.txt', 'disp_min_max.txt']
            if start_from >= 5:
                need.append('rectified_disp.tif')
            return all(os.path.exists(os.path.join(d, f)) for f in need)
        n_before = len(tiles_pairs)
        tiles_pairs = [tp for tp in tiles_pairs if _pair_products_exist(tp)]
        if len(tiles_pairs) < n_before:
            logger.info('resume: dropped %d / %d tile-pairs whose products are '
                        'missing (discarded by the original run)',
                        n_before - len(tiles_pairs), n_before)

    # matching step:
    if start_from <= 4:
        logger.info('4) running stereo matching...')
        if cfg['max_processes_stereo_matching'] is not None:
            nb_workers_stereo = cfg['max_processes_stereo_matching']
        else:
            nb_workers_stereo = nb_workers

        if cfg["gpu_total_memory"] is not None:
            gpu_total_memory = cfg["gpu_total_memory"]
            # keep some space for the CUDA contexts
            gpu_total_memory -= nb_workers_stereo * 120
            gpu_mem_manager = GPUMemoryManager.make_bounded(
                max_memory_in_megabytes=gpu_total_memory,
                mp_context=parallel.get_mp_context(),
            )
        else:
            gpu_mem_manager = GPUMemoryManager.make_unbounded()

        parallel.launch_calls(cfg, stereo_matching, tiles_pairs,
                              nb_workers_stereo,
                              gpu_mem_manager,
                              timeout=timeout)

    ### UPDATE TILES_WITH_CFG FROM CURRENT TILES_PAIRS
    tilesdict = dict( [(t.json,t) for _,t,_ in tiles_pairs] )
    tiles_with_cfg = [(cfg,t) for t in tilesdict.values()]

    if start_from <= 5:
        if n > 2:
            # disparity-to-height step:
            logger.info('5a) computing height maps...')
            parallel.launch_calls(cfg, disparity_to_height, tiles_pairs, nb_workers,
                                  timeout=timeout)

            logger.info('5b) computing local pairwise height offsets...')
            parallel.launch_calls(cfg, mean_heights, tiles_with_cfg, nb_workers, timeout=timeout)

            # global-mean-heights step:
            logger.info('5c) computing global pairwise height offsets...')
            global_mean_heights(cfg, tiles)

            # height-map fusion step (merge the n-1 pairs per tile):
            logger.info('5d) merging height maps...')
            parallel.launch_calls(cfg, heights_fusion, tiles_with_cfg, nb_workers,
                                  timeout=timeout)

            # inter-tile height registration (removes tile-boundary seams):
            if cfg.get('inter_tile_align', True):
                logger.info('5e) aligning tile heights...')
                align_tile_heights(cfg, tiles)
            else:
                logger.info('5e) inter-tile alignment DISABLED (inter_tile_align=false)')

            # heights-to-ply step:
            logger.info('5f) computing point clouds...')
            parallel.launch_calls(cfg, heights_to_ply, tiles_with_cfg, nb_workers,
                                  timeout=timeout)
        else:
            # triangulation step:
            logger.info('5) triangulating tiles...')
            parallel.launch_calls(cfg, disparity_to_ply, tiles_with_cfg, nb_workers,
                                  timeout=timeout)

    # local-dsm-rasterization step:
    if start_from <= 6:
        logger.info('6) computing DSM by tile...')
        parallel.launch_calls(cfg, plys_to_dsm, tiles_with_cfg, nb_workers, timeout=timeout)

    # global-dsm-rasterization step:
    if start_from <= 7:
        logger.info('7) computing global DSM...')
        global_dsm(cfg, tiles)
    common.print_elapsed_time()
    common.print_elapsed_time(since_first_call=True)


def make_path_relative_to_file(path, f):
    return os.path.join(os.path.abspath(os.path.dirname(f)), path)


def read_tiles(tiles_file):
    outdir = os.path.dirname(tiles_file)

    with open(tiles_file) as f:
        tiles = f.readlines()

    # Strip trailing \n
    tiles = list(map(str.strip, tiles))
    tiles = [os.path.join(outdir, t) for t in tiles]

    return tiles


def read_config_file(config_file):
    """
    Read a json configuration file and interpret relative paths.

    If any input or output path is a relative path, it is interpreted as
    relative to the config_file location (and not relative to the current
    working directory). Absolute paths are left unchanged.
    """
    with open(config_file, 'r') as f:
        user_cfg = json.load(f)

    # output paths
    if not os.path.isabs(user_cfg['out_dir']):
        user_cfg['out_dir'] = make_path_relative_to_file(user_cfg['out_dir'],
                                                         config_file)

    # ROI path
    k = "roi_geojson"
    if k in user_cfg and isinstance(user_cfg[k], str) and not os.path.isabs(user_cfg[k]):
        user_cfg[k] = make_path_relative_to_file(user_cfg[k], config_file)

    if 'exogenous_dem' in user_cfg and user_cfg['exogenous_dem'] is not None:
        if not os.path.isabs(user_cfg['exogenous_dem']):
            user_cfg['exogenous_dem'] = make_path_relative_to_file(user_cfg['exogenous_dem'], config_file)

    # input paths
    for img in user_cfg['images']:
        for d in ['img', 'rpc', 'clr', 'cld', 'roi', 'wat']:
            if d in img and isinstance(img[d], str) and not os.path.isabs(img[d]):
                img[d] = make_path_relative_to_file(img[d], config_file)

    return user_cfg
