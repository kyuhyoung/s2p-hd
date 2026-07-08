# Copyright (C) 2015, Carlo de Franchis <carlo.de-franchis@cmla.ens-cachan.fr>
# Copyright (C) 2015, Gabriele Facciolo <facciolo@cmla.ens-cachan.fr>
# Copyright (C) 2015, Enric Meinhardt <enric.meinhardt@cmla.ens-cachan.fr>


import os
import logging
import warnings

import numpy as np

from s2p import rpc_utils
from s2p import estimation
from s2p import evaluation
from s2p import common
from s2p import visualisation
from s2p import homography


logger = logging.getLogger(__name__)


# Module-level cache for the global DL-stereo flip decision.
# The altitude-consistency flip is a property of the image pair geometry,
# not of a tile; making it per-tile causes adjacent tiles that happen to
# land on different sides of the numeric threshold to disagree, producing
# seams of 1-3 m at tile boundaries. We cache the first tile's decision
# and reuse it for every subsequent tile in the same run.
_dl_global_flip_decision = None

# Module-level cache of per-tile rectification homographies.
# Keyed by tile origin (x, y). Value is (H1, H2). Populated when
# cfg['dl_h_smooth'] is True so later tiles can blend their local H with
# already-computed neighbor tiles' H -- reduces tile-boundary seams that
# SGM never exhibits but DL stereo does, because the DL model amplifies
# small per-tile rectification differences.
#
# Caveat: single-pass implementation. First tiles have no neighbors cached
# yet and use pure local H. Tiles processed later benefit from accumulated
# neighbor info. The smoothing is therefore uneven in time, though in
# space it tends to converge as the tile grid fills in.
_dl_h_tile_cache = {}


def _reset_dl_global_flip_decision():
    """Reset the cached flip decision and H cache. Called at the start of each s2p run."""
    global _dl_global_flip_decision
    _dl_global_flip_decision = None
    _dl_h_tile_cache.clear()


def _correspondence_smooth_single_h(H_local, tile_x, tile_y, tile_w, tile_h,
                                    neighbor_radius_tiles=1, self_weight=0.4,
                                    grid_n=5):
    """Correspondence-based smoothing of a single rectification H.

    This is the right mathematical tool for averaging projective
    transforms (Begelfor & Werman 2005 style): sample a grid of points,
    apply each candidate H to produce target displacements, blend the
    target displacements, then recover the smoothed H via DLT from the
    (grid, blended target) pairs.

    To keep each tile's per-tile translation intact, targets are centered
    on each tile's own center-target before averaging and recentered on
    the LOCAL tile's center-target before DLT. So the "shape" part of H
    (rotation, scale, shear, projective row) is blended across neighbors
    while the translation column that maps this tile's ROI origin to its
    rectified origin is preserved.

    Returns the smoothed H, or H_local unchanged if no neighbor is
    cached yet or DLT fails.
    """
    if not _dl_h_tile_cache:
        return H_local

    try:
        import cv2
    except ImportError:
        logger.warning('cv2 not available; falling back to no smoothing')
        return H_local

    # Local tile grid in absolute image coordinates.
    u = np.linspace(0, 1, grid_n)
    v = np.linspace(0, 1, grid_n)
    uu, vv = np.meshgrid(u, v)
    G_local = np.stack([tile_x + uu.ravel() * tile_w,
                        tile_y + vv.ravel() * tile_h], axis=1).astype(np.float32)
    c_local = np.array([[tile_x + tile_w / 2.0, tile_y + tile_h / 2.0]],
                       dtype=np.float32)

    T_local = cv2.perspectiveTransform(G_local[None], H_local)[0]
    cT_local = cv2.perspectiveTransform(c_local[None], H_local)[0, 0]
    C_local = T_local - cT_local

    r = neighbor_radius_tiles
    C_nbr_accum = np.zeros_like(C_local)
    weights_sum = 0.0
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            if dx == 0 and dy == 0:
                continue
            nx = tile_x + dx * tile_w
            ny = tile_y + dy * tile_h
            entry = _dl_h_tile_cache.get((nx, ny))
            if entry is None:
                continue
            # pick matching H (H1 vs H2) by identity of the local; caller
            # passes one at a time, so use the same index.
            # We key the cache as (nx, ny) -> (H1, H2); the caller decides
            # which slot. Here we accept both and pick later.
            H_n = entry  # entry is already the single H matrix now
            # Local grid but as if observed from neighbor's frame:
            # apply H_n to G_local to get where neighbor's transform would
            # place THIS tile's grid points.
            T_n = cv2.perspectiveTransform(G_local[None], H_n)[0]
            cT_n = cv2.perspectiveTransform(c_local[None], H_n)[0, 0]
            C_n = T_n - cT_n
            w = float(np.exp(-0.5 * (dx * dx + dy * dy)))
            C_nbr_accum += w * C_n
            weights_sum += w

    if weights_sum <= 0:
        return H_local

    C_nbr_avg = C_nbr_accum / weights_sum
    # Blend displacements and re-add local center-target to recover
    # absolute rectified coordinates in the LOCAL tile's frame.
    C_blend = self_weight * C_local + (1.0 - self_weight) * C_nbr_avg
    T_blend = C_blend + cT_local

    # DLT fit of smoothed homography.
    H_new, _ = cv2.findHomography(G_local, T_blend.astype(np.float32), method=0)
    if H_new is None:
        return H_local
    return H_new.astype(H_local.dtype)


def _log_euclidean_smooth_single_h(H_local, tile_x, tile_y, tile_w, tile_h,
                                    neighbor_radius_tiles=1, self_weight=0.4):
    """Log-Euclidean (Lie-algebra) smoothing of a single rectification H.

    Each H is normalized to det=1, mapped to sl(3) via matrix logarithm,
    averaged linearly in that vector space, then mapped back via matrix
    exponential. Stays on the homography manifold by construction.

    ~15 lines; simpler than correspondence-based. Does not preserve any
    geometric invariant (e.g. ROI origin at 0), so caller must recenter.
    """
    if not _dl_h_tile_cache:
        return H_local
    try:
        from scipy.linalg import logm, expm
    except ImportError:
        logger.warning('scipy not available for log-Euclidean smoothing')
        return H_local

    def _norm(H):
        det = np.linalg.det(H)
        if abs(det) < 1e-12:
            return H
        return H / np.cbrt(abs(det))

    try:
        L_local = logm(_norm(H_local))
    except Exception:
        return H_local

    r = neighbor_radius_tiles
    L_nbr_accum = np.zeros_like(L_local)
    weights_sum = 0.0
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            if dx == 0 and dy == 0:
                continue
            nx = tile_x + dx * tile_w
            ny = tile_y + dy * tile_h
            H_n = _dl_h_tile_cache.get((nx, ny))
            if H_n is None:
                continue
            try:
                L_n = logm(_norm(H_n))
            except Exception:
                continue
            w = float(np.exp(-0.5 * (dx * dx + dy * dy)))
            L_nbr_accum += w * L_n
            weights_sum += w

    if weights_sum <= 0:
        return H_local

    L_nbr = L_nbr_accum / weights_sum
    L_blend = self_weight * L_local + (1.0 - self_weight) * L_nbr
    try:
        H_new = np.real(expm(L_blend))
    except Exception:
        return H_local

    # Restore original scale (we normalized to det=1).
    det_local = abs(np.linalg.det(H_local))
    det_new = abs(np.linalg.det(H_new))
    if det_new > 1e-12:
        H_new = H_new * np.cbrt(det_local / det_new)
    return H_new.astype(H_local.dtype)


def _smooth_h_pair_with_neighbors(H1_local, H2_local, tile_x, tile_y,
                                  tile_w, tile_h,
                                  neighbor_radius_tiles=1, self_weight=0.4,
                                  method='correspondence'):
    """Smooth the pair (H1, H2) using the selected method.

    method: 'correspondence' (DLT of blended pixel targets) or
            'log_euclidean' (Lie-algebra mean of det-normalized H).
    """
    global _dl_h_tile_cache
    if not _dl_h_tile_cache:
        return H1_local, H2_local

    # Build two views of the cache keyed by the same tile origin but
    # exposing H1 / H2 respectively.
    cache_snapshot_h1 = {k: v[0] for k, v in _dl_h_tile_cache.items()}
    cache_snapshot_h2 = {k: v[1] for k, v in _dl_h_tile_cache.items()}

    if method == 'log_euclidean':
        smoother = _log_euclidean_smooth_single_h
    else:
        smoother = _correspondence_smooth_single_h

    original = _dl_h_tile_cache
    try:
        _dl_h_tile_cache = cache_snapshot_h1
        H1_s = smoother(
            H1_local, tile_x, tile_y, tile_w, tile_h,
            neighbor_radius_tiles=neighbor_radius_tiles, self_weight=self_weight,
        )
        _dl_h_tile_cache = cache_snapshot_h2
        H2_s = smoother(
            H2_local, tile_x, tile_y, tile_w, tile_h,
            neighbor_radius_tiles=neighbor_radius_tiles, self_weight=self_weight,
        )
    finally:
        _dl_h_tile_cache = original
    return H1_s, H2_s


class NoHorizontalRegistrationWarning(Warning):
    pass


def filter_matches_epipolar_constraint(F, matches, thresh):
    """
    Discards matches that are not consistent with the epipolar constraint.

    Args:
        F: fundamental matrix
        matches: list of pairs of 2D points, stored as a Nx4 numpy array
        thresh: maximum accepted distance between a point and its matched
            epipolar line

    Returns:
        the list of matches that satisfy the constraint. It is a sub-list of
        the input list.
    """
    out = []
    for match in matches:
        x = np.array([match[0], match[1], 1])
        xx = np.array([match[2], match[3], 1])
        d1 = evaluation.distance_point_to_line(x, np.dot(F.T, xx))
        d2 = evaluation.distance_point_to_line(xx, np.dot(F, x))
        if max(d1, d2) < thresh:
            out.append(match)

    return np.array(out)


def register_horizontally_shear(matches, H1, H2, debug=False):
    """
    Adjust rectifying homographies with tilt, shear and translation to reduce the disparity range.

    Args:
        matches: list of pairs of 2D points, stored as a Nx4 numpy array
        H1, H2: two homographies, stored as numpy 3x3 matrices

    Returns:
        H2: corrected homography H2

    The matches are provided in the original images coordinate system. By
    transforming these coordinates with the provided homographies, we obtain
    matches whose disparity is only along the x-axis.
    """
    # transform the matches according to the homographies
    p1 = homography.points_apply_homography(H1, matches[:, :2])
    x1 = p1[:, 0]
    y1 = p1[:, 1]
    p2 = homography.points_apply_homography(H2, matches[:, 2:])
    x2 = p2[:, 0]
    y2 = p2[:, 1]

    if debug:
        logging.info("Residual vertical disparities: max, min, mean. Should be zero")
        logging.info("%s %s %s", np.max(y2 - y1), np.min(y2 - y1), np.mean(y2 - y1))

    # we search the (a, b, c) vector that minimises \sum (x1 - (a*x2+b*y2+c))^2
    # it is a least squares minimisation problem
    A = np.column_stack((x2, y2, y2*0+1))
    a, b, c = np.linalg.lstsq(A, x1, rcond=None)[0]

    # correct H2 with the estimated tilt, shear and translation
    return np.dot(np.array([[a, b, c], [0, 1, 0], [0, 0, 1]]), H2)


def register_horizontally_translation(matches, H1, H2, flag='center', debug=False):
    """
    Adjust rectifying homographies with a translation to modify the disparity range.

    Args:
        matches: list of pairs of 2D points, stored as a Nx4 numpy array
        H1, H2: two homographies, stored as numpy 3x3 matrices
        flag: option needed to control how to modify the disparity range:
            'center': move the barycenter of disparities of matches to zero
            'positive': make all the disparities positive
            'negative': make all the disparities negative. Required for
                Hirshmuller stereo (java)

    Returns:
        H2: corrected homography H2

    The matches are provided in the original images coordinate system. By
    transforming these coordinates with the provided homographies, we obtain
    matches whose disparity is only along the x-axis. The second homography H2
    is corrected with a horizontal translation to obtain the desired property
    on the disparity range.
    """
    # transform the matches according to the homographies
    p1 = homography.points_apply_homography(H1, matches[:, :2])
    x1 = p1[:, 0]
    y1 = p1[:, 1]
    p2 = homography.points_apply_homography(H2, matches[:, 2:])
    x2 = p2[:, 0]
    y2 = p2[:, 1]

    # for debug, print the vertical disparities. Should be zero.
    if debug:
        logging.info("Residual vertical disparities: max, min, mean. Should be zero")
        logging.info("%s %s %s", np.max(y2 - y1), np.min(y2 - y1), np.mean(y2 - y1))

    # compute the disparity offset according to selected option.
    # 앵커 계산 전에 이상치 매치를 제거한다: min/max 앵커는 가장 극단적인 매치
    # 하나에 정합 전체가 인질로 잡히는 구조라, 가짜 SIFT 매치 하나가 지면 밴드를
    # 수백 px 깊은 곳에 앉혀 DL 매처를 어려운 영역으로 밀어넣는다
    # (대전 lower full: 지면이 -50대가 아니라 -500에 앉음, 런마다 ±240px 요동).
    d_all = x2 - x1
    _med = np.median(d_all)
    _mad = np.median(np.abs(d_all - _med))
    d_rob = d_all[np.abs(d_all - _med) <= 5 * 1.4826 * _mad + 1e-6]
    if d_rob.size == 0:
        d_rob = d_all
    t = 0
    if (flag == 'center'):
        t = np.mean(d_rob)
    if (flag == 'positive'):
        t = np.min(d_rob)
    if (flag == 'negative'):
        t = np.max(d_rob)

    # correct H2 with a translation
    return np.dot(common.matrix_translation(-t, 0), H2)


def disparity_grows_with_altitude(H1, H2, rpc1, rpc2, x_center, y_center, alt_ground):
    """
    Check whether disparity grows with altitude for the given homographies.

    This is required for DL stereo matchers: after enforcing negative unipolar
    disparities, higher altitude must produce larger |disparity| (more negative).
    If this doesn't hold, the images need to be horizontally flipped.

    Based on diachronicstereo Algorithm 1 (arXiv:2601.22808).

    Args:
        H1, H2: rectifying homographies (3x3 arrays)
        rpc1, rpc2: RPC camera models
        x_center, y_center: ROI center in original image coordinates
        alt_ground: mean ground altitude

    Returns:
        True if disparity grows with altitude (correct orientation).
    """
    alt_high = alt_ground + 50

    # project ground and elevated point through both cameras
    lon, lat = rpc1.localization(x_center, y_center, alt_ground)

    x1_g, y1_g = rpc1.projection(lon, lat, alt_ground)
    x2_g, y2_g = rpc2.projection(lon, lat, alt_ground)
    x1_h, y1_h = rpc1.projection(lon, lat, alt_high)
    x2_h, y2_h = rpc2.projection(lon, lat, alt_high)

    # apply homographies
    p1_g = homography.points_apply_homography(H1, [[x1_g, y1_g]])[0]
    p2_g = homography.points_apply_homography(H2, [[x2_g, y2_g]])[0]
    p1_h = homography.points_apply_homography(H1, [[x1_h, y1_h]])[0]
    p2_h = homography.points_apply_homography(H2, [[x2_h, y2_h]])[0]

    disp_ground = p1_g[0] - p2_g[0]
    disp_high = p1_h[0] - p2_h[0]

    # Disparity (left_x - right_x) should increase with altitude.
    # This is the normal geometric relationship for satellite stereo.
    grows = disp_ground < disp_high
    logger.info('altitude consistency check: disp_ground=%.2f, disp_high=%.2f, grows=%s',
                disp_ground, disp_high, grows)
    return grows


def disparity_range_from_matches(matches, H1, H2, disp_range_extra_margin):
    """
    Compute the disparity range of a ROI from a list of point matches.

    Args:
        matches: Nx4 numpy array containing a list of matches, in the full
            image coordinates frame, before rectification
        w, h: width and height of the rectangular ROI in the first image.
        H1, H2: two rectifying homographies, stored as numpy 3x3 matrices

    Returns:
        disp_min, disp_max: horizontal disparity range
    """
    # transform the matches according to the homographies
    p1 = homography.points_apply_homography(H1, matches[:, :2])
    x1 = p1[:, 0]
    p2 = homography.points_apply_homography(H2, matches[:, 2:])
    x2 = p2[:, 0]

    # compute the final disparity range
    #disp_min = np.floor(np.min(x2 - x1))
    #disp_max = np.ceil(np.max(x2 - x1))
    disp_min, disp_max = np.quantile (x2-x1, [0.01, 0.99])

    # add a security margin to the disparity range
    disp_min -= (disp_max - disp_min) * disp_range_extra_margin
    disp_max += (disp_max - disp_min) * disp_range_extra_margin
    return disp_min, disp_max


def disparity_range(cfg, rpc1, rpc2, x, y, w, h, H1, H2, matches, A=None):
    """
    Compute the disparity range of a ROI from a list of point matches.

    Args:
        rpc1, rpc2 (rpcm.RPCModel): two RPC camera models
        x, y, w, h (int): 4-tuple of integers defining the rectangular ROI in
            the first image. (x, y) is the top-left corner, and (w, h) are the
            dimensions of the rectangle.
        H1, H2 (np.array): two rectifying homographies, stored as 3x3 arrays
        matches (np.array): Nx4 array containing a list of sift matches, in the
            full image coordinates frame
        A (np.array): 3x3 array containing the pointing error correction for
            im2. This matrix is usually estimated with the pointing_accuracy
            module.

    Returns:
        disp: 2-uple containing the horizontal disparity range
    """
    # compute exogenous disparity range if needed
    exogenous_disp = None
    if cfg['exogenous_dem'] and cfg['disp_range_method'] in ['exogenous', 'wider_sift_exogenous']:
        exogenous_disp = rpc_utils.exogenous_disp_range_estimation(cfg, rpc1, rpc2,
                                                                   x, y, w, h,
                                                                   H1, H2, A,
                                                                   cfg['disp_range_exogenous_high_margin'],
                                                                   cfg['disp_range_exogenous_low_margin'])

        logging.info("exogenous disparity range: %s", exogenous_disp)

    # compute SIFT disparity range if needed
    if cfg['disp_range_method'] in ['sift', 'wider_sift_exogenous']:
        if matches is not None and len(matches) >= 2:
            sift_disp = disparity_range_from_matches(matches, H1, H2, cfg['disp_range_extra_margin'])
        else:
            sift_disp = None
        logging.info("SIFT disparity range: %s", sift_disp)

        # SIFT samples the ground densely but often finds ZERO matches on tall
        # untextured roofs (glass/concrete towers), so a pure-SIFT range hugs
        # the ground and tall buildings fall outside the matcher's search
        # window. On the long-baseline pair of a triplet they then flatten to
        # ground level, and fusion discards the roofs the short pair did get
        # (Daejeon lower: 145 m towers need ~290 px on pair 2 while the SIFT
        # range covered ~80 m). Extend the range upward by the disparity span
        # of `disp_range_building_margin` metres, converted with THIS pair's
        # own alt-to-disp sensitivity (signed, so the correct end grows).
        bm = cfg.get('disp_range_building_margin', 0)
        if bm > 0 and sift_disp is not None:
            alt = rpc_utils.altitude_range(cfg, rpc1, x, y, w, h)
            h0 = float(np.mean(alt))
            d_lo = rpc_utils.altitude_range_to_disp_range(h0, h0, rpc1, rpc2,
                                                          x, y, w, h, H1, H2, A)
            d_hi = rpc_utils.altitude_range_to_disp_range(h0 + bm, h0 + bm, rpc1,
                                                          rpc2, x, y, w, h, H1, H2, A)
            up = float(np.mean(d_hi) - np.mean(d_lo))
            # 부호 있는 단측 확장: alt->disp 변환(타일 자신의 H/A 사용)이 주는
            # 방향으로만 넓힌다. 이 방향 계산의 신뢰성은 RPC 대조로 확인됨
            # (대전 lower 아파트 타일: 예언 지면 -503/지붕 -327 = 실측과 일치,
            # 즉 이 pair에서는 고도가 높을수록 0에 가깝다). 과거의 "방향을 믿을
            # 수 없다"는 결론과 far-end/양측 확장 시도는 정합 이상치 요동을
            # 확장 문제로 오진한 것이었다 (registration robust화로 원인 제거).
            if up >= 0:
                sift_disp = (sift_disp[0], sift_disp[1] + up)
            else:
                sift_disp = (sift_disp[0] + up, sift_disp[1])
            logging.info("building margin %s m -> signed extension %+.1f px, range %s",
                         bm, up, sift_disp)

    # compute altitude range disparity if needed
    if cfg['disp_range_method'] == 'fixed_altitude_range':
        alt_disp = rpc_utils.altitude_range_to_disp_range(cfg['alt_min'],
                                                          cfg['alt_max'],
                                                          rpc1, rpc2,
                                                          x, y, w, h, H1, H2, A)
        logging.info("disparity range computed from fixed altitude range: %s", alt_disp)

    # now compute disparity range according to selected method
    if cfg['disp_range_method'] == 'exogenous':
        disp = exogenous_disp

    elif cfg['disp_range_method'] == 'sift':
        disp = sift_disp

    elif cfg['disp_range_method'] == 'wider_sift_exogenous':
        if sift_disp is not None and exogenous_disp is not None:
            disp = min(exogenous_disp[0], sift_disp[0]), max(exogenous_disp[1], sift_disp[1])
        else:
            disp = sift_disp or exogenous_disp

    elif cfg['disp_range_method'] == 'fixed_altitude_range':
        disp = alt_disp

    elif cfg['disp_range_method'] == 'fixed_pixel_range':
        disp = cfg['disp_min'], cfg['disp_max']

    # default disparity range to return if everything else broke
    if disp is None:
        disp = -3, 3

    # impose a minimal disparity range (only for center flag, not for DL stereo)
    if cfg.get('matching_algorithm') != 'dl_stereo':
        disp = min(-3, disp[0]), max(3, disp[1])
    else:
        logging.info("DL stereo: keeping unipolar disparity range [%.1f, %.1f], nearest_to_zero=%.1fpx",
                      disp[0], disp[1], min(abs(disp[0]), abs(disp[1])))

    logging.info("Final disparity range: %s", disp)
    return disp


def rectification_homographies(matches, x, y, w, h, debug=False):
    """
    Computes rectifying homographies from point matches for a given ROI.

    The affine fundamental matrix F is estimated with the gold-standard
    algorithm, then two rectifying similarities (rotation, zoom, translation)
    are computed directly from F.

    Args:
        matches: numpy array of shape (n, 4) containing a list of 2D point
            correspondences between the two images.
        x, y, w, h: four integers defining the rectangular ROI in the first
            image. (x, y) is the top-left corner, and (w, h) are the dimensions
            of the rectangle.
    Returns:
        S1, S2, F: three numpy arrays of shape (3, 3) representing the
        two rectifying similarities to be applied to the two images and the
        corresponding affine fundamental matrix.
    """
    # estimate the affine fundamental matrix with the Gold standard algorithm
    F = estimation.affine_fundamental_matrix(matches)

    # compute rectifying similarities
    S1, S2 = estimation.rectifying_similarities_from_affine_fundamental_matrix(F, debug)

    if debug:
        y1 = homography.points_apply_homography(S1, matches[:, :2])[:, 1]
        y2 = homography.points_apply_homography(S2, matches[:, 2:])[:, 1]
        err = np.abs(y1 - y2)
        logging.info("max, min, mean rectification error on point matches: ")
        logging.info("%s %s %s", np.max(err), np.min(err), np.mean(err))

    # pull back top-left corner of the ROI to the origin (plus margin)
    pts = homography.points_apply_homography(S1, [[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
    x0, y0 = common.bounding_box2D(pts)[:2]
    T = common.matrix_translation(-x0, -y0)
    return np.dot(T, S1), np.dot(T, S2), F


def rectify_pair(cfg, im1, im2, rpc1, rpc2, x, y, w, h, out1, out2, A=None, sift_matches=None,
                 method='rpc', hmargin=0, vmargin=0, pair_idx=1):
    """
    Rectify a ROI in a pair of images.

    Args:
        im1, im2: paths to two GeoTIFF image files
        rpc1, rpc2: two instances of the rpcm.RPCModel class
        x, y, w, h: four integers defining the rectangular ROI in the first
            image.  (x, y) is the top-left corner, and (w, h) are the dimensions
            of the rectangle.
        out1, out2: paths to the output rectified crops
        A (optional): 3x3 numpy array containing the pointing error correction
            for im2. This matrix is usually estimated with the pointing_accuracy
            module.
        sift_matches (optional): Nx4 numpy array containing a list of sift
            matches, in the full image coordinates frame
        method (default: 'rpc'): option to decide whether to use rpc of sift
            matches for the fundamental matrix estimation.
        {h,v}margin (optional): horizontal and vertical margins added on the
            sides of the rectified images

    Returns:
        H1, H2: Two 3x3 matrices representing the rectifying homographies that
        have been applied to the two original (large) images.
        disp_min, disp_max: horizontal disparity range
        success: bool (can be false if not enough matches, invalid homographies, ...)
    """
    _dl_canonical = False   # 방향 정준화 수행 여부 (수행 시 flip 기계장치 비활성)
    global _dl_global_flip_decision  # may be read early (global-H branch) or
                                      # written later (per-tile flip decision).
                                      # Hoist so both access paths are valid.
    debug = cfg['debug']
    # compute real or virtual matches
    if method == 'rpc':
        # find virtual matches from RPC camera models
        matches = rpc_utils.matches_from_rpc(cfg, rpc1, rpc2, x, y, w, h,
                                             cfg['n_gcp_per_axis'])

        # correct second image coordinates with the pointing correction matrix
        if A is not None:
            matches[:, 2:] = homography.points_apply_homography(np.linalg.inv(A),
                                                                matches[:, 2:])
    elif method == 'sift':
        matches = sift_matches

    else:
        raise Exception("Unknown value {} for argument 'method'".format(method))

    if matches is None or len(matches) < 4:
        logging.info("No or not enough matches found to rectify image pair")
        return None, None, None, None, False

    # If a global rectification was pre-computed (cfg['dl_global_rectification']
    # path), reuse it for every tile so adjacent tiles share a single H and
    # their rectifications match byte-for-byte at the shared boundary. Each
    # tile still computes its own disparity range and hmargin below, then
    # composes T_tile on top of the global H, which preserves per-tile
    # output framing without introducing per-tile rectification drift.
    _gH1 = cfg.get('_global_H1')
    _gH2 = cfg.get('_global_H2')
    if (_gH1 is not None and _gH2 is not None
            and pair_idx in _gH1 and pair_idx in _gH2):
        # Per-pair global H: each stereo pair (ref vs image i) has its own
        # epipolar geometry, so it needs its own global rectification. Reusing
        # pair 1's H for pair 2 yields garbage heights for pair 2 (then fusion
        # discards everything where the pairs disagree).
        H1 = _gH1[pair_idx].copy()
        H2 = _gH2[pair_idx].copy()
        F = None
        _used_global_H = True
        # rectification_homographies() normalizes H so that the ROI bbox it
        # was fit on maps to (0, 0). The global H was fit on the WHOLE ROI,
        # so for any tile that isn't at the ROI origin, this tile's ROI bbox
        # under H lands at a non-zero offset; the downstream
        # assert_allclose(bbox == (hmargin, vmargin)) then fails by exactly
        # that offset. Re-apply the same bbox-to-origin translation here,
        # per tile, so the invariant is preserved without perturbing the
        # shape of the rectification (still identical across tiles).
        roi_tile = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
        _pts = homography.points_apply_homography(H1, roi_tile)
        _x0, _y0 = common.bounding_box2D(_pts)[:2]
        T_local = common.matrix_translation(-_x0, -_y0)
        H1 = np.dot(T_local, H1)
        H2 = np.dot(T_local, H2)

        # Propagate the globally-decided flip to this tile's local cfg so the
        # image-flip step after warping runs consistently. The flag is set
        # per tile (not once at pre-compute) because cfg.pop removes it
        # after each tile's flip step.
        if (cfg.get('matching_algorithm') == 'dl_stereo' and
                cfg.get('_dl_global_flip', {}).get(pair_idx, False)):
            cfg['_dl_flip_images'] = True
    else:
        _used_global_H = False
        try:
            H1, H2, F = rectification_homographies(matches, x, y, w, h, debug=debug)
        except AssertionError:
            logging.info("rectification.rectify_pair.rectification_homographies assertion failed")
            return None, None, None, None, False

        # --- 방향 정준화 (DL 경로) ---
        # rectification_homographies가 만드는 H의 좌우 방향(orientation)은
        # SIFT 매치 분포에 따라 타일마다 뒤집힌다(det 부호 복불복). 방향이
        # 뒤집힌 타일에서는 고도가 높을수록 시차가 0 쪽으로 이동해 unipolarity
        # 금지선을 넘고, 고층 건물이 매칭 불가가 된다 (대전 lower 아파트:
        # 동일 설정에서 방향 +인 타일은 타워 포착, -인 타일은 실패 — 단일타일
        # 하네스 A/B로 확정). 타일 자신의 RPC 기하로 "고도 증가 -> 시차 감소
        # (더 깊은 음수)"가 되도록 강제해 동전던지기를 제거한다.
        if (cfg.get('matching_algorithm') == 'dl_stereo'
                and cfg.get('dl_orientation_canonical', True)):
            try:
                _cx, _cy = x + w // 2, y + h // 2
                _h0 = float(np.mean(rpc_utils.altitude_range(cfg, rpc1, x, y, w, h)))
                _d0 = rpc_utils.alt_to_disp(rpc1, rpc2, _cx, _cy, _h0, H1, H2, A)
                _d1 = rpc_utils.alt_to_disp(rpc1, rpc2, _cx, _cy, _h0 + 100, H1, H2, A)
                _up = float(np.mean(np.atleast_1d(_d1)) - np.mean(np.atleast_1d(_d0)))
                _dl_canonical = True
                if _up > 0:
                    M = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
                    H1 = np.dot(M, H1)
                    H2 = np.dot(M, H2)
                    # 거울 반전으로 ROI bbox가 음수 좌표로 가므로 원점 재정규화
                    # (rectification_homographies의 bbox 규약 복원)
                    _roi = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                    _pts = homography.points_apply_homography(H1, _roi)
                    _x0, _y0 = common.bounding_box2D(_pts)[:2]
                    _Tn = common.matrix_translation(-_x0, -_y0)
                    H1 = np.dot(_Tn, H1)
                    H2 = np.dot(_Tn, H2)
                    logging.info('orientation canonicalized (mirrored): up %+.2f px/100m -> %+.2f',
                                 _up, -_up)
            except Exception:
                logging.exception('orientation canonicalization failed; keeping original H')

    if cfg['register_with_shear'] and not _used_global_H:
        # compose H2 with a horizontal shear to reduce the disparity range
        a = np.mean(rpc_utils.altitude_range(cfg, rpc1, x, y, w, h))
        lon, lat, alt = rpc_utils.ground_control_points(rpc1, x, y, w, h, a, a, 4)
        x1, y1 = rpc1.projection(lon, lat, alt)
        x2, y2 = rpc2.projection(lon, lat, alt)
        m = np.vstack([x1, y1, x2, y2]).T
        m = np.vstack(list({tuple(row) for row in m}))  # remove duplicates due to no alt range
        H2 = register_horizontally_shear(m, H1, H2, debug=debug)

    # compose H2 with a horizontal translation to adjust disparity range
    # For DL stereo: enforce unipolar (negative) disparities with margin,
    # so that higher altitude always corresponds to larger |disparity|.
    # For classical matchers: center around 0 (original behavior).
    use_dl = cfg.get('matching_algorithm') == 'dl_stereo'

    # Under global rectification we skip local SIFT-driven H2 refinement:
    # running it per tile would reintroduce the exact tile-to-tile H drift
    # the global mode is meant to eliminate. The flip decision cache is
    # still populated (below) so the downstream image-flip step still runs.
    if sift_matches is not None and not _used_global_H:
        sift_matches = filter_matches_epipolar_constraint(F, sift_matches,
                                                          cfg['epipolar_thresh'])
        if len(sift_matches) < 1:
            warnings.warn(
                "Need at least one sift match for the horizontal registration",
                category=NoHorizontalRegistrationWarning,
            )
        else:
            if use_dl:
                t_margin = cfg.get('dl_unipolarity_margin', 50)

                # First try negative unipolarity
                H2_neg = register_horizontally_translation(sift_matches, H1, H2,
                                                           flag='negative',
                                                           debug=debug)
                H2_neg = np.dot(common.matrix_translation(-t_margin, 0), H2_neg)

                # Decide flip: this is an IMAGE-PAIR property, not a tile property.
                # Compute once per run (cached in _dl_global_flip_decision) so that
                # every tile uses the same convention and the tile-to-tile seams
                # caused by numeric jitter in the per-tile altitude consistency
                # check disappear. Override via cfg['dl_flip_mode'] in {'auto',
                # 'always', 'never'}; default 'auto' uses the cached first-tile
                # decision. (global declared at top of rectify_pair.)
                flip_mode = cfg.get('dl_flip_mode', 'auto')
                # Flip decided ONCE for the whole ROI in main() and shared via
                # cfg (per pair). rectify_pair runs in worker processes that may
                # be spawned (module globals reset to None), so without this each
                # worker's first tile decides independently; in borderline-
                # geometry regions different workers then pick opposite flips,
                # producing garbage altitudes in some tiles that fusion discards
                # (near-empty DSM). cfg is pickled to every worker -> consistent.
                _cfg_flip = cfg.get('_dl_flip_decision', {}).get(pair_idx)

                if _dl_canonical:
                    # 방향 정준화가 이미 "고도증가 -> 시차 깊어짐"을 타일별로
                    # 보장했으므로 두 번째 거울(image flip)은 금지 — 거울 두 장이
                    # 조합되면 pair별로 최종 방향이 반대로 갈린다 (A/B 실측).
                    need_flip = False
                elif flip_mode == 'always':
                    need_flip = True
                elif flip_mode == 'never':
                    need_flip = False
                elif _cfg_flip is not None:
                    need_flip = _cfg_flip
                elif _dl_global_flip_decision is not None:
                    need_flip = _dl_global_flip_decision
                    logger.info('using cached dl flip decision: flip=%s', need_flip)
                else:
                    mean_alt = np.mean(rpc_utils.altitude_range(cfg, rpc1, x, y, w, h))
                    grows = disparity_grows_with_altitude(H1, H2_neg, rpc1, rpc2,
                                                          x + w // 2, y + h // 2,
                                                          mean_alt)
                    need_flip = not grows
                    _dl_global_flip_decision = need_flip
                    # global declared at function top so this rebinding is valid
                    logger.info('dl flip decision (first tile, cached for rest of run): '
                                'flip=%s', need_flip)

                if not need_flip:
                    H2 = H2_neg
                else:
                    # Flip: use positive unipolarity instead
                    logger.info('altitude consistency failed, flipping to positive unipolarity')
                    H2 = register_horizontally_translation(sift_matches, H1, H2,
                                                           flag='positive',
                                                           debug=debug)
                    H2 = np.dot(common.matrix_translation(t_margin, 0), H2)

                    # Apply horizontal flip to both homographies
                    # This is done later during image warping by flipping the output
                    cfg['_dl_flip_images'] = True
            else:
                H2 = register_horizontally_translation(sift_matches, H1, H2,
                                                       debug=debug)

    # Under global rectification we deliberately do NOT apply any per-tile
    # translation to H2: every tile must keep the byte-identical global H1/H2
    # so adjacent tiles reconstruct a single continuous surface (no seams).
    # The disparity stays uncentered (large offset), which widens the rectified
    # tile and the DL cost volume -- that is bounded purely by tile_size, not
    # by mutating H. (A per-tile H2 shift was tried and reintroduced a seam at
    # every tile boundary, so it is rejected.)

    # DL continuous-H-field MVP. Blend this tile's (H1, H2) with whichever
    # neighbor tiles are already cached, then store the result for future
    # tiles to use. Single-pass so neighbor coverage is order-dependent but
    # a true two-pass refactor would require splitting rectify_pair.
    # Skip under global rectification: all tiles already share a single H,
    # so blending with neighbors only reintroduces drift.
    if use_dl and cfg.get('dl_h_smooth', False) and not _used_global_H:
        H1_raw, H2_raw = H1.copy(), H2.copy()
        H1, H2 = _smooth_h_pair_with_neighbors(
            H1, H2, x, y, w, h,
            neighbor_radius_tiles=cfg.get('dl_h_smooth_radius', 1),
            self_weight=cfg.get('dl_h_smooth_self_weight', 0.4),
            method=cfg.get('dl_h_smooth_method', 'correspondence'),
        )
        # rectification_homographies() returns H normalized so that the
        # ROI bbox under H has its top-left at (0, 0); the assert_allclose
        # at the bottom of this function relies on that invariant. Smoothing
        # breaks it, so re-apply the same translation fix used upstream.
        def _recenter_to_roi_origin(H):
            roi_corners = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
            pts = homography.points_apply_homography(H, roi_corners)
            x0, y0 = common.bounding_box2D(pts)[:2]
            return np.dot(common.matrix_translation(-x0, -y0), H)
        H1 = _recenter_to_roi_origin(H1)
        H2 = _recenter_to_roi_origin(H2)
        _dl_h_tile_cache[(x, y)] = (H1, H2)
        # Quantify how much the smoothing changed H so that activation is
        # observable in the main log (visible delta = smoothing actually
        # happened; zero delta = first tile with no cached neighbors yet).
        dH1 = float(np.max(np.abs(H1 - H1_raw)))
        dH2 = float(np.max(np.abs(H2 - H2_raw)))
        msg = ('[dl_h_smooth] tile (%d, %d)  cached=%d  '
               'max|dH1|=%.4g  max|dH2|=%.4g' %
               (x, y, len(_dl_h_tile_cache), dH1, dH2))
        logger.info(msg)
        print(msg, flush=True)  # also to stdout for visibility in test_margin.log

    # compute disparity range
    if debug and sift_matches is not None:
        out_dir = os.path.dirname(out1)
        np.savetxt(os.path.join(out_dir, 'sift_matches_disp.txt'),
                   sift_matches, fmt='%9.3f')
        visualisation.plot_matches(cfg, im1, im2, rpc1, rpc2, sift_matches,
                                   os.path.join(out_dir,
                                                'sift_matches_disp.png'),
                                   x, y, w, h)

    disp_m, disp_M = disparity_range(cfg, rpc1, rpc2, x, y, w, h, H1, H2,
                                     sift_matches, A)

    # recompute hmargin and homographies
    hmargin = int(np.ceil(max([hmargin, np.fabs(disp_m), np.fabs(disp_M)])))
    T = common.matrix_translation(hmargin, vmargin)
    H1, H2 = np.dot(T, H1), np.dot(T, H2)

    # compute output images size
    roi = [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
    pts1 = homography.points_apply_homography(H1, roi)
    x0, y0, w0, h0 = common.bounding_box2D(pts1)
    # check that the first homography maps the ROI in the positive quadrant
    np.testing.assert_allclose(np.round([x0, y0]), [hmargin, vmargin], atol=.01)

    # apply homographies and do the crops
    out_w = w0 + 2*hmargin
    out_h = h0 + 2*vmargin
    success = homography.image_apply_homography(out1, im1, H1, out_w, out_h, verbose=debug)
    success = success and homography.image_apply_homography(out2, im2, H2, out_w, out_h, verbose=debug)

    # For DL stereo: if altitude consistency failed, flip rectified images horizontally
    # and update homographies to account for the flip.
    if success and use_dl and cfg.get('_dl_flip_images', False):
        import rasterio
        for img_path in [out1, out2]:
            with rasterio.open(img_path, 'r') as src:
                data = src.read()
                profile = src.profile.copy()
            data = data[:, :, ::-1].copy()  # flip horizontally
            with rasterio.open(img_path, 'w', **profile) as dst:
                dst.write(data)

        # Update homographies: compose with horizontal flip matrix
        H_flip = np.array([[-1, 0, out_w - 1], [0, 1, 0], [0, 0, 1]], dtype=float)
        H1 = np.dot(H_flip, H1)
        H2 = np.dot(H_flip, H2)

        # Disparity range also flips sign
        disp_m, disp_M = -disp_M, -disp_m

        cfg.pop('_dl_flip_images', None)
        logger.info('applied horizontal flip to rectified images')

    return H1, H2, disp_m, disp_M, success
