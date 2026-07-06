#!/usr/bin/env python3
"""Daejeon full-area foundation config 생성.

upper/lower 각각 3개 scene의 RPC footprint를 lonlat에서 교차시켜
3장 공통 겹침 영역을 ref(첫번째) 영상 픽셀좌표 ROI로 산출하고,
기존 foundation_config.json에서 ROI/out_dir만 바꾼
foundation_config_full.json을 생성한다 (원본 보존).

Usage:
    python3 make_daejeon_full_config.py            # upper + lower
    python3 make_daejeon_full_config.py upper      # 한쪽만
"""
import json
import math
import sys

import rasterio
import rpcm
from shapely.geometry import Polygon

HOST_ROOT = "/data/kevin_workspace/dataset_stereo/satellite"
DOCKER_ROOT = "/data/satellite"
TILE = 2000  # config의 tile_size (요약 출력용)


def to_host(docker_path):
    return docker_path.replace(DOCKER_ROOT, HOST_ROOT, 1)


def footprint_lonlat(img_host, rpc, alt):
    """영상 테두리(모서리 4 + 변 중점 4)를 지상 alt로 localize한 lonlat polygon."""
    with rasterio.open(img_host) as ds:
        w, h = ds.width, ds.height
    pts_px = [
        (0, 0), (w / 2, 0), (w, 0), (w, h / 2),
        (w, h), (w / 2, h), (0, h), (0, h / 2),
    ]
    lons, lats = rpc.localization(
        [p[0] for p in pts_px], [p[1] for p in pts_px], [alt] * len(pts_px)
    )
    return Polygon(zip(lons, lats)), (w, h)


def make_full_config(part):
    base_host = f"{HOST_ROOT}/daejeon/{part}"
    cfg = json.load(open(f"{base_host}/foundation_config.json"))

    rpcs, polys, sizes = [], [], []
    for im in cfg["images"]:
        img_host = to_host(im["img"])
        rpc = rpcm.rpc_from_rpc_file(to_host(im["rpc"]))
        rpcs.append(rpc)
        poly, size = footprint_lonlat(img_host, rpc, rpc.alt_offset)
        polys.append(poly)
        sizes.append(size)

    overlap = polys[0]
    for p in polys[1:]:
        overlap = overlap.intersection(p)
    if overlap.is_empty:
        raise RuntimeError(f"[{part}] 3-scene 공통 겹침 없음")

    # 겹침 polygon 꼭짓점을 ref 영상 픽셀로 투영 -> bbox
    ref = rpcs[0]
    lons, lats = zip(*overlap.exterior.coords)
    cols, rows = ref.projection(list(lons), list(lats), [ref.alt_offset] * len(lons))
    w0, h0 = sizes[0]
    x0 = max(0, int(math.floor(min(cols))))
    y0 = max(0, int(math.floor(min(rows))))
    x1 = min(w0, int(math.ceil(max(cols))))
    y1 = min(h0, int(math.ceil(max(rows))))
    roi = {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}

    cfg["roi"] = roi
    cfg["out_dir"] = cfg["out_dir"].rstrip("/") + "_full"
    cfg["temporary_dir"] = cfg["temporary_dir"].rstrip("/") + "_full"

    out_path = f"{base_host}/foundation_config_full.json"
    json.dump(cfg, open(out_path, "w"), indent=2)

    ntx = math.ceil(roi["w"] / TILE)
    nty = math.ceil(roi["h"] / TILE)
    cover = 100.0 * roi["w"] * roi["h"] / (w0 * h0)
    print(f"[{part}]")
    for (w, h), im in zip(sizes, cfg["images"]):
        print(f"  scene {im['img'].split('/')[-1]}: {w} x {h}")
    print(f"  ref 대비 겹침 ROI: x={x0} y={y0} w={roi['w']} h={roi['h']}  ({cover:.0f}% of ref)")
    print(f"  tiles (tile_size {TILE} 상한): {ntx} x {nty} = {ntx * nty}")
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    parts = sys.argv[1:] or ["upper", "lower"]
    for part in parts:
        make_full_config(part)
