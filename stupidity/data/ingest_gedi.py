"""
Helper for discovering, downloading, and parsing GEDI HDF5 granules from NASA LP DAAC (via CMR).

Usage examples:
  - List granules in a bbox and temporal range:
      python data/ingest_gedi.py --product GEDI02_A --bbox "-122.6,37.6,-122.3,37.9" --start 2020-04-01 --end 2021-01-01 --list

  - Download the first matching granule (requires EARTHDATA_USERNAME and EARTHDATA_PASSWORD env vars):
      python data/ingest_gedi.py --product GEDI02_A --bbox "-122.6,37.6,-122.3,37.9" --start 2020-04-01 --end 2021-01-01 --download --out data/gedi

  - Parse local directory of GEDI .h5 files into a CSV summary:
      python data/ingest_gedi.py --local-dir data/gedi --parse --out data/gedi_summary.csv

Notes:
 - This script uses the CMR API to search for granules and prints their available "links". Downloading requires NASA Earthdata credentials stored in env vars `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD`.
 - HDF5 parsing is best-effort: the script searches recursively for datasets whose name contains common tokens (latitude, longitude, elevation, rh100, rh90, elev_lowestmode, etc.) and returns them for inspection.
"""

import os
import sys
import argparse
import requests
import math
import h5py
import numpy as np
import csv
from urllib.parse import urlencode

CMR_GRANULES = "https://cmr.earthdata.nasa.gov/search/granules.json"

COMMON_ELEV_KEYS = ['elevation', 'elev_lowestmode', 'elev_highestreturn', 'elev_lowestmode', 'rh100', 'rh95', 'rh90', 'rh50']
COMMON_LAT_KEYS = ['latitude', 'lat']
COMMON_LON_KEYS = ['longitude', 'lon', 'long']


def search_granules(product: str, bbox: str = None, start: str = None, end: str = None, page_size: int = 20):
    params = {
        'short_name': product,
        'page_size': page_size,
    }
    if bbox:
        params['bounding_box'] = bbox
    if start and end:
        params['temporal'] = f"{start}T00:00:00Z/{end}T23:59:59Z"

    resp = requests.get(CMR_GRANULES, params=params, timeout=30)
    resp.raise_for_status()
    js = resp.json()
    items = js.get('feed', {}).get('entry', [])
    return items


def print_granule_links(granules):
    for g in granules:
        print(f"Title: {g.get('title')}")
        print(f"ID: {g.get('id')}")
        links = g.get('links', [])
        for l in links:
            href = l.get('href')
            rel = l.get('rel')
            type_ = l.get('type')
            title = l.get('title')
            print(f"  - href: {href}\n    rel: {rel} type: {type_} title: {title}")
        print('-' * 60)


def download_granule(url: str, out_dir: str):
    """Download a granule URL using Earthdata credentials if available.

    The user must set EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment variables.
    """
    user = os.environ.get('EARTHDATA_USERNAME')
    pwd = os.environ.get('EARTHDATA_PASSWORD')
    if not user or not pwd:
        raise RuntimeError('EARTHDATA_USERNAME and EARTHDATA_PASSWORD must be set to download from LP DAAC')

    os.makedirs(out_dir, exist_ok=True)
    local_name = os.path.join(out_dir, os.path.basename(url.split('?')[0]))

    with requests.Session() as s:
        # Request with auth - CMR may redirect through an Earthdata login page which will require the session
        r = s.get(url, auth=(user, pwd), stream=True, timeout=60)
        r.raise_for_status()
        with open(local_name, 'wb') as fh:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    fh.write(chunk)
    return local_name


def find_datasets(h5_obj):
    """Recursively walk HDF5 file and collect dataset names."""
    datasets = []

    def _walk(name, obj):
        if isinstance(obj, h5py.Dataset):
            datasets.append((name, obj.shape))

    h5_obj.visititems(_walk)
    return datasets


def extract_best_arrays(filepath: str):
    """Open HDF5 and attempt to extract arrays for lat/lon and elevation metrics.

    Returns a dict with keys found in the file.
    """
    res = {}
    with h5py.File(filepath, 'r') as f:
        datasets = find_datasets(f)
        names = [n for n, s in datasets]

        # helper to find candidate paths by substring
        def find_by_tokens(tokens):
            for n in names:
                low = n.lower()
                for t in tokens:
                    if t in low:
                        return n
            return None

        lat_path = find_by_tokens(COMMON_LAT_KEYS)
        lon_path = find_by_tokens(COMMON_LON_KEYS)
        elev_path = find_by_tokens(COMMON_ELEV_KEYS)

        # fallback: print available datasets and return empty
        if not lat_path or not lon_path:
            res['available_datasets'] = names
            return res

        res['latitude'] = f[lat_path][:]
        res['longitude'] = f[lon_path][:]
        if elev_path:
            res['elevation_metric'] = elev_path
            res['elevation'] = f[elev_path][:]
        else:
            # attempt to find any numeric dataset of same length
            # pick first numeric dataset with same first-dim length
            lat_len = res['latitude'].shape[0]
            for n, s in datasets:
                if s and s[0] == lat_len:
                    try:
                        arr = f[n][:]
                        if np.issubdtype(arr.dtype, np.number):
                            res['fallback_elevation'] = n
                            res['elevation'] = arr
                            break
                    except Exception:
                        continue
    return res


def parse_local_dir(local_dir: str, out_csv: str = None):
    files = [os.path.join(local_dir, f) for f in os.listdir(local_dir) if f.lower().endswith('.h5') or f.lower().endswith('.hdf5')]
    rows = []
    for fp in files:
        try:
            info = extract_best_arrays(fp)
            if 'latitude' in info and 'elevation' in info:
                npoints = int(info['latitude'].shape[0])
            else:
                npoints = 0
            rows.append({'file': fp, 'npoints': npoints, 'has_latlon': ('latitude' in info), 'has_elev': ('elevation' in info)})
        except Exception as e:
            rows.append({'file': fp, 'error': str(e)})

    if out_csv:
        keys = set()
        for r in rows:
            keys.update(r.keys())
        keys = list(keys)
        with open(out_csv, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"Wrote summary to {out_csv}")
    else:
        for r in rows:
            print(r)

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--product', default='GEDI02_A', help='CMR short_name product (e.g., GEDI02_A)')
    parser.add_argument('--bbox', help='bounding box as a single comma-separated string: minLon,minLat,maxLon,maxLat')
    parser.add_argument('--bbox4', nargs=4, type=float, help='bounding box as four numeric values: minLon minLat maxLon maxLat (alternate to --bbox)')
    parser.add_argument('--minlon', type=float, help='min longitude (alternative to --bbox)')
    parser.add_argument('--minlat', type=float, help='min latitude (alternative to --bbox)')
    parser.add_argument('--maxlon', type=float, help='max longitude (alternative to --bbox)')
    parser.add_argument('--maxlat', type=float, help='max latitude (alternative to --bbox)')
    parser.add_argument('--start', help='start date YYYY-MM-DD')
    parser.add_argument('--end', help='end date YYYY-MM-DD')
    parser.add_argument('--list', action='store_true', help='List matching granules')
    parser.add_argument('--download', action='store_true', help='Download first matching granule (requires EARTHDATA creds in env)')
    parser.add_argument('--out', default='data/gedi', help='output dir for downloads or parse summary')
    parser.add_argument('--local-dir', help='parse local GEDI .h5 files in this directory')
    parser.add_argument('--parse', action='store_true', help='Parse local-dir and summarize')
    args = parser.parse_args()

    def _normalize_bbox(bbox_arg):
        if not bbox_arg:
            return None
        # bbox_arg expected as a single comma-separated string
        if isinstance(bbox_arg, str):
            parts = [p.strip() for p in bbox_arg.replace(',', ' ').split() if p.strip()]
        else:
            # fallback: try to coerce iterable to strings
            parts = []
            for item in bbox_arg:
                parts.extend([p.strip() for p in str(item).replace(',', ' ').split() if p.strip()])

        if len(parts) != 4:
            raise ValueError('BBox must contain four numeric values: minLon minLat maxLon maxLat')
        return ','.join(parts)

    if args.local_dir and args.parse:
        parse_local_dir(args.local_dir, out_csv=os.path.join(args.out, 'gedi_summary.csv'))
        return

    # Allow bbox to be provided either as --bbox string/list or as four numeric flags
    try:
        # Priority: explicit numeric bbox flags, then --bbox4, then --bbox string
        if args.minlon is not None or args.minlat is not None or args.maxlon is not None or args.maxlat is not None:
            if None in (args.minlon, args.minlat, args.maxlon, args.maxlat):
                raise ValueError('When using --minlon/--minlat/--maxlon/--maxlat you must provide all four values')
            bbox_param = f"{args.minlon},{args.minlat},{args.maxlon},{args.maxlat}"
        elif args.bbox4 is not None:
            bbox_param = ','.join([str(float(v)) for v in args.bbox4])
        else:
            bbox_param = _normalize_bbox(args.bbox)
    except Exception as e:
        print(f'Invalid bbox argument: {e}')
        sys.exit(1)

    granules = search_granules(args.product, bbox=bbox_param, start=args.start, end=args.end)
    if args.list or (not args.download):
        print_granule_links(granules)

    if args.download:
        if len(granules) == 0:
            print('No granules found')
            return
        # try to find an appropriate link to download (prefer data links)
        links = granules[0].get('links', [])
        download_url = None
        for l in links:
            href = l.get('href')
            type_ = l.get('type', '')
            rel = l.get('rel', '')
            # prefer direct HDF5/data links
            if href and (href.lower().endswith('.h5') or href.lower().endswith('.hdf5') or 'data' in rel or 'data' in type_):
                download_url = href
                break
        if not download_url:
            # fallback: first link
            download_url = links[0].get('href')

        print(f"Attempting download: {download_url}")
        out_dir = args.out
        os.makedirs(out_dir, exist_ok=True)
        try:
            local = download_granule(download_url, out_dir)
            print(f"Downloaded to {local}")
        except Exception as e:
            print(f"Download failed: {e}")
            sys.exit(1)


if __name__ == '__main__':
    main()
