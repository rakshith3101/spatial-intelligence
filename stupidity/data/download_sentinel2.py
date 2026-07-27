"""
Download Sentinel-2 L2A products from the Copernicus Data Space Ecosystem.

This script searches the official STAC catalog, filters by bounding box/date/cloud cover,
and downloads the selected products into an output directory.

Authentication:
  Create an OAuth client in the Copernicus Data Space Ecosystem dashboard, then set:
    CDSE_CLIENT_ID
    CDSE_CLIENT_SECRET

Example:
  python data/download_sentinel2.py ^
    --bbox "72.0,18.0,73.0,19.0" ^
    --start 2024-01-01 --end 2024-02-01 ^
    --max-cloud 20 ^
    --out data/sentinel2
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.parse import urlparse

import requests


STAC_SEARCH_URL = "https://catalogue.dataspace.copernicus.eu/stac/search"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"


def get_access_token(username: str, password: str) -> str:
    resp = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": username,
            "client_secret": password,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def search_products(token: str, bbox: str, start: str, end: str, max_cloud: float, limit: int = 20):
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "collections": ["sentinel-2-l2a"],
        "bbox": [float(v) for v in bbox.split(",")],
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "limit": limit,
        "filter": f"eo:cloud_cover<{max_cloud}",
    }
    resp = requests.post(STAC_SEARCH_URL, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json().get("features", [])


def _pick_download_link(item: dict) -> str | None:
    for link in item.get("links", []):
        rel = link.get("rel", "")
        href = link.get("href")
        if href and rel == "download":
            return href
    for link in item.get("assets", {}).values():
        href = link.get("href")
        if href:
            return href
    return None


def download_file(url: str, token: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = Path(urlparse(url).path).name or "sentinel2_product.zip"
    out_path = out_dir / name

    headers = {"Authorization": f"Bearer {token}"}
    with requests.get(url, headers=headers, stream=True, timeout=300, allow_redirects=True) as r:
        r.raise_for_status()
        with open(out_path, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Download Sentinel-2 L2A from Copernicus CDSE.")
    parser.add_argument("--bbox", required=True, help="minLon,minLat,maxLon,maxLat")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--max-cloud", type=float, default=20.0, help="Maximum cloud cover percentage")
    parser.add_argument("--limit", type=int, default=5, help="Number of products to fetch")
    parser.add_argument("--out", default="data/sentinel2", help="Download folder")
    args = parser.parse_args()

    username = os.environ.get("CDSE_CLIENT_ID")
    password = os.environ.get("CDSE_CLIENT_SECRET")
    if not username or not password:
        raise RuntimeError("Set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET in your environment.")

    token = get_access_token(username, password)
    items = search_products(token, args.bbox, args.start, args.end, args.max_cloud, limit=args.limit)
    if not items:
        print("No Sentinel-2 products found.")
        return

    out_dir = Path(args.out)
    for item in items:
        title = item.get("properties", {}).get("title", item.get("id", "unknown"))
        link = _pick_download_link(item)
        if not link:
            print(f"Skipping {title}: no downloadable link found.")
            continue
        print(f"Downloading {title}")
        path = download_file(link, token, out_dir)
        print(f"Saved to {path}")


if __name__ == "__main__":
    main()
