"""Download Lichess PGN archives."""

import re
from pathlib import Path

import requests
from tqdm import tqdm

from .config import LICHESS_DB_URL, RAW_DIR


def download_pgn(year: int, month: int, output_dir: Path | None = None) -> Path:
    """Download a monthly Lichess PGN archive (.pgn.zst).

    Args:
        year: e.g. 2023
        month: 1-12
        output_dir: Where to save. Defaults to data/raw/

    Returns:
        Path to downloaded file
    """
    output_dir = output_dir or RAW_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"lichess_db_standard_rated_{year}-{month:02d}.pgn.zst"
    url = f"{LICHESS_DB_URL}/{filename}"
    output_path = output_dir / filename

    if output_path.exists():
        print(f"Already downloaded: {output_path}")
        return output_path

    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(output_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=filename
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

    print(f"Saved to {output_path}")
    return output_path


def list_available_months() -> list[tuple[int, int]]:
    """List available year/month combinations from Lichess.

    Returns:
        List of (year, month) tuples
    """
    response = requests.get(LICHESS_DB_URL)
    response.raise_for_status()

    pattern = r"lichess_db_standard_rated_(\d{4})-(\d{2})\.pgn\.zst"
    matches = re.findall(pattern, response.text)

    return sorted([(int(y), int(m)) for y, m in matches])
