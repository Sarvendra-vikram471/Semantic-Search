import hashlib
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
URL_TEMPLATE = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"

DATASETS = {
    "scifact": "5f7d1de60b170fc8027bb7898e2efca1",
    "nfcorpus": "a89dba18a62ef92f7d323ec890a0d38d",
}

REQUIRED_FILES = ("corpus.jsonl", "queries.jsonl")


def has_dataset(dataset_dir: Path) -> bool:
    return dataset_dir.is_dir() and all((dataset_dir / name).exists() for name in REQUIRED_FILES)


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url) as response, destination.open("wb") as out_file:
        shutil.copyfileobj(response, out_file)


def ensure_dataset(name: str, expected_md5: str) -> None:
    dataset_dir = DATA_DIR / name
    if has_dataset(dataset_dir):
        print(f"[Dataset] {name} already present at {dataset_dir}")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    url = URL_TEMPLATE.format(name=name)

    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = Path(temp_dir) / f"{name}.zip"
        print(f"[Dataset] Downloading {name} from {url}")
        download_file(url, zip_path)

        actual_md5 = md5sum(zip_path)
        if actual_md5 != expected_md5:
            raise RuntimeError(
                f"{name} checksum mismatch: expected {expected_md5}, got {actual_md5}"
            )

        print(f"[Dataset] Extracting {name} into {DATA_DIR}")
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(DATA_DIR)

    if not has_dataset(dataset_dir):
        raise RuntimeError(f"{name} download finished, but required files are missing in {dataset_dir}")

    print(f"[Dataset] {name} ready at {dataset_dir}")


def main() -> None:
    for name, checksum in DATASETS.items():
        ensure_dataset(name, checksum)


if __name__ == "__main__":
    main()
