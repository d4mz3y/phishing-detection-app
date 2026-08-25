import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path


URL = 'https://www.kaggle.com/api/v1/datasets/download/rohansood98/phishing-email-dataset-nazario-5-and-trec07'
ARCHIVE_SHA256 = '47136985a463c70c9581704886153fce0ff7da30f91e44bfd55a4577084b705c'


def main():
    raw = Path('data/raw')
    archive = raw / 'phishing-email-dataset.zip'
    dataset = raw / 'dataset'
    raw.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        with urllib.request.urlopen(URL, timeout=60) as response, archive.open('wb') as output:
            shutil.copyfileobj(response, output)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != ARCHIVE_SHA256:
        raise RuntimeError(f'Dataset checksum mismatch: expected {ARCHIVE_SHA256}, received {digest}')
    dataset.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(dataset)
    print(f'Dataset ready at {dataset}; archive SHA-256: {digest}')


if __name__ == '__main__':
    main()
