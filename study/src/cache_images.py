"""Deterministic official CelebA image preprocessing; no learned transforms."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image


def preprocess_image(path):
    with Image.open(path) as image:
        image = image.convert('RGB')
        width, height = image.size
        # Same centered 178-pixel crop for every official aligned image.
        left, top = round((width - 178) / 2), round((height - 178) / 2)
        image = image.crop((left, top, left + 178, top + 178))
        image = image.resize((64, 64), resample=Image.Resampling.BICUBIC)
        return np.asarray(image, dtype=np.uint8).transpose(2, 0, 1).copy()


def build_cache(image_dir, output, workers=4):
    image_dir, output = Path(image_dir), Path(output)
    paths = [image_dir / f'{i:06d}.jpg' for i in range(1, 202600)]
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        raise RuntimeError(f'Missing {len(missing)} official images; first {missing[:5]}')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite image cache {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.incomplete.npy')
    array = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.uint8,
                                      shape=(202599, 3, 64, 64))
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Bounded chunks avoid retaining 202,599 pending futures/images in RAM.
        for start in range(0, len(paths), 1024):
            for offset, pixels in enumerate(pool.map(preprocess_image, paths[start:start+1024])):
                array[start + offset] = pixels
            if start % 10240 == 0:
                print(json.dumps({'cache_completed': min(start+1024,len(paths)),
                                  'total': len(paths), 'elapsed_s': time.monotonic()-started}), flush=True)
    array.flush()
    del array
    temporary.replace(output)
    digest = hashlib.sha256()
    with output.open('rb') as stream:
        for chunk in iter(lambda: stream.read(16*1024*1024), b''):
            digest.update(chunk)
    record = {'filename': output.name, 'sha256': digest.hexdigest(),
              'shape': [202599,3,64,64], 'dtype':'uint8',
              'ordering':'official filename 000001.jpg through 202599.jpg',
              'crop': 'center 178x178 using round((dimension-178)/2)',
              'resize': 'Pillow RGB bicubic 64x64', 'elapsed_s':time.monotonic()-started,
              'corrupt_images':0, 'missing_images':0}
    output.with_suffix('.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record), flush=True)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    build_cache(args.image_dir, args.output, args.workers)
