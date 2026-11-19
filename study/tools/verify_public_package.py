"""Read-only stdlib verification of the public source-and-record package."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'PUBLIC_MANIFEST.json').read_text())
    expected = set(manifest['files']) | {'PUBLIC_MANIFEST.json'}
    actual = set()
    for path in root.rglob('*'):
        if path.is_symlink():
            raise RuntimeError('Symlink in package')
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != expected:
        raise RuntimeError('Public payload membership differs')
    for name, item in manifest['files'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise RuntimeError('Unsafe manifest member')
        path = root / relative
        if path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
            raise RuntimeError('Payload changed: ' + name)
    inventory = json.loads((root / 'provenance/SOURCE_INVENTORY.json').read_text())
    copied = [row for row in inventory['original_core_payloads'] if row['included']]
    for row in copied:
        if digest(root / row['public_path']) != row['source_sha256']:
            raise RuntimeError('Preserved source bytes differ')
    print(json.dumps({'status': 'PUBLIC_SOURCE_AND_RECORD_BYTES_VERIFIED',
                      'payload_files': len(manifest['files']),
                      'preserved_core_payloads': len(copied),
                      'original_core_payloads': len(inventory['original_core_payloads']),
                      'model_or_metric_replay_performed': False}, sort_keys=True))


if __name__ == '__main__':
    main()
