"""Verify pinned-source snapshot bytes with stdlib only; execute no upstream code."""
from pathlib import Path
import hashlib
import json


def main():
    root = Path(__file__).resolve().parent
    files = 0
    snapshots = 0
    link_only = 0
    for manifest_path in sorted(root.glob('*/SNAPSHOT_MANIFEST.json')):
        record = json.loads(manifest_path.read_text())
        folder = manifest_path.parent
        expected = {'README.md', 'SNAPSHOT_MANIFEST.json'} | {'source/' + p for p in record['included_files']}
        actual = set()
        for path in folder.rglob('*'):
            if path.is_symlink():
                raise RuntimeError('Symlink inside snapshot')
            if path.is_file():
                actual.add(path.relative_to(folder).as_posix())
        if actual != expected:
            raise RuntimeError('Snapshot membership mismatch')
        for name, item in record['included_files'].items():
            path = folder / 'source' / name
            data = path.read_bytes()
            blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256'] or blob != item['git_blob_sha1']:
                raise RuntimeError('Source byte identity differs')
            files += 1
        if record['included_files']:
            if 'LICENSE' not in record['included_files']:
                raise RuntimeError('License absent')
            snapshots += 1
        else:
            if record['status'] != 'LINK_ONLY_NO_CODE_REDISTRIBUTED':
                raise RuntimeError('Unclear empty-snapshot status')
            link_only += 1
    print(json.dumps({'status': 'PINNED_SOURCE_BYTES_VERIFIED_NOT_EXECUTED',
                      'licensed_snapshots': snapshots, 'link_only_records': link_only,
                      'source_files': files, 'upstream_code_executed': False}, sort_keys=True))


if __name__ == '__main__':
    main()
