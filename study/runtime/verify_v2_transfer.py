"""Local-only stdlib verification of v2 ASCII-chunk artifact transfers.

No network, cloud, Torch, original-study imports, model loading or inference.
Transport integrity is relative to the supplied manifests/optional expected
record; these files alone are not an independent authenticity signature.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
import zipfile

CHUNK_LIMIT = 512 * 1024  # Default toy transport; actual input declares one known cap.
KNOWN_CHUNK_LIMITS = frozenset((512*1024, 1024*1024, 2*1024*1024))
RAW_CHUNK_BYTES = (CHUNK_LIMIT // 77) * 57
MAX_ARCHIVE = 2 * 1024**3
MAX_PAYLOAD = 8 * 1024**3
MAX_FILES = 100000
MAX_JSON = 16 * 1024**2
SHA_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            value.update(block)
    return value.hexdigest()


def no_duplicate_keys(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, 'Duplicate JSON key')
        value[key] = item
    return value


def json_bytes(data):
    require(len(data) <= MAX_JSON, 'JSON exceeds local verifier limit')
    def invalid_constant(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(data.decode('utf-8'), object_pairs_hook=no_duplicate_keys,
                      parse_constant=invalid_constant)


def regular_file(path):
    path = Path(path)
    require(not path.is_symlink() and path.is_file(), 'Missing or symlink file: ' + str(path))
    require(all(not parent.is_symlink() for parent in path.absolute().parents),
            'Symlink parent refused: ' + str(path))
    return path


def read_json(path):
    path = regular_file(path)
    require(path.stat().st_size <= MAX_JSON, 'JSON file exceeds local verifier limit')
    return json_bytes(path.read_bytes())


def nonnegative_integer(value):
    return type(value) is int and value >= 0


def file_spec(item):
    require(isinstance(item, dict) and nonnegative_integer(item.get('bytes'))
            and isinstance(item.get('sha256'), str) and SHA_PATTERN.fullmatch(item['sha256']),
            'Invalid byte/hash record')


def safe_member(name):
    require(isinstance(name, str) and bool(name) and '\\' not in name and ':' not in name
            and not any(ord(c) < 32 or ord(c) == 127 for c in name), 'Unsafe archive member')
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and '..' not in relative.parts
            and str(relative) == name and name not in {'.', '..'}, 'Noncanonical archive member')
    return name


def fresh_destination(path):
    path = Path(path).absolute()
    require(not path.is_symlink() and not path.exists(), 'Destination already exists: ' + str(path))
    require(all(not parent.is_symlink() for parent in path.parents), 'Symlink output parent refused')
    return path.resolve()


def load_chunks(manifest_path):
    manifest_path = regular_file(manifest_path).absolute()
    manifest = read_json(manifest_path)
    require(isinstance(manifest, dict), 'Chunk manifest must be an object')
    name = safe_member(manifest.get('archive_name'))
    require('/' not in name and name.endswith('.zip'), 'Expected a plain ZIP archive name')
    require(isinstance(manifest.get('archive_sha256'), str)
            and SHA_PATTERN.fullmatch(manifest['archive_sha256']), 'Invalid archive SHA256')
    size = manifest.get('archive_bytes')
    require(type(size) is int and 0 < size <= MAX_ARCHIVE, 'Invalid or oversized archive')
    limit = manifest.get('max_encoded_bytes_per_chunk')
    require(type(limit) is int and limit in KNOWN_CHUNK_LIMITS
            and manifest.get('columns') == 76
            and manifest.get('encoding') == 'base64_ASCII_LF_final_newline', 'Wrong armour definition')
    raw_chunk_bytes = (limit // 77) * 57
    chunks = manifest.get('chunks')
    count = (size + raw_chunk_bytes - 1) // raw_chunk_bytes
    require(isinstance(chunks, list) and type(manifest.get('chunk_count')) is int
            and manifest['chunk_count'] == len(chunks) == count, 'Missing or duplicate chunk count')
    prefix = Path(name).stem + '--' + manifest['archive_sha256'][:16]
    for index, item in enumerate(chunks, 1):
        file_spec(item)
        require(item.get('filename') == f'{prefix}--chunk{index:05d}.b64.txt',
                'Chunk names/order differ from canonical transport order')
        require(0 < item['bytes'] <= limit, 'Oversized encoded chunk')
        expected_size = raw_chunk_bytes if index < count else size - (count - 1) * raw_chunk_bytes
        require(type(item.get('decoded_bytes')) is int and item['decoded_bytes'] == expected_size
                and isinstance(item.get('decoded_sha256'), str)
                and SHA_PATTERN.fullmatch(item['decoded_sha256']), 'Invalid decoded chunk record')
    return manifest_path, manifest


def verified_chunks(manifest_path, manifest):
    limit = manifest['max_encoded_bytes_per_chunk']
    require(type(limit) is int and limit in KNOWN_CHUNK_LIMITS, 'Unknown armour cap')
    for item in manifest['chunks']:
        path = regular_file(manifest_path.parent / item['filename'])
        require(path.stat().st_size <= limit + limit // 76 + 2, 'Oversized copied chunk')
        text = path.read_bytes().replace(b'\r\n', b'\n')
        require(text.isascii() and len(text) == item['bytes']
                and hashlib.sha256(text).hexdigest() == item['sha256'], 'Copied chunk text/hash differs')
        raw = base64.b64decode(text.replace(b'\n', b''), validate=True)
        require(len(raw) == item['decoded_bytes']
                and hashlib.sha256(raw).hexdigest() == item['decoded_sha256'], 'Decoded chunk/hash differs')
        encoded = base64.b64encode(raw)
        canonical = b'\n'.join(encoded[i:i + 76] for i in range(0, len(encoded), 76)) + b'\n'
        require(text == canonical, 'Noncanonical armour whitespace/line wrapping')
        yield raw


def expected_archive_record(path, manifest):
    if path is None:
        return None
    record = read_json(path)
    if 'archives' in record:
        require(manifest['archive_name'] in record['archives'], 'Archive absent from supplied export record')
        expected = record['archives'][manifest['archive_name']]
        file_spec(expected)
        require(expected.get('archive') == manifest['archive_name'], 'Export archive name differs')
        size, digest = expected['bytes'], expected['sha256']
    else:
        require(record.get('status') in {'durable_EMA_archive_and_text_chunks_verified',
                                        'durable_selected_EMA_archive_and_text_chunks_verified'}
                and record.get('archive') == manifest['archive_name'], 'Not a matching final durable record')
        expected = record
        size, digest = record['archive_bytes'], record['archive_sha256']
    require(size == manifest['archive_bytes'] and digest == manifest['archive_sha256'],
            'Supplied expected archive record differs')
    require(expected.get('armour_manifest_sha256') == sha(manifest['_path']),
            'Supplied expected chunk-manifest SHA differs')
    return expected


def verify_zip(path, expected_internal_sha=None):
    regular_file(path)
    require(path.stat().st_size <= MAX_ARCHIVE, 'ZIP exceeds local verifier limit')
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        require(len(names) == len(set(names)) and len(names) <= MAX_FILES + 1, 'Duplicate/excess ZIP members')
        for info in infos:
            safe_member(info.filename)
            mode = info.external_attr >> 16
            require(not info.is_dir() and not (info.flag_bits & 1)
                    and stat.S_IFMT(mode) in {0, stat.S_IFREG}, 'Directory, encrypted or special ZIP member refused')
        require(names.count('MANIFEST.json') == 1, 'Exactly one internal manifest required')
        manifest_info = archive.getinfo('MANIFEST.json')
        require(manifest_info.file_size <= MAX_JSON, 'Oversized internal manifest')
        data = archive.read(manifest_info)
        if expected_internal_sha is not None:
            require(hashlib.sha256(data).hexdigest() == expected_internal_sha, 'Internal manifest SHA differs')
        manifest = json_bytes(data)
        require(isinstance(manifest, dict) and manifest.get('manifest_excludes_itself') is True
                and isinstance(manifest.get('files'), dict), 'Invalid internal ZIP manifest')
        require(set(names) == set(manifest['files']) | {'MANIFEST.json'}, 'Internal ZIP membership differs')
        total = 0
        for name, item in manifest['files'].items():
            safe_member(name); file_spec(item)
            info = archive.getinfo(name)
            require(info.file_size == item['bytes'], 'ZIP header/manifest size differs')
            total += item['bytes']
            require(total <= MAX_PAYLOAD, 'Uncompressed payload exceeds local verifier limit')
        for name, item in manifest['files'].items():
            digest, count = hashlib.sha256(), 0
            with archive.open(name) as stream:
                for block in iter(lambda: stream.read(1024**2), b''):
                    count += len(block)
                    require(count <= item['bytes'], 'ZIP payload exceeds declared size')
                    digest.update(block)
            require(count == item['bytes'] and digest.hexdigest() == item['sha256'], 'ZIP member hash differs: ' + name)
    return manifest, hashlib.sha256(data).hexdigest()


def extract_verified(path, destination, manifest):
    destination = fresh_destination(destination)
    destination.mkdir(parents=True)
    with zipfile.ZipFile(path) as archive:
        for name in ['MANIFEST.json'] + sorted(manifest['files']):
            target = destination / safe_member(name)
            require(target.resolve().is_relative_to(destination), 'Extraction escapes destination')
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as source, target.open('xb') as output:
                for block in iter(lambda: source.read(1024**2), b''):
                    output.write(block)
            if name != 'MANIFEST.json':
                require(target.stat().st_size == manifest['files'][name]['bytes']
                        and sha(target) == manifest['files'][name]['sha256'], 'Extracted member differs')
    return str(destination)


def decode_one(manifest_path, output, expected_record=None, extract_to=None):
    manifest_path, manifest = load_chunks(manifest_path)
    manifest['_path'] = str(manifest_path)
    expected = expected_archive_record(expected_record, manifest)
    output = Path(output).absolute()
    require(output != manifest_path, 'Decode output would replace input manifest')
    if output.exists() or output.is_symlink():
        regular_file(output)
        require(output.stat().st_size == manifest['archive_bytes'] and sha(output) == manifest['archive_sha256'],
                'Existing output differs; never overwritten')
        # Verify copied text even when the decoded archive already exists.
        digest, count = hashlib.sha256(), 0
        for raw in verified_chunks(manifest_path, manifest):
            digest.update(raw); count += len(raw)
        require(count == manifest['archive_bytes'] and digest.hexdigest() == manifest['archive_sha256'],
                'Copied ordered chunks differ from the existing verified ZIP')
    else:
        fresh_destination(output)
        partial = fresh_destination(output.with_name(output.name + '.partial'))
        output.parent.mkdir(parents=True, exist_ok=True)
        digest, count = hashlib.sha256(), 0
        with partial.open('xb') as stream:
            for raw in verified_chunks(manifest_path, manifest):
                stream.write(raw); digest.update(raw); count += len(raw)
        require(count == manifest['archive_bytes'] and digest.hexdigest() == manifest['archive_sha256'],
                'Ordered chunks do not reconstruct expected ZIP')
        verify_zip(partial, expected.get('manifest_sha256') if expected else None)
        partial.replace(output)
    internal, internal_sha = verify_zip(output, expected.get('manifest_sha256') if expected else None)
    extracted = extract_verified(output, extract_to, internal) if extract_to is not None else None
    return {'status': 'VERIFIED', 'manifest': str(manifest_path), 'manifest_sha256': sha(manifest_path),
            'archive': str(output), 'archive_bytes': output.stat().st_size, 'archive_sha256': sha(output),
            'payload_files': len(internal['files']), 'internal_manifest_sha256': internal_sha,
            'internal_metadata': internal.get('metadata'), 'expected_record': str(expected_record) if expected_record else None,
            'expected_record_sha256': sha(expected_record) if expected_record else None, 'extracted_to': extracted,
            'scope': 'transport and ZIP integrity; no model loading, inference, quality or authenticity signature'}


def auto_record(manifest_path, manifest):
    parent = manifest_path.parent.parent
    for name in ['complete.json', 'export_manifest.json']:
        candidate = parent / name
        if candidate.exists():
            record = read_json(candidate)
            if record.get('archive') == manifest['archive_name'] or manifest['archive_name'] in record.get('archives', {}):
                return candidate
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--manifest', type=Path)
    mode.add_argument('--tree', type=Path)
    mode.add_argument('--self-test', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--expected-record', type=Path)
    parser.add_argument('--extract-to', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--extract-all', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    if args.manifest:
        require(args.output is not None and args.output_dir is None and not args.extract_all,
                'Single mode requires --output, not tree options')
        manifest_path, manifest = load_chunks(args.manifest)
        expected = args.expected_record or auto_record(manifest_path, manifest)
        reports = [decode_one(manifest_path, args.output, expected, args.extract_to)]
    else:
        require(args.output_dir is not None and args.output is None and args.expected_record is None
                and args.extract_to is None, 'Tree mode requires --output-dir and no single options')
        tree = args.tree.absolute()
        require(tree.is_dir() and not tree.is_symlink(), 'A regular local tree is required')
        require(all(not p.is_symlink() for p in tree.parents), 'Symlink input-tree parent refused')
        tree = tree.resolve()
        require(all(not p.is_symlink() for p in tree.rglob('*')), 'Symlink in transfer tree refused')
        output_dir = fresh_destination(args.output_dir)
        require(not output_dir.is_relative_to(tree), 'Output directory must be outside input tree')
        paths = sorted(tree.rglob('CHUNKS.json'))
        require(bool(paths), 'No CHUNKS.json manifests found')
        output_dir.mkdir(parents=True)
        reports = []
        for manifest_path in paths:
            _, manifest = load_chunks(manifest_path)
            relative = manifest_path.parent.relative_to(tree)
            destination = output_dir / relative
            reports.append(decode_one(manifest_path, destination / manifest['archive_name'],
                auto_record(manifest_path, manifest), destination / 'extracted' if args.extract_all else None))
    report = {'status': 'VERIFIED', 'archives': reports, 'archive_count': len(reports),
              'execution': 'local stdlib only; no cloud, HTTP, model loading or inference'}
    if args.report:
        report_path = fresh_destination(args.report); report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open('x') as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


def self_test():
    """Synthetic-only fixtures; never imports any producer module or Torch."""
    import random
    def toy_archive(folder, unsafe=False):
        payload = random.Random(52).randbytes(2 * ((max(KNOWN_CHUNK_LIMITS) // 77) * 57) + 123)
        member = '../escape.bin' if unsafe else 'nested/toy.bin'
        manifest = {'metadata': {'synthetic_only': True}, 'manifest_excludes_itself': True,
                    'files': {member: {'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}}}
        archive = folder / 'toy.zip'
        with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr(member, payload)
            z.writestr('MANIFEST.json', json.dumps(manifest))
        return archive
    def armour(archive, limit=CHUNK_LIMIT):
        raw_chunk_bytes = (limit // 77) * 57
        folder = archive.parent / 'transport'; folder.mkdir()
        raw = archive.read_bytes(); prefix = archive.stem + '--' + hashlib.sha256(raw).hexdigest()[:16]
        chunks = []
        for number, offset in enumerate(range(0, len(raw), raw_chunk_bytes), 1):
            block = raw[offset:offset + raw_chunk_bytes]; encoded = base64.b64encode(block)
            text = b'\n'.join(encoded[i:i+76] for i in range(0, len(encoded), 76)) + b'\n'
            name = f'{prefix}--chunk{number:05d}.b64.txt'; (folder/name).write_bytes(text)
            chunks.append({'filename': name, 'bytes': len(text), 'sha256': hashlib.sha256(text).hexdigest(),
                           'decoded_bytes': len(block), 'decoded_sha256': hashlib.sha256(block).hexdigest()})
        record = {'archive_name': archive.name, 'archive_sha256': hashlib.sha256(raw).hexdigest(),
                  'archive_bytes': len(raw), 'chunk_count': len(chunks), 'max_encoded_bytes_per_chunk': limit,
                  'columns': 76, 'encoding': 'base64_ASCII_LF_final_newline', 'chunks': chunks}
        path = folder/'CHUNKS.json'; path.write_text(json.dumps(record)); return path
    def rejects(call):
        try: call()
        except (ValueError, KeyError, zipfile.BadZipFile): return
        raise RuntimeError('Synthetic malformed transfer was accepted')
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve(); archive = toy_archive(root); path = armour(archive)
        record = {'status':'durable_EMA_archive_and_text_chunks_verified','archive':archive.name,
                  'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,
                  'armour_manifest_sha256':sha(path)}
        expected = root/'complete.json'; expected.write_text(json.dumps(record))
        result = decode_one(path, root/'decoded.zip', expected, root/'extracted')
        require(result['payload_files']==1 and sha(root/'decoded.zip')==sha(archive), 'Synthetic positive failed')
        first = path.parent/read_json(path)['chunks'][0]['filename']; original = first.read_bytes()
        first.write_bytes(original.replace(b'\n',b'\r\n'))
        decode_one(path, root/'decoded.zip', expected)
        first.write_bytes(b'!' + original[1:]); rejects(lambda: decode_one(path,root/'decoded.zip',expected))
        first.write_bytes(original)
        # Every copied chunk can be self-consistent yet not reconstruct the ZIP.
        # Existing-output mode must aggregate the ordered bytes too.
        original_manifest = path.read_bytes(); changed = read_json(path)
        changed_raw = bytearray(base64.b64decode(original.replace(b'\n', b''), validate=True))
        changed_raw[20] ^= 1
        encoded = base64.b64encode(changed_raw)
        changed_text = b'\n'.join(encoded[i:i+76] for i in range(0, len(encoded), 76)) + b'\n'
        first.write_bytes(changed_text)
        changed['chunks'][0].update(bytes=len(changed_text), sha256=hashlib.sha256(changed_text).hexdigest(),
                                    decoded_sha256=hashlib.sha256(changed_raw).hexdigest())
        path.write_text(json.dumps(changed))
        rejects(lambda: decode_one(path, root/'decoded.zip'))
        path.write_bytes(original_manifest); first.write_bytes(original)
        data = read_json(path); data['chunks'].reverse(); path.write_text(json.dumps(data))
        rejects(lambda: decode_one(path,root/'reordered.zip'))
        unsafe = root/'unsafe';unsafe.mkdir();bad_path=armour(toy_archive(unsafe,True))
        rejects(lambda: decode_one(bad_path, unsafe/'decoded.zip'))
        require(not (root/'escape.bin').exists(),'Traversal fixture escaped')
        rejects(lambda: safe_member('/absolute')); rejects(lambda: safe_member('x\\y'))
        for limit in sorted(KNOWN_CHUNK_LIMITS):
            case = root / ('known_cap_' + str(limit)); case.mkdir()
            cap_archive = toy_archive(case); cap_manifest = armour(cap_archive, limit)
            require(len(read_json(cap_manifest)['chunks']) >= 3, 'Known-cap fixture must span chunks')
            cap_expected = case / 'complete.json'
            cap_expected.write_text(json.dumps({'status':'durable_selected_EMA_archive_and_text_chunks_verified',
                'archive':cap_archive.name,'archive_sha256':sha(cap_archive),'archive_bytes':cap_archive.stat().st_size,
                'armour_manifest_sha256':sha(cap_manifest)}))
            decode_one(cap_manifest,case/'decoded.zip',cap_expected)
            malformed = read_json(cap_manifest); malformed['max_encoded_bytes_per_chunk'] = limit + 1
            cap_manifest.write_text(json.dumps(malformed))
            rejects(lambda: decode_one(cap_manifest,case/'unknown_cap.zip'))
    print('PASS: synthetic three-cap multichunk roundtrip, old/v2 expected records, extraction, CRLF, corruption, ordering, unknown-cap and traversal; no data/models/cloud')


if __name__ == '__main__':
    main()
