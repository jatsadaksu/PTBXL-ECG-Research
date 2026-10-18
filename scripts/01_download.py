"""Restart-safe PTB-XL v1.0.3 ZIP download and selective verified extraction."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import urljoin
import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'config/dataset_config.yaml'
REPORT = ROOT / 'reports/dataset_manifest.json'
CHUNK = 1024 * 1024


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(CHUNK), b''):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def fetch_text(session, url, timeout, retries):
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=(20, timeout))
            r.raise_for_status()
            return r.content.decode('utf-8-sig')
        except (requests.RequestException, UnicodeError):
            if attempt == retries - 1:
                raise
            time.sleep(min(2 ** attempt, 30))


def checksums_parse(txt):
    found = {}
    for line in txt.splitlines():
        m = re.match(r'^([a-fA-F0-9]{64})\s+\*?(.+?)\s*$', line)
        if m:
            found[m.group(2).removeprefix('./').replace('\\', '/')] = m.group(1).lower()
    return found


def zip_member_map(zf):
    result = {}
    for info in zf.infolist():
        if info.is_dir():
            continue
        name = info.filename.replace('\\', '/')
        parts = PurePosixPath(name).parts
        if '..' in parts or name.startswith('/'):
            raise RuntimeError('Unsafe ZIP entry: ' + name)
        # ZIP usually contains a top-level directory; find the dataset-relative suffix.
        for idx, part in enumerate(parts):
            if part in ('records100', 'records500'):
                rel = '/'.join(parts[idx:])
                break
            if part in ('ptbxl_database.csv', 'scp_statements.csv'):
                rel = '/'.join(parts[idx:])
                break
        else:
            continue
        if rel in result:
            raise RuntimeError('Duplicate ZIP entry: ' + rel)
        result[rel] = info
    return result


def remote_probe(session, url, timeout):
    # A GET with Range is more reliable than HEAD for archive endpoints.
    with session.get(url, headers={'Range': 'bytes=0-0', 'Accept-Encoding': 'identity'},
                     stream=True, timeout=(20, timeout), allow_redirects=True) as r:
        r.raise_for_status()
        cr = r.headers.get('Content-Range', '')
        m = re.fullmatch(r'bytes 0-0/(\d+)', cr)
        size = int(m.group(1)) if r.status_code == 206 and m else (
            int(r.headers['Content-Length']) if r.status_code == 200 and r.headers.get('Content-Length', '').isdigit() else None)
        return {'size': size, 'etag': r.headers.get('ETag'),
                'last_modified': r.headers.get('Last-Modified'),
                'range_supported': r.status_code == 206 and bool(m)}


def download_archive(session, url, part, archive, state, timeout, retries):
    if archive.exists() and zipfile.is_zipfile(archive):
        return
    probe = remote_probe(session, url, timeout)
    prior = state.get('archive_source', {})
    offset = part.stat().st_size if part.exists() else 0
    if offset:
        if not probe['range_supported']:
            raise RuntimeError('Server does not support byte-range resume; partial ZIP retained unchanged.')
        if prior.get('size') and probe['size'] != prior['size']:
            raise RuntimeError('Remote ZIP size changed; refusing unsafe resume.')
        if prior.get('etag') and probe['etag'] != prior['etag']:
            raise RuntimeError('Remote ETag changed; refusing unsafe resume.')
        if prior.get('last_modified') and probe['last_modified'] != prior['last_modified']:
            raise RuntimeError('Remote Last-Modified changed; refusing unsafe resume.')
        if not prior:
            raise RuntimeError('Partial ZIP has no source checkpoint; refusing unsafe resume.')
    else:
        state['archive_source'] = probe
        atomic_json(REPORT, state)
    for attempt in range(retries):
        offset = part.stat().st_size if part.exists() else 0
        headers = {'Accept-Encoding': 'identity'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
            validator = probe['etag'] or probe['last_modified']
            if validator:
                headers['If-Range'] = validator
        try:
            with session.get(url, headers=headers, stream=True, timeout=(20, timeout)) as r:
                if offset:
                    expected = f'bytes {offset}-'
                    if r.status_code != 206 or not r.headers.get('Content-Range', '').startswith(expected):
                        raise RuntimeError('Server rejected exact resume range; partial ZIP preserved.')
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', r.headers['Content-Range'])
                    if not match or int(match.group(1)) != offset or (probe['size'] and int(match.group(3)) != probe['size']):
                        raise RuntimeError('Invalid Content-Range; partial ZIP preserved.')
                else:
                    r.raise_for_status()
                    if r.status_code not in (200, 206):
                        raise RuntimeError('Unexpected HTTP status')
                with open(part, 'ab' if offset else 'wb') as f:
                    for chunk in r.iter_content(CHUNK):
                        if chunk:
                            f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
            if probe['size'] is not None and part.stat().st_size != probe['size']:
                raise IOError('Incomplete transfer; will resume')
            if not zipfile.is_zipfile(part):
                raise RuntimeError('Downloaded file is not a valid ZIP container')
            os.replace(part, archive)
            state['archive_downloaded'] = True
            atomic_json(REPORT, state)
            return
        except (requests.RequestException, OSError) as exc:
            print(f'Transfer interrupted: {exc}; retry {attempt + 1}/{retries}', flush=True)
            if attempt == retries - 1:
                raise
            time.sleep(min(2 ** attempt, 30))


def extract_verified(zf, info, dest, expected, state):
    if dest.is_file() and sha256(dest) == expected:
        return 'verified_existing'
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_name(dest.name + '.part')
    # ZIP members cannot generally resume decompression at an arbitrary byte.
    # Re-extract only the interrupted member, never a completed verified member.
    with zf.open(info) as source, open(temp, 'wb') as target:
        for chunk in iter(lambda: source.read(CHUNK), b''):
            target.write(chunk)
        target.flush()
        os.fsync(target.fileno())
    if sha256(temp) != expected:
        temp.unlink(missing_ok=True)
        raise RuntimeError(f'Upstream SHA256 mismatch: {dest}')
    os.replace(temp, dest)
    state['verified_files'] = state.get('verified_files', 0) + 1
    return 'extracted_verified'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--keep-zip', action='store_true', help='Keep verified source ZIP in raw directory')
    args = parser.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    ds, dl = cfg['dataset'], cfg['download']
    if ds['version'] != '1.0.3' or ds['sampling_rate'] != 100:
        raise RuntimeError('Configuration violates locked dataset version or sampling rate')
    raw = ROOT / ds['raw_dir']
    raw.mkdir(parents=True, exist_ok=True)
    base = ds['base_url'].rstrip('/') + '/'
    url = 'https://physionet.org/content/ptb-xl/get-zip/1.0.3/'
    archive = raw / 'ptb-xl-1.0.3.zip'
    part = raw / 'ptb-xl-1.0.3.zip.part'
    state = json.loads(REPORT.read_text(encoding='utf-8')) if REPORT.exists() else {}
    if state.get('archive_url') not in (None, url):
        raise RuntimeError('Checkpoint belongs to a different source archive')
    state.update({'dataset': 'PTB-XL', 'version': ds['version'], 'archive_url': url,
                  'status': 'in_progress', 'complete_download_claim': False})
    atomic_json(REPORT, state)
    session = requests.Session()
    timeout, retries = int(dl['timeout_seconds']), int(dl['retries'])
    checksums = checksums_parse(fetch_text(session, urljoin(base, 'SHA256SUMS.txt'), timeout, retries))
    if not checksums:
        raise RuntimeError('Missing upstream SHA256SUMS.txt')
    print('Downloading/resuming complete 1.7 GB ZIP...', flush=True)
    download_archive(session, url, part, archive, state, timeout, retries)
    print('Inspecting ZIP and extracting only locked research files...', flush=True)
    with zipfile.ZipFile(archive) as zf:
        members = zip_member_map(zf)
        for rel in ('ptbxl_database.csv', 'scp_statements.csv'):
            if rel not in members or rel not in checksums:
                raise RuntimeError('Missing required metadata in ZIP/checksum list: ' + rel)
            extract_verified(zf, members[rel], raw / rel, checksums[rel], state)
            atomic_json(REPORT, state)
        metadata = pd.read_csv(raw / 'ptbxl_database.csv')
        if len(metadata) != ds['expected_records'] or metadata['ecg_id'].duplicated().any():
            raise RuntimeError('Metadata count or ecg_id invalid')
        paths = metadata['filename_lr'].dropna().astype(str).tolist()
        if len(paths) != len(metadata) or len(set(paths)) != len(paths):
            raise RuntimeError('Missing/duplicate filename_lr')
        requested = ['ptbxl_database.csv', 'scp_statements.csv']
        for record in sorted(paths):
            if not re.fullmatch(r'records100/\d{5}/\d+_lr', record):
                raise RuntimeError('Unexpected record path: ' + record)
            requested.extend((record + '.hea', record + '.dat'))
        for i, rel in enumerate(requested[2:], 1):
            if rel not in members or rel not in checksums:
                raise RuntimeError('Missing ZIP member/checksum: ' + rel)
            result = extract_verified(zf, members[rel], raw / rel, checksums[rel], state)
            if i % 100 == 0 or i == len(requested) - 2:
                state['processed_waveform_files'] = i
                atomic_json(REPORT, state)
                print(f'{i}/{len(requested)-2}: {result}', flush=True)
    state.update({'status': 'download_and_sha256_verified', 'metadata_rows': len(metadata),
                  'expected_waveform_files': len(requested) - 2,
                  'complete_download_claim': True,
                  'note': 'WFDB signal integrity and patient split audit remain pending in 02_verify_dataset.py'})
    atomic_json(REPORT, state)
    if not args.keep_zip:
        archive.unlink()
    print('Completed: all required records100 and metadata SHA256-verified.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr, flush=True)
        sys.exit(1)
