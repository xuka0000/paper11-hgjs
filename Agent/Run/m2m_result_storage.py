"""Lossless, versioned result artifacts for the frozen M2M queue.

The reader accepts old plain JSON without rewriting it. New records use an
atomic gzip file and a completed-row hash of the exact uncompressed bytes.
"""
import errno
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil


FORMAT = 'json-gzip-v1'
MIN_FREE_BYTES = 2 * 1024**3


def digest_bytes(data):
    return sha256(data).hexdigest()


def result_path(out, task_id, *, compressed=True):
    return Path(out)/'results'/(task_id + ('.json.gz' if compressed else '.json'))


def admit_disk(path, minimum=MIN_FREE_BYTES):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    if free < minimum:
        raise OSError(errno.ENOSPC, f'Result storage admission requires {minimum} free bytes; observed {free}', str(path))
    return free


def write_result(out, task_id, result):
    directory = Path(out)/'results'
    admit_disk(directory)
    path = result_path(out,task_id)
    if path.exists() or result_path(out,task_id,compressed=False).exists():
        raise ValueError('Result already exists; single attempt cannot overwrite evidence')
    payload = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(',',':')).encode('utf-8')
    admit_disk(directory, max(MIN_FREE_BYTES, len(payload)*2))
    packed = gzip.compress(payload,compresslevel=1,mtime=0)
    temp = path.with_suffix(path.suffix+'.tmp')
    if temp.exists():
        raise ValueError('Preserved temporary result requires diagnosis')
    try:
        with temp.open('xb') as stream:
            stream.write(packed)
            stream.flush(); os.fsync(stream.fileno())
        if path.exists():
            raise ValueError('Result appeared during write')
        os.replace(temp,path)
    except Exception:
        # A partial temp is forensic evidence, especially after ENOSPC.
        raise
    return dict(path=str(path),format=FORMAT,sha256=digest_bytes(payload),
                storage_sha256=digest_bytes(packed),uncompressed_bytes=len(payload),stored_bytes=len(packed))


def read_result(out,task_id,completed_row=None):
    old = result_path(out,task_id,compressed=False)
    new = result_path(out,task_id)
    if old.exists() and new.exists():
        raise ValueError('Ambiguous old/new result artifacts')
    if old.exists():
        data = old.read_bytes(); storage = data; kind = 'json-v1'
    elif new.exists():
        storage = new.read_bytes(); data = gzip.decompress(storage); kind = FORMAT
    else:
        raise FileNotFoundError(task_id)
    raw_sha = digest_bytes(data); storage_sha = digest_bytes(storage)
    if completed_row is not None:
        if raw_sha != completed_row['sha256']:
            raise ValueError('Changed completed result '+task_id)
        expected_format = completed_row.get('storage_format','json-v1')
        if expected_format != kind:
            raise ValueError('Completed result storage format mismatch '+task_id)
        if 'storage_sha256' in completed_row and storage_sha != completed_row['storage_sha256']:
            raise ValueError('Changed compressed result '+task_id)
    return json.loads(data),dict(format=kind,sha256=raw_sha,storage_sha256=storage_sha,
        uncompressed_bytes=len(data),stored_bytes=len(storage),path=str(old if old.exists() else new))


def stream_sha256(path,pulse=None):
    """Hash an artifact in bounded memory and refresh an optional preflight beat."""
    hashed=sha256();size=0
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            hashed.update(chunk);size+=len(chunk)
            if pulse:pulse()
    return hashed.hexdigest(),size


def verify_completed_artifact(out,task_id,completed_row,pulse=None):
    """Verify exact raw and stored hashes without parsing or retaining JSON."""
    old=result_path(out,task_id,compressed=False)
    new=result_path(out,task_id)
    if old.exists() and new.exists():
        raise ValueError('Ambiguous old/new result artifacts')
    kind=completed_row.get('storage_format','json-v1')
    if kind=='json-v1' and old.exists():
        raw,size=stream_sha256(old,pulse)
        stored=raw;stored_size=size
    elif kind==FORMAT and new.exists():
        stored,stored_size=stream_sha256(new,pulse)
        hashed=sha256();size=0
        with gzip.open(new,'rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b''):
                hashed.update(chunk);size+=len(chunk)
                if pulse:pulse()
        raw=hashed.hexdigest()
    else:
        raise ValueError('Completed result storage format mismatch '+task_id)
    if raw!=completed_row['sha256']:
        raise ValueError('Changed completed result '+task_id)
    if 'storage_sha256' in completed_row and stored!=completed_row['storage_sha256']:
        raise ValueError('Changed compressed result '+task_id)
    if 'uncompressed_bytes' in completed_row and size!=completed_row['uncompressed_bytes']:
        raise ValueError('Changed completed result byte count '+task_id)
    if 'stored_bytes' in completed_row and stored_size!=completed_row['stored_bytes']:
        raise ValueError('Changed stored result byte count '+task_id)
    return dict(format=kind,sha256=raw,storage_sha256=stored,
                uncompressed_bytes=size,stored_bytes=stored_size)
