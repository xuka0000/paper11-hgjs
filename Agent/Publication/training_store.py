"""Atomic, hash-checked training records and an exclusive process lock."""
from contextlib import contextmanager
from hashlib import sha256
import gzip
import io
import json
import os
from pathlib import Path
import torch


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return sha256(canonical(value)).hexdigest()


def stream_seed(base, *keys):
    return int(digest([base, *keys])[:15], 16)


class TrainingStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, name):
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Record path escapes experiment directory')
        return path

    def exists(self, name):
        return self.path(name).exists()

    def _write(self, name, data, mutable=False):
        path = self.path(name)
        if path.exists() and not mutable:
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.pending')
        with temporary.open('wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    def put(self, name, payload, mutable=False):
        raw = canonical({'payload_sha256': digest(payload), 'payload': payload})
        self._write(name, gzip.compress(raw, mtime=0) if name.endswith('.gz') else raw, mutable)

    def get(self, name):
        raw = self.path(name).read_bytes()
        record = json.loads(gzip.decompress(raw) if name.endswith('.gz') else raw)
        if digest(record['payload']) != record['payload_sha256']:
            raise ValueError('Corrupt training record: ' + name)
        return record['payload']

    def save_model(self, name, payload):
        stream = io.BytesIO()
        torch.save(payload, stream)
        raw = stream.getvalue()
        # Model and its hash are one atomic envelope, preventing a half-written
        # model/hash pair after a crash.
        self._write(name, sha256(raw).hexdigest().encode() + b'\n' + raw)

    def load_model(self, name):
        expected, raw = self.path(name).read_bytes().split(b'\n', 1)
        if sha256(raw).hexdigest().encode() != expected:
            raise ValueError('Corrupt model checkpoint: ' + name)
        return torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)

    @contextmanager
    def lock(self):
        path = self.path('process.lock')
        with path.open('a+b') as stream:
            stream.seek(0, 2)
            if stream.tell() == 0:
                stream.write(b'0'); stream.flush()
            stream.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                stream.seek(0)
                if os.name == 'nt':
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
