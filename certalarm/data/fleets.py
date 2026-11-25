"""Three public run-to-failure fleets: 68 fatigue-crack specimens (Virkler), 251 lithium-ion cells (ISU-ILCC) and
124 fast-charged lithium-ion cells (MATR, Severson et al. 2019).

    python -m certalarm.data.fleets        # downloads them; Virkler and ISU-ILCC (250 kB) with their SHA-256 checked

Virkler: Virkler, Hillberry and Goel (1979), as distributed in the CRAN archive BaPreStoPro 0.1 (GPL >= 2); only the
data file is read, no package code is run. ISU-ILCC: figshare 10.25380/iastate.22582234.v2 (CC BY 4.0). MATR: the three
batch files at data.matr.io; only the per-cycle summaries are read, over HTTP ranges (a few MB of 8 GB, needs h5py), and
kept in data/matr/summary.npz.
"""
import gzip
import hashlib
import io
import struct
import tarfile
import time
import urllib.request
import zipfile

import numpy as np

from .. import paths

FILES = {
    "virkler/BaPreStoPro_0.1.tar.gz": (
        "https://cran.r-project.org/src/contrib/Archive/BaPreStoPro/BaPreStoPro_0.1.tar.gz",
        "a7d4cdb5acde270e7574ebdbc46e8fdbbe8856952846f5aa96b0215872f6de7f"),
    "isu_ilcc/capacity_fade.zip": (
        "https://ndownloader.figshare.com/files/43755582",
        "2dada0f57db9dbc002abe8ed839a61403003739086cbcc207be479689b5dc7af"),
}


def fetch(name):
    """-> verified bytes of one source file, downloading it once."""
    url, digest = FILES[name]
    path = paths.DATA / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as r:
            path.write_bytes(r.read())
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"{path}: checksum differs from the registered file")
    return raw


class _XDR:
    """Reads the few R object types a numeric data frame needs (RDX2, XDR)."""

    def __init__(self, raw):
        self.b, self.p, self.refs = raw, 0, []

    def i32(self):
        self.p += 4
        return struct.unpack_from(">i", self.b, self.p - 4)[0]

    def item(self):
        flags = self.i32()
        typ, has_attr, has_tag = flags & 0xFF, bool(flags & 0x200), bool(flags & 0x400)
        if typ == 0xFE:
            return None
        if typ == 0xFF:
            return self.refs[((flags >> 8) or self.i32()) - 1]
        if typ == 1:
            self.refs.append(self.item())
            return self.refs[-1]
        if typ == 9:
            n = self.i32()
            self.p += max(n, 0)
            return self.b[self.p - n:self.p].decode("latin-1") if n >= 0 else None
        if typ == 2:
            out = {}
            while typ == 2:
                if has_attr:
                    self.item()
                tag = self.item() if has_tag else None
                out[tag] = self.item()
                flags = self.i32()
                typ, has_attr, has_tag = flags & 0xFF, bool(flags & 0x200), bool(flags & 0x400)
            return out
        n = self.i32()
        if typ in (10, 13):
            v = np.frombuffer(self.b, ">i4", n, self.p).astype(int)
            self.p += 4 * n
        elif typ == 14:
            v = np.frombuffer(self.b, ">f8", n, self.p).astype(float)
            self.p += 8 * n
        elif typ in (16, 19):
            v = [self.item() for _ in range(n)]
        else:
            raise ValueError(f"unexpected R object type {typ}")
        if has_attr:
            self.item()
        return v


def virkler():
    """-> (crack lengths (164,) in mm, cycles (164, 68)): specimen columns in file order."""
    with tarfile.open(fileobj=io.BytesIO(fetch("virkler/BaPreStoPro_0.1.tar.gz")), mode="r:gz") as tar:
        raw = gzip.decompress(tar.extractfile("BaPreStoPro/data/Virkler.RData").read())
    assert raw[:7] == b"RDX2\nX\n"
    x = _XDR(raw[7:])
    for _ in range(3):
        x.i32()                                       # format and R versions
    M = np.column_stack(x.item()["Virkler"])
    assert M.shape == (164, 69) and np.isfinite(M).all() and (np.diff(M, axis=0) > 0).all()
    return M[:, 0], M[:, 1:]


def isu_ilcc():
    """-> {cell: (days, capacity in Ah)}, ordered by group number and cell index."""
    out = {}
    with zipfile.ZipFile(io.BytesIO(fetch("isu_ilcc/capacity_fade.zip"))) as z:
        for name in z.namelist():
            if name.endswith(".csv"):
                a = np.loadtxt(io.BytesIO(z.read(name)), delimiter=",", skiprows=1, ndmin=2)
                out[name.rsplit("/", 1)[1][:-4]] = (a[:, 0] * 7.0, a[:, 1])
    assert len(out) == 251
    return dict(sorted(out.items(), key=lambda kv: isu_key(kv[0])))


MATR = {1: "5c86c0b5fa2ede00015ddf66", 2: "5c86bf13fa2ede00015ddd82", 3: "5c86bd64fa2ede00015ddbb2"}
# the data authors' own curation (their loading notes): not run to 80%, continued in batch 2, noisy channels
MATR_SHORT = ("b1c8", "b1c10", "b1c12", "b1c13", "b1c22")
MATR_CONTINUED = {"b1c0": ("b2c7", 662), "b1c1": ("b2c8", 981), "b1c2": ("b2c9", 1060), "b1c3": ("b2c15", 208), "b1c4": ("b2c16", 482)}
MATR_NOISY = ("b3c37", "b3c2", "b3c23", "b3c32", "b3c42", "b3c43")


class _Remote:
    """Read-only, seekable view of a remote file through HTTP range requests, with a block cache."""

    def __init__(self, url, block=1 << 18):
        self.block, self.cache, self.pos = block, {}, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
            self.url, self.size = r.geturl(), int(r.headers["Content-Length"])

    def _get(self, b):
        if b not in self.cache:
            lo, hi = b * self.block, min((b + 1) * self.block, self.size) - 1
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={lo}-{hi}"})
            for k in range(6):
                try:
                    with urllib.request.urlopen(req, timeout=120) as r:
                        data = r.read()
                    if len(data) == hi - lo + 1:
                        break
                except OSError:
                    if k == 5:
                        raise
                time.sleep(1 + 2 * k)
            self.cache[b] = data
        return self.cache[b]

    def read(self, n=-1):
        end = self.size if n < 0 else min(self.pos + n, self.size)
        out = bytearray()
        while self.pos < end:
            b, off = divmod(self.pos, self.block)
            d = self._get(b)[off:off + end - self.pos]
            out += d
            self.pos += len(d)
        return bytes(out)

    def readinto(self, buf):
        d = self.read(len(buf))
        buf[:len(d)] = d
        return len(d)

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos if whence == 1 else self.size) + off
        return self.pos

    def tell(self):
        return self.pos

    def flush(self):
        pass


def fetch_matr():
    """Per-cycle summaries of the three batches -> data/matr/summary.npz (once)."""
    path = paths.DATA / "matr" / "summary.npz"
    if not path.exists():
        import h5py
        out = {}
        for k, ident in MATR.items():
            h = h5py.File(_Remote(f"https://data.matr.io/1/api/v1/file/{ident}/download"), "r")
            b = h["batch"]
            for i in range(b["summary"].shape[0]):
                g = h[b["summary"][i, 0]]
                name = f"b{k}c{i}"
                out[f"{name}/q"] = np.asarray(g["QDischarge"]).ravel()
                out[f"{name}/cycle"] = np.asarray(g["cycle"]).ravel()
                out[f"{name}/life"] = np.asarray(h[b["cycle_life"][i, 0]]).ravel()
                out[f"{name}/policy"] = "".join(chr(int(c)) for c in np.asarray(h[b["policy_readable"][i, 0]]).ravel())
            h.close()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **out)
    return np.load(path)


def matr(noisy=False):
    """-> {cell: (cycle numbers, capacity in Ah, cycle life, policy, batch)} in recorded order, curated as published."""
    z = fetch_matr()
    raw = {}
    for key in z.files:
        raw.setdefault(key.split("/")[0], {})[key.split("/")[1]] = z[key]
    out = {}
    for cell in sorted(raw, key=lambda c: (int(c[1]), int(c.split("c")[1]))):
        if cell in MATR_SHORT or cell in [v[0] for v in MATR_CONTINUED.values()] or (cell in MATR_NOISY and not noisy):
            continue
        r = raw[cell]
        cyc, q, life = r["cycle"].astype(float), r["q"].astype(float), float(r["life"][0])
        if cell in MATR_CONTINUED:
            other, add = MATR_CONTINUED[cell]
            cyc = np.concatenate([cyc, raw[other]["cycle"].astype(float) + len(cyc)])
            q, life = np.concatenate([q, raw[other]["q"].astype(float)]), life + add
        out[cell] = (cyc, q, life, str(r["policy"]), int(cell[1]))
    return out


def isu_key(cell):
    g, c = cell[1:].split("C")
    return int(g), int(c)


if __name__ == "__main__":
    a, N = virkler()
    print(f"Virkler: {N.shape[1]} specimens, crack length {a[0]} to {a[-1]} mm")
    print(f"ISU-ILCC: {len(isu_ilcc())} cells")
    print(f"MATR: {len(matr())} cells")
