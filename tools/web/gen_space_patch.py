"""Generate the browser patch for the "space 17.1" Digitakt II image (site: spacejam.github.io/space17).

    uv run python -m tools.web.gen_space_patch --stock firmware/Digitakt_II_OS1.17.syx \
        --target out/patch/plaits-20models-fast4-names-splash-48k-1.17.syx --out-dir DIR

The patch holds, for each changed section (3 and 7), a COPY/ADD delta against the DECOMPRESSED stock section. COPY
takes bytes from the stock section, which the user's own file supplies in the browser; ADD carries only our new bytes.
No stock bytes go into the patch. The generator fails when a section's literals exceed --max-literal of it (25 %) or
the section's growth plus 2 %, whichever is larger (stock bytes leaking in). The browser applier repacks the sections with a JS port of tools/firmware/aplibpack.py and rebuilds
the container exactly as tools/firmware/build.py does, so the result is the target, byte for byte.

Patch file (little endian):  "S17P" u8 version=1 u8 nsections, then per section
    u8 id, u32 out_len, u32 nops, ops: 0 varint(offset) varint(len)  = COPY
                                       1 varint(len) bytes           = ADD
Varints are LEB128. The manifest (space17.json) names the version, the base and output sha256 and sizes.
"""

import argparse
import hashlib
import json
import pathlib
import struct
import sys

from tools.patchkit.patched_image import sections_of

BLOCK = 32
CHANGED = (3, 7)
VERSION = "17.1"
OUT_NAME = "space-17.1.syx"


def diff(stock, target, block=BLOCK):
    """-> [('C', off, n) | ('A', bytes)] that rebuilds `target` from `stock`."""
    index = {}
    for p in range(0, len(stock) - block + 1, block):
        index.setdefault(stock[p : p + block], []).append(p)
    ops = []
    lit = bytearray()

    def flush():
        if lit:
            ops.append(("A", bytes(lit)))
            lit.clear()

    n, i, expect = len(target), 0, -1
    while i < n:
        src = -1
        if (
            expect >= 0
            and i + block <= n
            and stock[expect : expect + block] == target[i : i + block]
        ):
            src = expect
        elif i + block <= n:
            cands = index.get(target[i : i + block])
            if cands:
                src = max(cands[:16], key=lambda c: _run(stock, c, target, i))
        if src < 0:
            lit.append(target[i])
            i += 1
            continue
        back = 0
        while (
            back < len(lit)
            and src - back > 0
            and stock[src - back - 1] == lit[-1 - back]
        ):
            back += 1
        if back:
            del lit[len(lit) - back :]
        flush()
        length = _run(stock, src, target, i)
        ops.append(("C", src - back, length + back))
        i += length
        expect = src + length
    flush()
    return ops


def _run(a, ai, b, bi):
    m = 0
    lim = min(len(a) - ai, len(b) - bi)
    while m + 64 <= lim and a[ai + m : ai + m + 64] == b[bi + m : bi + m + 64]:
        m += 64
    while m < lim and a[ai + m] == b[bi + m]:
        m += 1
    return m


def apply_ops(stock, ops):
    out = bytearray()
    for op in ops:
        out += stock[op[1] : op[1] + op[2]] if op[0] == "C" else op[1]
    return bytes(out)


def literal_bytes(ops):
    return sum(len(o[1]) for o in ops if o[0] == "A")


def _varint(v):
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        out.append(b | (0x80 if v else 0))
        if not v:
            return bytes(out)


def _read_varint(buf, pos):
    v = shift = 0
    while True:
        b = buf[pos]
        pos += 1
        v |= (b & 0x7F) << shift
        shift += 7
        if not b & 0x80:
            return v, pos


def encode_patch(secs):
    """secs: {id: (out_len, ops)} -> patch bytes."""
    out = bytearray(b"S17P" + bytes([1, len(secs)]))
    for sid, (out_len, ops) in sorted(secs.items()):
        out += struct.pack("<BII", sid, out_len, len(ops))
        for op in ops:
            if op[0] == "C":
                out += b"\x00" + _varint(op[1]) + _varint(op[2])
            else:
                out += b"\x01" + _varint(len(op[1])) + op[1]
    return bytes(out)


def decode_patch(buf):
    assert buf[:4] == b"S17P" and buf[4] == 1
    pos, secs = 6, {}
    for _ in range(buf[5]):
        sid, out_len, nops = struct.unpack_from("<BII", buf, pos)
        pos += 9
        ops = []
        for _ in range(nops):
            tag = buf[pos]
            pos += 1
            if tag == 0:
                off, pos = _read_varint(buf, pos)
                ln, pos = _read_varint(buf, pos)
                ops.append(("C", off, ln))
            else:
                ln, pos = _read_varint(buf, pos)
                ops.append(("A", bytes(buf[pos : pos + ln])))
                pos += ln
        secs[sid] = (out_len, ops)
    return secs


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--stock", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-literal", type=float, default=0.25)
    a = ap.parse_args(argv)
    stock_raw, target_raw = (
        pathlib.Path(a.stock).read_bytes(),
        pathlib.Path(a.target).read_bytes(),
    )
    st, tg = sections_of(a.stock), sections_of(a.target)
    changed = sorted(s for s in st if st[s] != tg.get(s))
    if changed != list(CHANGED):
        raise SystemExit(
            "changed sections are %s, expected %s" % (changed, list(CHANGED))
        )
    secs, bad = {}, False
    for sid in CHANGED:
        ops = diff(st[sid], tg[sid])
        assert apply_ops(st[sid], ops) == tg[sid]
        lit = literal_bytes(ops)
        frac = lit / len(tg[sid])
        print(
            "section %d: %d bytes (stock %d), %d ops, %d literal bytes (%.1f%%)"
            % (sid, len(tg[sid]), len(st[sid]), len(ops), lit, 100 * frac)
        )
        # A section may grow by our new bytes; literals beyond that growth plus 2 % would be stock bytes.
        allowed = max(
            a.max_literal * len(tg[sid]),
            len(tg[sid]) - len(st[sid]) + 0.02 * len(tg[sid]),
        )
        bad |= lit > allowed
        secs[sid] = (len(tg[sid]), ops)
    if bad:
        raise SystemExit(
            "literals exceed the limit: stock code would leak into the patch"
        )
    patch = encode_patch(secs)
    assert {k: apply_ops(st[k], v[1]) for k, v in decode_patch(patch).items()} == {
        k: tg[k] for k in CHANGED
    }
    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "space17.patch").write_bytes(patch)
    manifest = {
        "version": VERSION,
        "output_name": OUT_NAME,
        "patch": "space17.patch",
        "base_sha256": sha(stock_raw),
        "base_bytes": len(stock_raw),
        "output_sha256": sha(target_raw),
        "output_bytes": len(target_raw),
        "patch_bytes": len(patch),
        "literal_bytes": {str(s): literal_bytes(secs[s][1]) for s in CHANGED},
    }
    (out / "space17.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("patch %d bytes" % len(patch))
    return 0


if __name__ == "__main__":
    sys.exit(main())
