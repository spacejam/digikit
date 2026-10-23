import random

from tools.web import gen_space_patch as g


def test_diff_roundtrip_and_literals_only_new():
    rnd = random.Random(1)
    stock = bytes(rnd.randrange(256) for _ in range(5000))
    new = bytes(rnd.randrange(256) for _ in range(300))
    target = stock[:1000] + new + stock[1000:3000] + b"\x01\x02" + stock[3100:]
    ops = g.diff(stock, target)
    assert g.apply_ops(stock, ops) == target
    assert g.literal_bytes(ops) < len(new) + 100
    dec = g.decode_patch(g.encode_patch({3: (len(target), ops)}))
    assert g.apply_ops(stock, dec[3][1]) == target


def test_varint():
    for v in (0, 127, 128, 3_296_324, 1 << 30):
        assert g._read_varint(g._varint(v), 0) == (v, len(g._varint(v)))
