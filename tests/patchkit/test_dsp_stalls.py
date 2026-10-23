"""tools/patchkit/pk_dsp_stalls.py: the cache model and the address regions (pure parts, no rig)."""

from tools.patchkit import pk_dsp_stalls as S


def test_cache_is_two_way_lru() -> None:
    c = S.Cache(nbytes=4 * 64, line=64)  # 2 sets x 2 ways
    assert c.access(0) and c.access(2)  # both into set 0
    assert not c.access(0) and not c.access(2)
    assert c.access(4)  # evicts the least recently used: line 0
    assert not c.access(2)
    assert c.access(0)


def test_far_line_maps_l2_spaces_onto_one_line() -> None:
    byte = 0x20001000
    assert S.far_line(byte) == S.far_line(byte + 63) != S.far_line(byte + 64)
    assert S.far_line(0x8000000 + (byte - 0x20000000) // 4) == S.far_line(byte)
    assert S.far_line(0xB80000 + (byte - 0x20000000) // 2) == S.far_line(byte)
    assert S.far_line(0x24F0AC) is None  # L1
    assert S.far_line(0x31000000) is None  # MMR
