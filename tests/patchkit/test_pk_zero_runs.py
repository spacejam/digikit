"""pk_build.check_zero_runs: a loaded data section with a zero run over 4 KB (a zero-initialised C static: selcc emits it as
loaded data, so the zeros go into the DSP boot stream) is refused with a pointer to `.PKBSS`; `.PKBSS` and an explicit
`.PKDATA` are fine; small statics and non-zero tables are fine."""

from __future__ import annotations

import pytest

from tools.patchkit import pk_build, pk_sharc


def c(src: str) -> pk_sharc.SharcObject:
    return pk_sharc.compile_c(src)


def test_big_zero_static_is_refused() -> None:
    obj = c("static unsigned int a[2048];\nint f(void) { return (int)a[3]; }\n")
    with pytest.raises(pk_build.BuildError, match=r"\.PKBSS"):
        pk_build.check_zero_runs(obj)


def test_zero_run_at_the_limit_passes_and_over_it_fails() -> None:
    ok = c(
        "static unsigned int a[1024];\nint f(void) { return (int)a[3]; }\n"
    )  # 4,096 bytes
    pk_build.check_zero_runs(ok)
    over = c("static unsigned int a[1025];\nint f(void) { return (int)a[3]; }\n")
    with pytest.raises(pk_build.BuildError):
        pk_build.check_zero_runs(over)


def test_nonzero_table_and_small_static_pass() -> None:
    obj = c(
        "static const unsigned int t[3000] = {1u, 2u, 3u, 4u, 5u};\n"
        "static unsigned int z[100];\n"
        "int f(void) { return (int)(t[1] + z[2]) + (int)t[2999]; }\n"
    )
    # a table whose tail is zeros is a zero run: the 3000-word initialiser is 12 KB of mostly zeros
    with pytest.raises(pk_build.BuildError):
        pk_build.check_zero_runs(obj)
    small = c("static unsigned int z[100];\nint f(void) { return (int)z[2]; }\n")
    pk_build.check_zero_runs(small)


def test_pkbss_and_pkdata_are_exempt() -> None:
    obj = pk_sharc.assemble(
        ".SECTION/PM seg_pmco;\n.GLOBAL f.;\nf.:\n    R0 = 0;\n    RTS;\n"
        ".SECTION/DM seg_dmda;\n.PKBSS big_bss, 65536;\n.PKDATA big_data, 65536;\n"
    )
    pk_build.check_zero_runs(obj)


def test_custom_limit() -> None:
    obj = c("static unsigned int a[100];\nint f(void) { return (int)a[3]; }\n")
    with pytest.raises(pk_build.BuildError):
        pk_build.check_zero_runs(obj, limit=64)
