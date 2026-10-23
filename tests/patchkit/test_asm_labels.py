"""Duplicate assembly labels fail closed (tools/patchkit/pk_sharc.py duplicate_labels, called by assemble before the
assembler runs): selas accepts a redefined label and binds every reference to one definition, so a label shared by two
of the sources pk_build.dsp_object concatenates sent snare_synth_kernel's jumps into the string-synth kernel (lane
opt-o3). No toolchain needed."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from tools.patchkit import pk_build, pk_sharc

ROOT = Path(__file__).resolve().parents[2]


def test_a_redefined_label_is_found_with_both_lines() -> None:
    src = "a:\n  R0 = 1;\n// b: in a comment\n/* c:\n*/\n  b :\n x: R1 = 2;\nb:\n"
    assert pk_sharc.duplicate_labels(src) == [("b", 6, 8)]


def test_labels_in_exclusive_conditional_branches_are_allowed() -> None:
    src = "#if M == 1\n_c:\n#else\n_c:\n#endif\n"
    assert pk_sharc.duplicate_labels(src) == []
    nested = "#if A\n#if B\n_c:\n#endif\n_c:\n#endif\n"
    assert pk_sharc.duplicate_labels(nested) == [("_c", 3, 5)]
    after = "#if A\n_c:\n#endif\n_c:\n"
    assert pk_sharc.duplicate_labels(after) == [("_c", 2, 4)]


def test_dsp_object_refuses_a_label_in_two_files_with_file_and_line(
    tmp_path: Path,
) -> None:
    one, two = tmp_path / "one.s", tmp_path / "two.s"
    one.write_text(".SECTION/PM seg_pmco;\n_done:\n    NOP;\n")
    two.write_text("// two\n_done:\n    NOP;\n")
    with pytest.raises(pk_sharc.PkSharcError) as e:
        pk_build.dsp_object([one, two])
    assert f"{two}:2:" in str(e.value) and f"{one}:2:" in str(e.value), str(e.value)


def test_every_module_objects_assembly_has_unique_labels() -> None:
    """Backstop over the committed modules: every object's assembly sources, concatenated as dsp_object does."""
    checked = 0
    for toml in sorted((ROOT / "modules").glob("*/*/module.toml")):
        spec = tomllib.loads(toml.read_text())
        for obj in spec.get("object", []):
            asm = [
                (toml.parent / s).resolve()
                for s in obj.get("sources", [])
                if s.endswith(".s")
            ]
            if len(asm) < 2:
                continue
            text = "\n".join(p.read_text() for p in asm)
            assert pk_sharc.duplicate_labels(text) == [], (toml, [p.name for p in asm])
            checked += 1
    assert checked >= 2
