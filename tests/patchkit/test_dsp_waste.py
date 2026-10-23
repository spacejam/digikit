"""tools/patchkit/dsp_waste.py: each detector on a small synthetic instruction sequence, one positive and one negative."""

from __future__ import annotations

from typing import Any

from tools.patchkit import dsp_waste as W

I6 = "I6"


def mk(pc: int, text: str, **k: Any) -> W.Op:
    """An Op from its text; registers from the text unless given."""
    defs, uses = W._text_effects(text)
    unit = k.pop("unit", "ALU" if not k.get("mem") and not k.get("branch") else None)
    base: dict[str, Any] = dict(
        pc=pc,
        nw=1,
        text=text,
        form=k.pop("form", "2a"),
        defs=frozenset(k.pop("defs", defs)),
        uses=frozenset(k.pop("uses", uses)),
        unit=unit,
        flags_w=unit is not None,
        shape=W.SHAPE.sub("r", text),
    )
    base.update(k)
    return W.Op(**base)


def ld(pc: int, reg: str, off: int, base: str = I6) -> W.Op:
    m = W.Mem("DM", "load", base, ("DM", base, off), reg, False, off)
    return mk(
        pc,
        f"{reg} = DM({base} - {-off})",
        form="15b",
        defs={reg},
        uses={base},
        mem=(m,),
    )


def st(pc: int, reg: str, off: int, base: str = I6) -> W.Op:
    m = W.Mem("DM", "store", base, ("DM", base, off), reg, False, off)
    return mk(
        pc,
        f"DM({base} - {-off}) = {reg}",
        form="15b",
        defs=set(),
        uses={base, reg},
        mem=(m,),
    )


def prog(
    ops: list[W.Op], n: float = 10.0, pairs: dict | None = None, funcs=None
) -> W.Program:
    return W.Program(
        ops, {o.pc: n for o in ops}, pairs or {}, funcs=funcs or [(ops[0].pc, "f")]
    )


def names(fs: list[W.Finding]) -> set[str]:
    return {f.detector for f in fs}


def test_build_op_parses_branches_and_text_effects() -> None:
    j = W.build_op(0x10, 2, "8a_rel", {}, "JUMP IF NE non-delayed  target=0x20")
    assert (j.branch, j.target, j.delayed, j.cond) == ("cond_jump", 0x20, False, True)
    c = W.build_op(0x10, 4, "25a_direct", {}, "CALL (linked, delayed)  target=0x30")
    assert (c.branch, c.target, c.delayed) == ("call", 0x30, True)
    d, u = W._text_effects("F1 = F2 * F4, F15 = F8 + F15; DM(I4 + 6) = R8")
    assert d == {"R1", "R15"} and u == {"R2", "R4", "R8", "R15"}


def test_memory_spill_reload_and_redundant_load() -> None:
    p = prog([st(0, "R3", -4), mk(1, "R5 = add(R1, R2)"), ld(2, "R3", -4)])
    (f,) = [x for x in W.det_memory(p) if x.detector == "spill_reload"]
    assert f.saving_each == 1.0 and f.confidence == "high"
    p = prog(
        [
            ld(0, "R3", -4),
            mk(1, "R4 = add(R3, R2)"),
            ld(2, "R6", -4),
            mk(3, "R3 = add(R6, R6)"),
            mk(4, "R6 = add(R3, R3)"),
        ]
    )
    assert "redundant_load" in names(W.det_memory(p))
    # negative: the register holding the value is overwritten before the reload
    p = prog([st(0, "R3", -4), mk(1, "R3 = add(R1, R2)"), ld(2, "R3", -4)])
    assert not W.det_memory(p)


def test_memory_redundant_store_and_alias_confidence() -> None:
    p = prog([ld(0, "R6", -9), mk(1, "R7 = add(R6, R1)"), st(2, "R6", -9)])
    assert names(W.det_memory(p)) == {"redundant_store"}
    other = mk(
        1,
        "DM(I4, M6) = R2",
        form="3a",
        defs=set(),
        uses={"I4", "R2"},
        mem=(W.Mem("DM", "store", "I4", None, "R2", False),),
    )
    p = prog([ld(0, "R6", -9), other, st(2, "R6", -9)])
    (f,) = W.det_memory(p)
    assert f.confidence == "low"  # a store through another pointer might alias
    p = prog([ld(0, "R6", -9), mk(1, "R6 = add(R6, R1)"), st(2, "R6", -9)])
    assert not W.det_memory(p)


def test_copy_removal() -> None:
    ops = [mk(0, "R1 = pass(R7)"), mk(1, "R2 = add(R1, R4)"), mk(2, "R1 = add(R3, R3)")]
    assert names(W.det_copy_prop(prog(ops))) == {"copy_removal"}
    # negative: the source changes before the copy is dead
    ops = [
        mk(0, "R1 = pass(R7)"),
        mk(1, "R7 = add(R3, R3)"),
        mk(2, "R2 = add(R1, R4)"),
        mk(3, "R1 = add(R3, R3)"),
    ]
    assert not W.det_copy_prop(prog(ops))


def test_dead_definition() -> None:
    ops = [
        mk(0, "R4 = add(R1, R2)"),
        mk(1, "R4 = add(R5, R6)"),
        mk(2, "R7 = add(R4, R4)"),
    ]
    (f,) = W.det_dead_def(prog(ops))
    assert f.pc == 0
    ops = [
        mk(0, "R4 = add(R1, R2)"),
        mk(1, "R7 = add(R4, R4)"),
        mk(2, "R4 = add(R5, R6)"),
    ]
    assert not W.det_dead_def(prog(ops))


def _loop(counter_used: bool = False) -> list[W.Op]:
    ops = [
        mk(0x10, "R3 = add(R3, R4)"),
        mk(0x11, "R14 = dec(R14)"),
        mk(
            0x12,
            "JUMP IF NE non-delayed  target=0x10",
            form="8a_rel",
            branch="cond_jump",
            target=0x10,
            cond=True,
            unit=None,
            flags_r=True,
        ),
        mk(0x13, "R0 = pass(R3)"),
    ]
    if counter_used:
        ops[0] = mk(0x10, "R3 = add(R3, R14)")
    return ops


def test_sw_loop_to_hw_counted_and_not() -> None:
    p = W.Program(
        _loop(),
        {0x10: 100, 0x11: 100, 0x12: 100, 0x13: 1},
        {(0x12, 0x13): 1},
        funcs=[(0x10, "f")],
    )
    (f,) = W.det_sw_loops(p)
    assert (
        f.confidence == "medium" and f.saving_each > 3
    )  # counter + branch + 2 stall cycles per iteration
    p = W.Program(
        _loop(True),
        {0x10: 100, 0x11: 100, 0x12: 100, 0x13: 1},
        {(0x12, 0x13): 1},
        funcs=[(0x10, "f")],
    )
    (f,) = W.det_sw_loops(p)
    assert f.confidence == "low"  # the counter is also an index
    ops = _loop()
    ops[1] = mk(0x11, "R14 = add(R14, R9)")  # not a unit step: no pattern
    p = W.Program(
        ops,
        {0x10: 100, 0x11: 100, 0x12: 100, 0x13: 1},
        {(0x12, 0x13): 1},
        funcs=[(0x10, "f")],
    )
    assert not W.det_sw_loops(p)


def _callee(extra: int) -> list[W.Op]:
    ops = [
        mk(0x100, "DM(I7, M7) = 0x0", form="16a", defs={"I7"}, uses={"I7"}, unit=None)
    ]
    ops += [mk(0x101 + i, f"R{i % 4} = add(R1, R2)") for i in range(extra)]
    ops.append(
        mk(
            0x101 + extra,
            "RETURN",
            form="9b_abs",
            branch="return",
            delayed=True,
            unit=None,
        )
    )
    return ops


def test_call_inline_small_leaf_and_nothing() -> None:
    call = mk(
        0x10,
        "CALL (linked, delayed)  target=0x100",
        form="25a_direct",
        branch="call",
        target=0x100,
        delayed=True,
        unit=None,
    )
    p = prog(
        [call, mk(0x11, "NOP"), mk(0x12, "NOP"), *_callee(4)],
        funcs=[(0x10, "f"), (0x100, "g")],
    )
    (f,) = W.det_calls(p)
    assert f.extra["leaf"] and f.confidence == "medium" and f.saving_each >= 3
    p = prog([mk(0x10, "R1 = add(R1, R1)")])  # no call at all
    assert not W.det_calls(p)


def test_fold_transfer() -> None:
    t = mk(
        1,
        "R5 = DM(I4, M5)",
        form="3c",
        defs={"R5"},
        uses={"I4"},
        unit=None,
        mem=(W.Mem("DM", "load", "I4", ("DM", "I4", 0), "R5", False, 0),),
    )
    p = prog([mk(0, "R1 = add(R2, R3)"), t])
    (f,) = W.det_fold_transfer(p)
    assert f.detector == "fold_transfer"
    # negative: the move loads a register the compute reads... fine; but the compute result feeds the address
    c = mk(0, "R4 = add(R2, R3)", defs={"R4"}, uses={"R2", "R3"})
    t2 = mk(
        1,
        "R5 = DM(I4, M5)",
        form="3c",
        defs={"R5"},
        uses={"I4", "R4"},
        unit=None,
        mem=(W.Mem("DM", "load", "I4", ("DM", "I4", 0), "R5", False, 0),),
    )
    assert not W.det_fold_transfer(prog([c, t2]))
    # a stack slot beyond the 6-bit field is reported as the far variant
    assert names(
        W.det_fold_transfer(prog([mk(0, "R1 = add(R2, R3)"), ld(1, "R5", -60)]))
    ) == {"fold_transfer_far"}


def test_compute_pairs_simd_and_multifunction() -> None:
    ops = [mk(0, "F1 = fabs(F2)"), mk(1, "F5 = fabs(F6)")]
    assert names(W.det_compute_pairs(prog(ops))) == {"simd_pair"}
    ops = [mk(0, "F1 = fabs(F2)"), mk(1, "F5 = fabs(F1)")]  # dependent
    assert not W.det_compute_pairs(prog(ops))
    mul = mk(0, "F8 = F1 * F4", unit="MULT")
    add = mk(1, "F9 = F10 + F12", unit="ALU")
    (f,) = W.det_compute_pairs(prog([mul, add]))
    assert f.detector == "multifunction_pair" and f.confidence == "medium"


def test_fill_delay_slots() -> None:
    jump = mk(
        2,
        "JUMP non-delayed  target=0x40",
        form="8a_rel",
        branch="jump",
        target=0x40,
        unit=None,
    )
    p = prog([mk(0, "R1 = add(R2, R3)"), mk(1, "R4 = add(R5, R6)"), jump])
    (f,) = W.det_delay_slots(p)
    assert f.saving_each == 2.0
    # negative: the instruction before the jump writes the register an indirect jump reads
    ind = mk(
        2,
        "JUMP non-delayed  target=indirect PM(I4, M5)",
        form="9b_abs",
        branch="indirect",
        uses={"I4", "M5"},
        unit=None,
    )
    p = prog([mk(0, "R1 = add(R2, R3)"), mk(1, "R4 = pass(R5)", defs={"I4"}), ind])
    assert not W.det_delay_slots(p)


def test_nop_outside_delay_slot() -> None:
    n = mk(1, "NOP", unit=None, nop=True)
    assert names(W.det_nops(prog([mk(0, "R1 = add(R2, R3)"), n]))) == {"nop"}
    call = mk(
        0,
        "CALL (linked, delayed)  target=0x100",
        form="25a_direct",
        branch="call",
        target=0x100,
        delayed=True,
        unit=None,
    )
    assert not W.det_nops(
        prog(
            [call, mk(1, "NOP", unit=None, nop=True), mk(2, "NOP", unit=None, nop=True)]
        )
    )


def test_convert_divide_roundtrip_and_hoist() -> None:
    ops = [mk(0, "F1 = float(R2)"), mk(1, "R3 = fix(F1)")]
    assert names(W.det_convert_divide(prog(ops))) == {"int_float_roundtrip"}
    assert not W.det_convert_divide(
        prog([mk(0, "F1 = float(R2)"), mk(1, "F3 = fadd(F1, F1)")])
    )
    body = [
        mk(0x10, "F0 = recips(F8)"),
        mk(0x11, "F2 = F0 * F4"),
        mk(0x12, "R14 = dec(R14)"),
        mk(
            0x13,
            "JUMP IF NE non-delayed  target=0x10",
            form="8a_rel",
            branch="cond_jump",
            target=0x10,
            cond=True,
            unit=None,
            flags_r=True,
        ),
        mk(0x14, "R0 = pass(R2)"),
    ]
    p = W.Program(
        body,
        {0x10: 50, 0x11: 50, 0x12: 50, 0x13: 50, 0x14: 1},
        {(0x13, 0x14): 1},
        funcs=[(0x10, "f")],
    )
    assert "divide_hoist" in names(W.det_convert_divide(p))
    body[0] = mk(0x10, "F8 = recips(F8)")  # the divisor changes inside the loop
    p = W.Program(
        body,
        {0x10: 50, 0x11: 50, 0x12: 50, 0x13: 50, 0x14: 1},
        {(0x13, 0x14): 1},
        funcs=[(0x10, "f")],
    )
    assert "divide_hoist" not in names(W.det_convert_divide(p))


def test_redundant_compute() -> None:
    ops = [
        mk(0, "R6 = lshift(R3, R5)"),
        mk(1, "R7 = add(R6, R1)"),
        mk(2, "R8 = lshift(R3, R5)"),
    ]
    assert "redundant_compute" in names(W.det_redundant_compute(prog(ops)))
    ops = [
        mk(0, "R6 = lshift(R3, R5)"),
        mk(1, "R3 = add(R6, R1)"),
        mk(2, "R8 = lshift(R3, R5)"),
    ]  # operand changed
    assert not W.det_redundant_compute(prog(ops))
    ops = [
        mk(0, "R6 = lshift(R3, R5)"),
        mk(1, "R6 = add(R6, R1)"),
        mk(2, "R8 = lshift(R3, R5)"),
    ]
    assert names(W.det_redundant_compute(prog(ops))) == {"recompute_after_clobber"}


def test_address_arith() -> None:
    mv = mk(2, "I4 = R7", form="5b_move", defs={"I4"}, uses={"R7"}, unit=None)
    acc = mk(
        3,
        "R1 = DM(I4, M5)",
        form="3c",
        defs={"R1"},
        uses={"I4"},
        unit=None,
        mem=(W.Mem("DM", "load", "I4", ("DM", "I4", 0), "R1", False, 0),),
    )
    ops = [mk(0, "R6 = lshift(R3, R5)"), mk(1, "R7 = add(R9, R6)"), mv, acc]
    assert names(W.det_address_arith(prog(ops))) == {"address_arith"}
    assert not W.det_address_arith(
        prog([mk(0, "R6 = lshift(R3, R5)"), mk(1, "R7 = add(R9, R6)")])
    )


def test_invariant_stack_reload() -> None:
    body = [
        ld(0x10, "R1", -4),
        mk(0x11, "R2 = add(R2, R1)"),
        mk(0x12, "R14 = dec(R14)"),
        mk(
            0x13,
            "JUMP IF NE non-delayed  target=0x10",
            form="8a_rel",
            branch="cond_jump",
            target=0x10,
            cond=True,
            unit=None,
            flags_r=True,
        ),
        mk(0x14, "R0 = pass(R2)"),
    ]
    counts = {0x10: 50, 0x11: 50, 0x12: 50, 0x13: 50, 0x14: 1}
    p = W.Program(body, counts, {(0x13, 0x14): 1}, funcs=[(0x10, "f")])
    (f,) = W.det_invariant_reloads(p)
    assert f.count == 49  # one load per entry stays
    # negative: the loop stores the slot
    body2 = [
        ld(0x10, "R1", -4),
        st(0x11, "R1", -4),
        mk(0x12, "R14 = dec(R14)"),
        mk(
            0x13,
            "JUMP IF NE non-delayed  target=0x10",
            form="8a_rel",
            branch="cond_jump",
            target=0x10,
            cond=True,
            unit=None,
            flags_r=True,
        ),
        mk(0x14, "R0 = pass(R2)"),
    ]
    p = W.Program(body2, counts, {(0x13, 0x14): 1}, funcs=[(0x10, "f")])
    assert not W.det_invariant_reloads(p)


def test_dead_code_after_unconditional_jump() -> None:
    jump = mk(
        0,
        "JUMP non-delayed  target=0x40",
        form="8a_rel",
        branch="jump",
        target=0x40,
        unit=None,
    )
    dead = mk(1, "R1 = add(R2, R3)")
    p = W.Program([jump, dead], {0: 5}, funcs=[(0, "f")])
    assert names(W.det_dead_code(p)) == {"dead_code"}
    p = W.Program([jump, dead], {0: 5, 1: 3}, funcs=[(0, "f")])  # it runs: not dead
    assert not W.det_dead_code(p)


def test_run_all_ranks_by_saving_and_groups() -> None:
    p = prog([st(0, "R3", -4), mk(1, "R5 = add(R1, R2)"), ld(2, "R3", -4)])
    fs = W.run_all(p)
    assert fs == sorted(fs, key=lambda f: (-f.saving, f.pc))
    g = W.group_findings(fs)
    assert g and g[0]["saving_cycles"] >= g[-1]["saving_cycles"]
