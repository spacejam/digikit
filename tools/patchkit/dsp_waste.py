"""Peephole and profile-guided waste detectors for SHARC+ code, on a decoded instruction model.

The model (`Op`, `Program`) is built from the one decoder of the repository: the typed decode fields of
tools/sharc_dev (sharcinv, sharcfn, sharcdb.regdefs for register effects, sharcdb.facts for memory accesses) and the
mnemonic text of sharcfn.render_instruction. Nothing here decodes bits. `pk_dsp_waste` builds a Program from the
isolated PLAITS routine (dynamic counts from the concrete core) or from the stock image database (dynamic counts from
an engine profile, else loop-nesting weights).

Every detector returns `Finding`s: where, how often it runs, the cycles one instance saves, the product, and a
confidence. Cycles are the instruction count plus the stall rules of tools/sharc_dev/sharc_timing.py (nominal
column); branch stalls assume a branch target buffer hit and no memory latency, so the cycle figures are a lower
bound. The analyses are intra-block (a block ends at every branch, delay-slot boundary and jump target) and use no
alias analysis beyond "same base register and different constant offset, or different address space": an
intervening store through any other pointer lowers the confidence to `low`. Registers live out of a block are
unknown, so a removal that needs a value dead after the block is only reported when the block itself redefines it.
Per-detector assumptions are in `ASSUMPTIONS` and are copied to the report.
"""

from __future__ import annotations

import bisect
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from tools.sharc_dev import sharc_timing as T
from tools.sharc_dev import sharcfn, sharcinv
from tools.sharc_dev.sharcdb import facts, regdefs

# nominal stalls of the manual's branch rules: (non-delayed, delayed) cycles of a taken branch with a BTB hit
BRANCH_STALL = T.UNCOND_BTB_HIT
STACK_BASES = ("I6", "I7")
PLAIN_FORMS = ("2a", "2a_short", "2c", "6b_shiftimm")
ZERO_M = (
    5,
    13,
)  # the firmware's M5 and M13 are the zero modifiers (plaits_rig sets them)
DATA_REG = re.compile(r"^R\d+$")
SHAPE = re.compile(r"\b[RFS]\d+\b")
FLOAT_REG = re.compile(r"\bF(\d+)\b")

ASSUMPTIONS = {
    "cycles": "instructions + dag-load/float-forward/float-to-move stalls (sharc_timing, nominal) + taken-branch "
    "stalls (BTB hit assumed: 2 non-delayed, 0 delayed); no cache or memory latency: a lower bound",
    "blocks": "analyses stay inside a basic block (split at branches, delay slots and jump targets); values live "
    "out of a block are unknown, so a dead-after claim needs a redefinition inside the block",
    "alias": "no alias analysis: same base register + different constant offset, or different space, is distinct; "
    "any other intervening store lowers the confidence to low",
    "m_zero": "M5 and M13 are zero modifiers (firmware convention, set by the rig)",
    "flags": "ASTAT flags are one pseudo-register FLAGS: written by every ALU/MULT/SHIFT compute, read by "
    "conditional instructions and carry operations",
    "encoding": "a rewrite is only a candidate until its instruction form passes pk_encoding_gate (stock precedent) and "
    "the output is bit-identical",
}


def norm(r: str) -> str:
    """F5 and R5 are one register; 5a/5b names, S registers and the rest are unchanged."""
    m = re.match(r"^[FR](\d+)$", r)
    return "R" + m.group(1) if m else r


@dataclass(frozen=True)
class Mem:
    space: str  # DM or PM
    direction: str  # load or store
    base: str | None  # the I register, None for an absolute address
    key: (
        tuple[Any, ...] | None
    )  # a stable address key, None when the address is not provably repeatable
    reg: (
        str | None
    )  # the data/universal register moved (normalised), None for dual moves
    post: bool  # the base register is modified by this access
    off: int | None = None


@dataclass
class Op:
    pc: int
    nw: int  # width in short words
    text: str
    form: str
    cond: bool = False  # conditionally executed (IF ...)
    defs: frozenset[str] = frozenset()
    uses: frozenset[str] = frozenset()
    def_kinds: tuple[tuple[str, str], ...] = ()
    mem: tuple[Mem, ...] = ()
    unit: str | None = (
        None  # compute unit: ALU MULT SHIFT MULTIFN SHORT, None = no compute
    )
    flags_w: bool = False
    flags_r: bool = False
    branch: str = ""  # jump cond_jump call return indirect loop
    delayed: bool = False
    target: int | None = None
    loop_end: int = 0
    trip: int | None = None
    nop: bool = False
    shape: str = ""  # text with register numbers removed
    fields: dict[str, int] = field(default_factory=dict, repr=False)

    @property
    def next_pc(self) -> int:
        return self.pc + self.nw

    @property
    def is_branch(self) -> bool:
        return bool(self.branch)

    @property
    def plain_compute(self) -> bool:
        """A single-function compute with no parallel move, no condition and no branch (forms 2a, 2c, shift
        immediate): the only ones a transfer can be folded into or a SIMD pair built from."""
        return (
            self.form in PLAIN_FORMS
            and self.unit in ("ALU", "MULT", "SHIFT", "SHORT")
            and not self.mem
            and not self.branch
            and not self.cond
        )

    def klass(self) -> str:
        if self.nop:
            return "nop"
        if self.branch in ("call",):
            return "call"
        if self.branch == "return":
            return "return"
        if self.branch == "loop":
            return "loop_setup"
        if self.branch:
            return "branch"
        if self.mem and self.unit:
            return "multifunction"
        if self.mem:
            return (
                "stack_transfer"
                if any(m.base in STACK_BASES for m in self.mem)
                else "transfer"
            )
        if self.unit:
            return "compute"
        return "move_other"


def _mem_rows(form: str, f: dict[str, int], text: str) -> list[Mem]:
    reg: str | None = None
    if "dreg" in f:
        reg = "R%d" % f["dreg"]
    elif "ureg" in f:
        reg = norm(sharcfn.ureg_name(f["ureg"]))
    out: list[Mem] = []
    if form == "3c":
        m = re.search(r"\b(DM|PM)\(", text)
        i, mm = f.get("dmi", 0), f.get("dmm", 0)
        zero = mm in ZERO_M
        space = m.group(1) if m else "DM"
        out.append(
            Mem(
                space,
                "store" if f.get("d") else "load",
                "I%d" % i,
                (space, "I%d" % i, 0) if zero else None,
                "R%d" % f["dreg"] if "dreg" in f else None,
                not zero,
                0 if zero else None,
            )
        )
        return out
    for space, direction, base, mod, u, mform, _w, absaddr in facts.extract_mem_access(
        form, f
    ):
        if mform == "dual":
            mm = int(mod[1:]) if mod else -1
            zero = mm in ZERO_M
            out.append(
                Mem(
                    space,
                    direction,
                    base,
                    (space, base, 0) if zero else None,
                    None,
                    not zero,
                    0 if zero else None,
                )
            )
        elif absaddr is not None:
            out.append(Mem(space, direction, None, (space, None, absaddr), reg, False))
        elif mform == "i+imm" or form in ("15a", "15b", "4a", "4b", "4d"):
            off = int(mod)
            out.append(Mem(space, direction, base, (space, base, off), reg, False, off))
        else:  # i,m-mod forms: u=1 post-modifies, u=0 pre-modifies without update
            mnum = int(mod[1:]) if mod else -1
            zero = mnum in ZERO_M
            if zero:
                out.append(
                    Mem(
                        space,
                        direction,
                        base,
                        (space, base, 0),
                        reg,
                        False,
                        0,
                    )
                )
            elif u:  # address I, then I += M
                out.append(Mem(space, direction, base, (space, base, 0), reg, True, 0))
            else:  # address I + M, no update
                out.append(
                    Mem(space, direction, base, (space, base, ("M", mnum)), reg, False)
                )
    return out


_ALU_NO_RESULT = ("comp(", "compu(", "fcomp(")


_REGTOK = re.compile(r"\b[RFS]\d+\b")


def _text_effects(t: str) -> tuple[set[str], set[str]]:
    """Registers written and read by the compute clauses of a rendered instruction ("F1 = F2 * F4, F15 = F8 + F15")."""
    comp = re.sub(r"^IF (?:NOT )?[A-Z]+ ", "", t.split(";")[0])
    defs: set[str] = set()
    uses: set[str] = set()
    for clause in re.split(r", (?=[RFS]\d+ =)", comp):
        lhs, eq, rhs = clause.partition(" = ")
        if not eq:
            continue
        defs |= {norm(r) for r in _REGTOK.findall(lhs)}
        uses |= {norm(r) for r in _REGTOK.findall(rhs)}
        if re.search(r"\bMR[0-9A-Z]*\b", lhs):
            defs.add("MR")
        if re.search(r"\bMR[0-9A-Z]*\b", rhs):
            uses.add("MR")
    return defs, uses


def build_op(
    pc: int, nbytes: int, form: str, raw_fields: dict[str, int], text: str
) -> Op:
    """One Op from the typed decode fields and the rendered text (the same two inputs for both sources)."""
    f = sharcinv.merge_fields(raw_fields)
    defs_k, uses, unk = regdefs.register_effects(form, f)
    defs_k, uses = list(defs_k), list(uses)
    unit: str | None = None
    if form in sharcinv.COMPUTE_FORMS and form != "11a":
        c = f.get("compute", 0)
        if c:
            unit = sharcinv.classify_compute(c)[0]
    elif form == "2c":
        unit = "SHORT" if f.get("compute") else None
    elif form == "6b_shiftimm":
        unit = "SHIFT"
    t = text.strip()
    cond = t.startswith("IF ")
    branch, delayed, target, loop_end, trip = "", False, None, 0, None
    mjump = re.search(r"(?:^|; )(JUMP|CALL|RETURN)\b", t)
    if t.startswith("DO "):
        branch = "loop"
        m = re.match(r"DO 0x([0-9a-f]+) UNTIL", t)
        loop_end = int(m.group(1), 16) if m else 0
        mt = re.search(r"trip count (\d+), literal", t)
        trip = int(mt.group(1)) if mt else None
    elif mjump:
        kind = mjump.group(1)
        head = t[mjump.start() :].split("target=")[0]
        conditional = " IF " in head
        mt = re.search(r"target=0x([0-9a-f]+)", t)
        target = int(mt.group(1), 16) if mt else None
        delayed = ("delayed" in head and "non-delayed" not in head) or (
            kind == "RETURN" and bool(f.get("j"))
        )
        if kind == "CALL":
            branch = "call" if target is not None else "indirect"
        elif kind == "RETURN":
            branch = "return"
        else:
            branch = "cond_jump" if conditional else "jump"
            if target is None:
                branch = "indirect"
    if (
        form == "11a" and not branch
    ):  # compute with a return (RTS/RTI), delayed by the J bit
        branch, delayed = "return", bool(f.get("j"))
    nop = bool(re.match(r"^NOP", t)) or t == "IDLE"
    defs = {norm(r) for r, _ in defs_k}
    use = {norm(r) for r in uses}
    if unit == "MULTIFN" or unk:
        # regdefs leaves the multiply + ALU multifunction forms (and a few others) unmodelled: take the registers
        # of the compute clauses from the rendered text instead (union: more uses is the safe direction)
        td, tu = _text_effects(t)
        defs |= td
        use |= tu
        defs_k += [(r, "compute") for r in td]
    kinds = tuple((norm(r), k) for r, k in defs_k)
    flags_w = unit in ("ALU", "MULT", "SHIFT", "SHORT", "MULTIFN")
    flags_r = cond or branch == "cond_jump" or bool(re.search(r"_c\(|\bCI\b", t))
    if any(s in t for s in _ALU_NO_RESULT) and unit in ("ALU", "SHORT"):
        defs = set()  # COMP sets flags only: the printed Rn is not written
        kinds = ()
    if unit == "MULT" and not re.match(r"^[RF]\d+ = [RF]\d+ \* [RF]\d+", t):
        defs.add("MR")
        use.add("MR")
    if branch in ("jump", "cond_jump", "indirect", "call") and "indirect" in t:
        use |= {r for r in re.findall(r"\b([IM]\d+)\b", t)}
    if cond:
        use |= defs  # a conditional write may leave the old value
    mem = tuple(_mem_rows(form, f, t))
    return Op(
        pc=pc,
        nw=nbytes // 2,
        text=t,
        form=form,
        cond=cond or (branch == "cond_jump"),
        defs=frozenset(defs),
        uses=frozenset(use),
        def_kinds=kinds,
        mem=mem,
        unit=unit,
        flags_w=flags_w,
        flags_r=flags_r,
        branch=branch,
        delayed=delayed,
        target=target,
        loop_end=loop_end,
        trip=trip,
        nop=nop,
        shape=SHAPE.sub("r", t),
        fields=f,
    )


@dataclass
class Loop:
    kind: str  # hw (DO ... UNTIL) or sw (backward branch)
    head: int
    end: int  # last pc of the body (inclusive)
    func: str
    trip: int | None
    iters: float = 0.0  # body executions in the profile
    entries: float = 0.0
    depth: int = 1
    static_ops: int = 0
    executed: float = 0.0  # instructions executed in the body (nested loops included)
    branch_pc: int = 0


@dataclass
class Finding:
    detector: str
    pc: int
    function: str
    where: str  # function+offset (the object has no line table)
    count: float  # executions of the anchor instruction
    saving_each: float  # cycles saved per execution (estimate)
    confidence: str  # high medium low
    note: str
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def saving(self) -> float:
        return self.count * self.saving_each

    def to_json(self) -> dict[str, Any]:
        d = {
            "detector": self.detector,
            "pc": "%#x" % self.pc,
            "function": self.function,
            "where": self.where,
            "count": self.count,
            "saving_each": self.saving_each,
            "saving_cycles": self.saving,
            "confidence": self.confidence,
            "note": self.note,
        }
        if self.extra:
            d["extra"] = self.extra
        return d


class Program:
    """Decoded instructions plus an execution profile (counts per pc and adjacent executed pairs)."""

    def __init__(
        self,
        ops: Iterable[Op],
        counts: Mapping[int, float],
        pairs: dict[tuple[int, int], int] | None = None,
        funcs: Sequence[tuple[int, str]] = (),
        labels: Sequence[tuple[int, str]] = (),
        extra_leaders: Iterable[int] = (),
        weights: str = "dynamic",
    ) -> None:
        self.ops: dict[int, Op] = {o.pc: o for o in ops}
        self.pcs = sorted(self.ops)
        self.counts: dict[int, float] = dict(counts)
        self.pairs = pairs or {}
        self.funcs = sorted(funcs)
        self._fstarts = [a for a, _ in self.funcs]
        self.labels = sorted(labels) or self.funcs
        self._lstarts = [a for a, _ in self.labels]
        self.weights = weights
        self.extra_leaders = set(extra_leaders)
        self._blocks: list[list[Op]] | None = None
        self._loops: list[Loop] | None = None
        self.stall: dict[int, float] = {}
        self.pair_stalls()

    # -- lookup
    def count(self, pc: int) -> float:
        return self.counts.get(pc, 0.0)

    def func_of(self, pc: int) -> tuple[int, str]:
        i = bisect.bisect_right(self._fstarts, pc) - 1
        return self.funcs[i] if i >= 0 else (pc, "?")

    def where(self, pc: int) -> str:
        i = bisect.bisect_right(self._lstarts, pc) - 1
        if i < 0:
            return "%#x" % pc
        a, n = self.labels[i]
        return "%s+%d" % (n, pc - a) if pc != a else n

    def next_op(self, op: Op) -> Op | None:
        return self.ops.get(op.next_pc)

    def slots(self, op: Op) -> list[Op]:
        """The delay-slot instructions of a delayed branch (up to two)."""
        out: list[Op] = []
        pc = op.next_pc
        for _ in range(2):
            o = self.ops.get(pc)
            if o is None:
                break
            out.append(o)
            pc = o.next_pc
        return out

    # -- cycles
    def taken(self, op: Op) -> float:
        n = self.count(op.pc)
        if op.branch not in ("cond_jump",) or not self.pairs:
            return n
        if op.delayed:
            sl = self.slots(op)
            last = sl[-1] if sl else op
            return n - self.pairs.get((last.pc, last.next_pc), 0)
        return n - self.pairs.get((op.pc, op.next_pc), 0)

    def pair_stalls(self) -> None:
        """Per-pc stall cycles: adjacent-pair stalls (from the executed pairs) and taken-branch stalls."""
        tins = {pc: to_timing(o) for pc, o in self.ops.items()}
        for (a, b), n in self.pairs.items():
            ia, ib = tins.get(a), tins.get(b)
            if ia is None or ib is None:
                continue
            c = (
                T.dag_use(ia, ib)[1]
                + T.float_forward(ia, ib)[1]
                + T.float_to_move(ia, ib)[1]
            )
            if c:
                self.stall[b] = self.stall.get(b, 0.0) + c * n
        for pc, o in self.ops.items():
            if o.branch in ("jump", "cond_jump", "call", "return", "indirect"):
                s = BRANCH_STALL[1 if o.delayed else 0]
                if s:
                    self.stall[pc] = self.stall.get(pc, 0.0) + s * self.taken(o)

    def cycles(self, pcs: Iterable[int] | None = None) -> float:
        sel = self.pcs if pcs is None else pcs
        return sum(self.count(p) + self.stall.get(p, 0.0) for p in sel)

    def instructions(self, pcs: Iterable[int] | None = None) -> float:
        sel = self.pcs if pcs is None else pcs
        return sum(self.count(p) for p in sel)

    # -- structure
    def leaders(self, after_branch: bool = True) -> set[int]:
        ld = set(self._fstarts) | set(self.extra_leaders)
        for o in self.ops.values():
            if o.target is not None:
                ld.add(o.target)
            if o.branch == "loop":
                ld.add(o.next_pc)
                ld.add(o.loop_end + 1)
            if after_branch and o.branch in (
                "jump",
                "cond_jump",
                "call",
                "return",
                "indirect",
            ):
                after = o.next_pc
                if o.delayed:
                    for s in self.slots(o):
                        after = s.next_pc
                ld.add(after)
                if o.delayed:
                    ld.add(o.next_pc)
        return ld

    def blocks(self) -> list[list[Op]]:
        """Basic blocks over contiguous decoded code; a branch ends its block, its delay slots form the next."""
        if self._blocks is not None:
            return self._blocks
        ld = self.leaders()
        ends = self._loop_ends()
        out: list[list[Op]] = []
        cur: list[Op] = []
        for pc in self.pcs:
            o = self.ops[pc]
            if cur:
                prev = cur[-1]
                if prev.next_pc != pc or pc in ld or prev.branch or prev.pc in ends:
                    out.append(cur)
                    cur = []
            cur.append(o)
        if cur:
            out.append(cur)
        self._blocks = out
        return out

    def _loop_ends(self) -> set[int]:
        return {o.loop_end for o in self.ops.values() if o.branch == "loop"}

    def loops(self) -> list[Loop]:
        if self._loops is not None:
            return self._loops
        loops: list[Loop] = []
        for o in self.ops.values():
            if o.branch == "loop":
                head = o.next_pc
                n = self.count(o.pc)
                lp = Loop(
                    "hw",
                    head,
                    o.loop_end,
                    self.func_of(o.pc)[1],
                    o.trip,
                    entries=n,
                    iters=self.count(o.loop_end),
                    branch_pc=o.pc,
                )
                loops.append(lp)
            elif (
                o.branch in ("jump", "cond_jump")
                and o.target is not None
                and o.target <= o.pc
                and self.func_of(o.target)[0] == self.func_of(o.pc)[0]
            ):
                end = o.pc
                if o.delayed:
                    end = (self.slots(o) or [o])[-1].pc
                hdr = self.count(o.target)
                back = self.taken(o)
                loops.append(
                    Loop(
                        "sw",
                        o.target,
                        end,
                        self.func_of(o.pc)[1],
                        None,
                        iters=hdr,
                        entries=max(hdr - back, 0.0),
                        branch_pc=o.pc,
                    )
                )
        for lp in loops:
            body = [p for p in self.pcs if lp.head <= p <= lp.end]
            lp.static_ops = len(body)
            lp.executed = self.instructions(body)
            lp.depth = (
                sum(
                    1
                    for other in loops
                    if other.head <= lp.head
                    and lp.end <= other.end
                    and (other.head, other.end) != (lp.head, lp.end)
                )
                + 1
            )
        self._loops = sorted(loops, key=lambda x: -x.executed)
        return self._loops

    def innermost(self, pc: int) -> Loop | None:
        best = None
        for lp in self.loops():
            if lp.head <= pc <= lp.end and (
                best is None or lp.end - lp.head < best.end - best.head
            ):
                best = lp
        return best

    def finding(
        self,
        det: str,
        op: Op,
        saving_each: float,
        confidence: str,
        note: str,
        count: float | None = None,
        **extra: Any,
    ) -> Finding:
        return Finding(
            det,
            op.pc,
            self.func_of(op.pc)[1],
            self.where(op.pc),
            self.count(op.pc) if count is None else count,
            saving_each,
            confidence,
            note,
            extra,
        )


def to_timing(o: Op) -> T.Ins:
    return T.Ins(
        pc=o.pc,
        nbytes=o.nw * 2,
        text=o.text,
        always=not o.cond,
        defs=tuple(o.def_kinds),
        uses=frozenset(o.uses),
        mem=tuple((m.space, m.base or "") for m in o.mem),
        branch={
            "cond_jump": "cond_jump",
            "indirect": "indirect",
            "loop": "loop",
        }.get(o.branch, o.branch),
        delayed=o.delayed,
        targets=(o.target,) if o.target is not None else (),
        loop_end=o.loop_end,
        loop_counter=o.branch == "loop",
    )


# -- dependence helpers (conservative)


def distinct(a: Mem, b: Mem) -> bool:
    """Provably different locations."""
    if a.space != b.space:
        return True
    if a.key is None or b.key is None:
        return False
    if a.key == b.key:
        return False
    if a.base == b.base and a.base is not None:
        return isinstance(a.off, int) and isinstance(b.off, int) and a.off != b.off
    return a.base is None and b.base is None


def mem_conflict(a: Op, b: Op) -> bool:
    for x in a.mem:
        for y in b.mem:
            if (x.direction == "store" or y.direction == "store") and not distinct(
                x, y
            ):
                return True
    return False


def can_swap(a: Op, b: Op) -> bool:
    """Adjacent a then b can be exchanged (no register, flag or memory dependence either way)."""
    if a.defs & b.uses or a.defs & b.defs or a.uses & b.defs:
        return False
    if (a.flags_w and (b.flags_r or b.flags_w)) or (a.flags_r and b.flags_w):
        return False
    return not mem_conflict(a, b)


def reads_flags_before_write(ops: Sequence[Op], start: int) -> bool:
    """A flag reader follows before the next unconditional flag writer (flags are not carried over a block end)."""
    for o in ops[start:]:
        if o.flags_r:
            return True
        if o.flags_w and not o.cond:
            return False
    return False


def forward_ok(blk: Sequence[Op], i: int, dst: str, src: str) -> tuple[bool, bool]:
    """Can every use of `dst` after blk[i] read `src` instead (copy propagation)? Needs `dst` redefined in the
    block and `src` unchanged until then. Returns (ok, a use is in a register-group restricted form)."""
    risky = False
    for o in blk[i + 1 :]:
        if dst in o.uses and (o.unit == "MULTIFN" or o.form in ("1a", "1b")):
            risky = True
        if dst in o.defs and not o.cond:
            return True, risky
        if src in o.defs or o.branch:
            return False, risky
    return False, risky


# -- detectors


def det_memory(p: Program) -> list[Finding]:
    """Spill reloads, store-to-load forwarding and redundant loads, per block."""
    out: list[Finding] = []
    for blk in p.blocks():
        # key -> [register holding the value, "load"|"store", may-alias stores seen, pc, Mem]
        avail: dict[tuple[Any, ...], list[Any]] = {}
        for idx, o in enumerate(blk):
            if o.cond or o.branch:
                for k in list(avail):
                    if o.defs & {avail[k][0], avail[k][4].base or ""}:
                        del avail[k]
                continue
            for m in o.mem:
                if m.direction == "load" and m.key in avail and m.reg is not None:
                    reg, kind, risk, spc, _em = avail[m.key]
                    same = reg == m.reg
                    fwd = same or forward_ok(blk, idx, m.reg, reg)[0]
                    if kind == "store":
                        det = (
                            "spill_reload"
                            if m.base in STACK_BASES
                            else "store_load_forward"
                        )
                    else:
                        det = "redundant_load"
                    if same:
                        how = "the value is still in %s: delete the load" % reg
                    elif fwd:
                        how = "the value is in %s: delete the load, use %s" % (reg, reg)
                    else:
                        how = (
                            "the value is in %s: the load becomes a register copy" % reg
                        )
                    out.append(
                        p.finding(
                            det,
                            o,
                            1.0 if fwd else 0.0,
                            "high" if risk == 0 else "low",
                            "%s (%s at %#x)" % (how, kind, spc),
                            key=repr(m.key),
                            may_alias_stores=risk,
                        )
                    )
            for k in list(avail):
                if o.defs & {avail[k][0], avail[k][4].base or ""}:
                    del avail[k]
            for m in o.mem:
                if (
                    m.direction == "store"
                    and m.key in avail
                    and avail[m.key][0] == m.reg
                ):
                    reg, kind, risk, spc, _em = avail[m.key]
                    out.append(
                        p.finding(
                            "redundant_store",
                            o,
                            1.0,
                            "high" if risk == 0 else "low",
                            "stores %s back to a location that already holds it (%s at %#x)"
                            % (reg, kind, spc),
                            key=repr(m.key),
                            may_alias_stores=risk,
                        )
                    )
            for m in o.mem:
                if m.direction == "store":
                    for k in list(avail):
                        if k == m.key:
                            del avail[k]
                        elif avail[k][4].space == m.space and not distinct(
                            m, avail[k][4]
                        ):
                            avail[k][2] += 1
            for m in o.mem:
                if m.post and m.base:
                    for k in list(avail):
                        if avail[k][4].base == m.base:
                            del avail[k]
            for m in o.mem:
                if (
                    m.key is not None
                    and m.reg is not None
                    and DATA_REG.match(m.reg)
                    and not (
                        m.direction == "load" and m.reg in o.defs and m.reg == m.base
                    )
                    and not m.post
                ):
                    avail[m.key] = [m.reg, m.direction, 0, o.pc, m]
    return out


def _is_copy(o: Op) -> tuple[str, str] | None:
    if o.cond or o.mem or o.branch:
        return None
    m = re.match(r"^[RF](\d+) = (?:f?pass\()?[RF](\d+)\)?$", o.text)
    return ("R" + m.group(1), "R" + m.group(2)) if m else None


def det_copy_prop(p: Program) -> list[Finding]:
    out: list[Finding] = []
    for blk in p.blocks():
        for i, o in enumerate(blk):
            c = _is_copy(o)
            if c is None:
                continue
            d, s = c
            if d == s:
                if not (o.flags_w and reads_flags_before_write(blk, i + 1)):
                    out.append(
                        p.finding(
                            "copy_removal", o, 1.0, "high", "self copy %s: delete" % d
                        )
                    )
                continue
            ok, risky = forward_ok(blk, i, d, s)
            if not ok or (o.flags_w and reads_flags_before_write(blk, i + 1)):
                continue
            out.append(
                p.finding(
                    "copy_removal",
                    o,
                    1.0,
                    "low" if risky else "medium",
                    "%s = %s: read %s instead of %s up to the next definition of %s%s"
                    % (
                        d,
                        s,
                        s,
                        d,
                        d,
                        " (multifunction operand groups to check)" if risky else "",
                    ),
                )
            )
    return out


def det_dead_def(p: Program) -> list[Finding]:
    out: list[Finding] = []
    for blk in p.blocks():
        for i, o in enumerate(blk):
            pure = (o.unit in ("ALU", "MULT", "SHIFT", "SHORT") and not o.mem) or any(
                k == "literal" for _, k in o.def_kinds
            )
            if not pure or o.cond or o.branch or not o.defs or "MR" in o.defs:
                continue
            if not all(DATA_REG.match(d) for d in o.defs):
                continue
            dead = True
            for d in o.defs:
                state = "unknown"
                for x in blk[i + 1 :]:
                    if d in x.uses:
                        state = "live"
                        break
                    if d in x.defs and not x.cond:
                        state = "dead"
                        break
                if state != "dead":
                    dead = False
                    break
            if dead and not (o.flags_w and reads_flags_before_write(blk, i + 1)):
                out.append(
                    p.finding(
                        "dead_definition",
                        o,
                        1.0,
                        "medium",
                        "%s is overwritten before any use in the block"
                        % ",".join(sorted(o.defs)),
                    )
                )
    return out


_COUNTER_RE = re.compile(r"^R(\d+) = (dec|inc)\(R(\d+)\)")
_LOOPCOND = ("EQ", "NE", "GT", "LT", "GE", "LE")


def det_sw_loops(p: Program) -> list[Finding]:
    out: list[Finding] = []
    block_of = {x.pc: b for b in p.blocks() for x in b}
    for lp in p.loops():
        if lp.kind != "sw":
            continue
        br = p.ops[lp.branch_pc]
        mcond = re.match(r"^JUMP IF ([A-Z]+) ", br.text)
        if br.branch != "cond_jump" or not mcond or mcond.group(1) not in _LOOPCOND:
            continue
        blk = block_of.get(br.pc)
        if blk is None:
            continue
        j = [x.pc for x in blk].index(br.pc)
        writer = next((x for x in reversed(blk[:j]) if x.flags_w and not x.cond), None)
        if writer is None:
            continue
        body = [p.ops[q] for q in p.pcs if lp.head <= q <= lp.end]
        nops = sum(1 for s in (p.slots(br) if br.delayed else []) if s.nop)
        stall = BRANCH_STALL[1 if br.delayed else 0]
        m = _COUNTER_RE.match(writer.text)
        if m:
            reg = "R" + m.group(3)
            used = sum(1 for x in body if reg in x.uses and x is not writer)
            per_iter = (1.0 if used == 0 else 0.0) + 1.0 + stall + nops
            conf = "medium" if used == 0 else "low"
            why = (
                "counter %s only feeds the branch: DO loop with LCNTR" % reg
                if used == 0
                else "counter %s is also an index: only the branch folds into LCNTR"
                % reg
            )
        elif re.search(r"\bf?comp\(", writer.text):
            per_iter, conf = 2.0 + stall + nops, "low"
            why = (
                "compare-and-branch against a bound (the trip count must be derivable)"
            )
        else:
            continue
        back = p.taken(br)
        if back <= 0:
            continue
        each = max(
            per_iter - lp.entries / back, 0.0
        )  # one cycle of set-up per loop entry
        out.append(
            p.finding(
                "sw_loop_to_hw",
                br,
                each,
                conf,
                "%s; %d back edges, %.0f entries, body %d instrs"
                % (why, back, lp.entries, lp.static_ops),
                count=back,
                loop="%#x-%#x" % (lp.head, lp.end),
            )
        )
    return out


def _func_extent(p: Program, entry: int) -> list[Op]:
    nxt = [a for a in p._fstarts if a > entry]
    end = nxt[0] if nxt else (p.pcs[-1] + 1)
    return [p.ops[q] for q in p.pcs if entry <= q < end]


_FRAME_FORMS = ("16a", "25c_rframe")


def frame_overhead(p: Program, entry: int) -> tuple[int, int, bool, int, bool]:
    """(call-protocol instructions of a callee, static size, leaf, delay-slot NOPs after its return, return is
    delayed) from the first and last instructions of the callee."""
    body = _func_extent(p, entry)
    if not body:
        return 0, 0, False, 0, False
    leaf = not any(o.branch in ("call", "indirect") for o in body)

    def is_frame(o: Op) -> bool:
        return (
            o.form in _FRAME_FORMS
            or bool(o.defs & {"I6", "I7", "I12", "I13"})
            or any(m.base == "I7" and m.post for m in o.mem)
            or o.branch == "return"
        )

    head = [o for o in body[:8] if is_frame(o)]
    ret_i = next((i for i, o in enumerate(body) if o.branch == "return"), len(body) - 1)
    tail = [
        o for o in body[max(0, ret_i - 6) : ret_i + 3] if is_frame(o) and o not in head
    ]
    nops = sum(1 for o in body[ret_i + 1 : ret_i + 3] if o.nop)
    return len(head) + len(tail), len(body), leaf, nops, body[ret_i].delayed


def det_calls(p: Program) -> list[Finding]:
    out: list[Finding] = []
    for o in p.ops.values():
        if o.branch != "call" or o.target is None or p.count(o.pc) <= 0:
            continue
        ovh, size, leaf, nops, rdelayed = frame_overhead(p, o.target)
        if size == 0:
            continue
        slot_nops = sum(1 for s in p.slots(o) if s.nop)
        stall = BRANCH_STALL[1 if o.delayed else 0] + BRANCH_STALL[1 if rdelayed else 0]
        each = (
            1.0 + ovh + slot_nops + nops + stall
        )  # CALL, frame protocol, empty slots, call+return stalls
        lp = p.innermost(o.pc)
        small = size <= 40
        out.append(
            p.finding(
                "call_inline",
                o,
                each,
                "medium" if (leaf and small) else "low",
                "%s%s call to %s (%d instrs, %d frame/protocol instrs)%s"
                % (
                    "leaf " if leaf else "",
                    "small" if small else "large",
                    p.func_of(o.target)[1],
                    size,
                    ovh,
                    " inside loop %#x-%#x" % (lp.head, lp.end) if lp else "",
                ),
                callee=p.func_of(o.target)[1],
                callee_size=size,
                leaf=leaf,
                in_loop=lp is not None,
            )
        )
    return out


def _foldable_transfer(o: Op) -> bool:
    """A standalone DM/PM transfer whose form has a compute-capable sibling (3a/4a, data register)."""
    if o.cond or o.branch or o.unit or len(o.mem) != 1:
        return False
    m = o.mem[0]
    if m.reg is None or not DATA_REG.match(m.reg):
        return False
    if o.form in ("3a", "4a", "3c"):
        return True
    if o.form in ("15a", "15b") and isinstance(m.off, int):
        return (
            -32 <= m.off <= 31
        )  # fits the 6-bit field of the compute + move immediate form (4a)
    return False


def _far_transfer(o: Op) -> bool:
    """The same, but the immediate offset is beyond the 6-bit field: needs a second base register near the slot."""
    if o.cond or o.branch or o.unit or len(o.mem) != 1:
        return False
    m = o.mem[0]
    return (
        o.form in ("15a", "15b")
        and m.reg is not None
        and bool(DATA_REG.match(m.reg))
        and isinstance(m.off, int)
        and not -32 <= m.off <= 31
    )


def det_fold_transfer(p: Program) -> list[Finding]:
    out: list[Finding] = []
    for blk in p.blocks():
        used: set[int] = set()
        for i in range(len(blk) - 1):
            a, b = blk[i], blk[i + 1]
            if a.pc in used or b.pc in used:
                continue
            for c, t in ((a, b), (b, a)):
                near = _foldable_transfer(t)
                if not (c.plain_compute and (near or _far_transfer(t))):
                    continue
                if c.defs & t.defs:
                    continue
                # parallel reads see the old values: the move must not read what the compute writes when the
                # compute came first, and the compute must not read what the move writes when the move came first
                if c is a and c.defs & t.uses:
                    continue
                if t is a and t.defs & c.uses:
                    continue
                used |= {a.pc, b.pc}
                out.append(
                    p.finding(
                        "fold_transfer" if near else "fold_transfer_far",
                        t,
                        1.0,
                        "medium" if near else "low",
                        "%s next to independent compute %#x (%s): one compute + move instruction%s"
                        % (
                            t.text,
                            c.pc,
                            c.text,
                            ""
                            if near
                            else "; the offset is beyond the 6-bit field: needs a frame pointer closer to the slot",
                        ),
                        count=min(p.count(a.pc), p.count(b.pc)),
                        compute_pc="%#x" % c.pc,
                    )
                )
                break
    return out


_MUL_RE = re.compile(r"^[RF]\d+ = [RF](\d+) \* [RF](\d+)")
_ADD_RE = re.compile(
    r"^[RF]\d+ = (?:[RF](\d+) [+-] [RF](\d+)|[a-z]*(?:add|sub|avg)\([RF](\d+), [RF](\d+)\))"
)


def _mult_alu_fits(mul: Op, alu: Op) -> bool:
    m, a = _MUL_RE.match(mul.text), _ADD_RE.match(alu.text)
    if not (m and a):
        return False
    x, y = int(m.group(1)), int(m.group(2))
    ax, ay = (int(g) for g in a.groups() if g is not None)
    return x < 4 <= y < 8 and 8 <= ax < 12 and 12 <= ay < 16


def det_compute_pairs(p: Program) -> list[Finding]:
    """Independent same-operation computes (SIMD PEx/PEy) and a multiply next to an add/sub (multifunction)."""
    out: list[Finding] = []
    for blk in p.blocks():
        used: set[int] = set()
        for i, a in enumerate(blk):
            if a.pc in used or not a.plain_compute:
                continue
            for j in range(i + 1, min(i + 5, len(blk))):
                b = blk[j]
                if b.pc in used or not b.plain_compute:
                    continue
                if a.defs & (b.uses | b.defs) or not all(
                    can_swap(blk[k], b) for k in range(i + 1, j)
                ):
                    continue
                if (
                    a.shape == b.shape
                    and a.unit == b.unit
                    and a.unit in ("ALU", "MULT", "SHIFT")
                ):
                    det, conf = "simd_pair", "low"
                    note = (
                        "%s and %s (%#x): the same operation on independent data; needs the second operand set in "
                        "PEy registers, SIMD mode set and restored, and 64-bit loads for the data"
                        % (a.text, b.text, b.pc)
                    )
                elif {a.unit, b.unit} == {"MULT", "ALU"}:
                    mul, alu = (a, b) if a.unit == "MULT" else (b, a)
                    if not (_MUL_RE.match(mul.text) and _ADD_RE.match(alu.text)):
                        continue
                    fits = _mult_alu_fits(mul, alu)
                    det, conf = "multifunction_pair", "medium" if fits else "low"
                    note = (
                        "%s with %s (%#x): multiply + add/sub in one instruction (operands in R0-3 x R4-7 and R8-11 x R12-15)%s"
                        % (
                            mul.text,
                            alu.text,
                            b.pc,
                            "" if fits else "; needs register renaming",
                        )
                    )
                else:
                    continue
                used |= {a.pc, b.pc}
                out.append(
                    p.finding(
                        det,
                        a,
                        1.0,
                        conf,
                        note,
                        count=min(p.count(a.pc), p.count(b.pc)),
                        partner="%#x" % b.pc,
                    )
                )
                break
    return out


_PURE_RHS = re.compile(r"^[RF]\d+ = (.*)$")


def det_redundant_compute(p: Program) -> list[Finding]:
    """The same pure expression on unchanged operands computed twice in a block (local value numbering)."""
    out: list[Finding] = []
    for blk in p.blocks():
        # rhs -> [dest, pc, operand registers, dest still holds the value]
        avail: dict[str, list[Any]] = {}
        for o in blk:
            m = _PURE_RHS.match(o.text)
            pure = (
                m is not None
                and o.plain_compute
                and len(o.defs) == 1
                and "MR" not in o.defs
                and not o.flags_r
                and not re.search(r"\bf?comp", o.text)
            )
            if pure:
                assert m is not None
                rhs = m.group(1)
                (dest,) = o.defs
                ent = avail.get(rhs)
                if ent is not None:
                    if ent[3]:
                        i = blk.index(o)
                        ok = ent[0] == dest or forward_ok(blk, i, dest, ent[0])[0]
                        out.append(
                            p.finding(
                                "redundant_compute",
                                o,
                                1.0 if ok else 0.0,
                                "high",
                                "%s was computed at %#x into %s and is unchanged"
                                % (rhs, ent[1], ent[0]),
                            )
                        )
                    else:
                        out.append(
                            p.finding(
                                "recompute_after_clobber",
                                o,
                                0.0,
                                "low",
                                "%s was computed at %#x, its register was reused since: a spare register would save "
                                "this instruction (register pressure)" % (rhs, ent[1]),
                                potential_each=1.0,
                            )
                        )
            for k in list(avail):
                if o.defs & avail[k][2]:
                    del avail[k]
                elif avail[k][0] in o.defs:
                    avail[k][3] = False
            if pure:
                assert m is not None
                avail[m.group(1)] = [
                    dest,
                    o.pc,
                    frozenset(SHAPE_REGS.findall(m.group(1))),
                    True,
                ]
                avail[m.group(1)][2] = frozenset(norm(r) for r in avail[m.group(1)][2])
    return out


SHAPE_REGS = re.compile(r"\b[RF]\d+\b")
_I_MOVE = re.compile(r"^I\d+ = R\d+$")
_SHIFT_RE = re.compile(r"^R(\d+) = lshift\(R(\d+), (?:R\d+|-?\d+)\)$")


def det_address_arith(p: Program) -> list[Finding]:
    """`idx << k`, `base + that`, a move to an I register, then the access: the address is rebuilt per element."""
    out: list[Finding] = []
    for blk in p.blocks():
        for i, a in enumerate(blk):
            ms = _SHIFT_RE.match(a.text)
            if not ms or a.cond:
                continue
            t = "R" + ms.group(1)
            window = blk[i + 1 : i + 6]
            add = next(
                (
                    x
                    for x in window
                    if t in x.uses and re.match(r"^R\d+ = add\(", x.text)
                ),
                None,
            )
            if add is None or not add.defs:
                continue
            (u,) = tuple(add.defs)[:1]
            mv = next(
                (
                    x
                    for x in window
                    if _I_MOVE.match(x.text) and x.text.endswith("= " + u)
                ),
                None,
            )
            if mv is None:
                continue
            j = window.index(mv)
            acc = next(
                (
                    x
                    for x in window[j + 1 :]
                    if x.mem and mv.text.split(" = ")[0] in x.uses
                ),
                None,
            )
            if acc is None:
                continue
            out.append(
                p.finding(
                    "address_arith",
                    a,
                    2.0,
                    "low",
                    "address built per access (shift %s, add, move to %s, then %s): an induction pointer in an I "
                    "register with a post-modify removes the shift and the add"
                    % (a.text, mv.text.split(" = ")[0], acc.text),
                )
            )
    return out


def det_invariant_reloads(p: Program) -> list[Finding]:
    """Stack slots loaded in a loop and never stored in it: pin the value in a register (needs a free one)."""
    out: list[Finding] = []
    for lp in p.loops():
        body = [p.ops[q] for q in p.pcs if lp.head <= q <= lp.end]
        inner = [o for o in body if p.innermost(o.pc) is lp]
        stores = {m.key for o in body for m in o.mem if m.direction == "store"}
        foreign = sum(
            1
            for o in body
            for m in o.mem
            if m.direction == "store" and m.base not in STACK_BASES
        )
        redef = {r for o in body for r in o.defs}
        slots: dict[tuple[Any, ...], list[Op]] = defaultdict(list)
        for o in inner:
            for m in o.mem:
                if (
                    m.direction == "load"
                    and m.base in STACK_BASES
                    and m.key is not None
                    and m.key not in stores
                    and m.base not in redef
                ):
                    slots[m.key].append(o)
        for key, loads in slots.items():
            execs = sum(p.count(o.pc) for o in loads)
            gain = execs - lp.entries
            if gain <= 0:
                continue
            out.append(
                p.finding(
                    "invariant_stack_reload",
                    loads[0],
                    1.0,
                    "low",
                    "slot %s loaded %dx per iteration and not stored in loop %#x-%#x: keep it in a register "
                    "(register pressure; %d stores through other pointers might alias)"
                    % (key[2], len(loads), lp.head, lp.end, foreign),
                    count=gain,
                    loads=len(loads),
                )
            )
    return out


def _movable_before(blk: Sequence[Op], br: Op, upto: int = 2) -> int:
    """Instructions directly before `br` in its block that can move into its delay slots."""
    k = 0
    for o in reversed(blk[:-1]):
        if k >= upto or o.branch or o.cond or o.nop:
            break
        if (br.branch == "cond_jump" and o.flags_w) or o.defs & br.uses:
            break
        if o.form == "16a" or any(m.post and m.base == "I7" for m in o.mem):
            break
        k += 1
    return k


def det_delay_slots(p: Program) -> list[Finding]:
    out: list[Finding] = []
    for blk in p.blocks():
        br = blk[-1]
        if br.branch not in ("jump", "cond_jump", "call", "return", "indirect"):
            continue
        n = p.count(br.pc)
        if n <= 0:
            continue
        k = _movable_before(blk, br)
        taken = p.taken(br)
        if not br.delayed:
            if k == 0:
                continue
            gain = taken * k - (n - taken) * (2 - k)
            out.append(
                p.finding(
                    "fill_delay_slots",
                    br,
                    gain / n,
                    "medium" if br.branch != "cond_jump" else "low",
                    "non-delayed %s with %d independent instruction(s) before it: make it a delayed branch"
                    % (br.branch, k),
                    movable=k,
                    taken=taken,
                )
            )
        else:
            nops = [s for s in p.slots(br) if s.nop]
            m = min(len(nops), k)
            if m:
                out.append(
                    p.finding(
                        "fill_delay_slots",
                        br,
                        float(m),
                        "medium" if br.branch != "cond_jump" else "low",
                        "delayed %s with %d NOP slot(s) and %d movable instruction(s) before it"
                        % (br.branch, len(nops), k),
                        movable=k,
                    )
                )
    return out


def det_nops(p: Program) -> list[Finding]:
    """NOPs that are not delay slots of a branch (those are in fill_delay_slots)."""
    slot_pcs: set[int] = set()
    for o in p.ops.values():
        if o.delayed:
            slot_pcs |= {s.pc for s in p.slots(o)}
    out: list[Finding] = []
    for o in p.ops.values():
        if o.nop and o.pc not in slot_pcs and p.count(o.pc) > 0:
            out.append(p.finding("nop", o, 1.0, "medium", "NOP outside a delay slot"))
    return out


_CONV = re.compile(r"\b(fix|float|trunc|fix_by|float_by|trunc_by)\(")
_DIV = re.compile(r"\b(recips|rsqrts)\(")


def det_convert_divide(p: Program) -> list[Finding]:
    out: list[Finding] = []
    for blk in p.blocks():
        for i, o in enumerate(blk):
            m = re.match(r"^[RF](\d+) = float\([RF](\d+)\)", o.text)
            if m and not o.cond:
                d = "R" + m.group(1)
                for x in blk[i + 1 : i + 4]:
                    mm = re.match(r"^[RF](\d+) = (fix|trunc)\([RF](\d+)\)", x.text)
                    if mm and "R" + mm.group(3) == d:
                        out.append(
                            p.finding(
                                "int_float_roundtrip",
                                o,
                                2.0,
                                "low",
                                "int -> float -> int round trip (%#x): exact only while |n| < 2^24"
                                % x.pc,
                            )
                        )
                        break
            if _DIV.search(o.text) and "recips" in o.text:
                src = re.search(r"recips\([RF](\d+)\)", o.text)
                lp = p.innermost(o.pc)
                if not src or lp is None:
                    continue
                reg = "R" + src.group(1)
                body = [p.ops[q] for q in p.pcs if lp.head <= q <= lp.end]
                if any(reg in b.defs for b in body):
                    continue
                chain = set(o.defs)
                n = 1
                for x in blk[i + 1 : i + 14]:
                    if x.uses & chain and x.defs:
                        chain |= x.defs
                        n += 1
                trips = lp.iters - lp.entries
                out.append(
                    p.finding(
                        "divide_hoist",
                        o,
                        float(n),
                        "low",
                        "reciprocal of loop-invariant %s inside loop %#x-%#x: hoist the %d-instruction chain"
                        % (reg, lp.head, lp.end, n),
                        count=max(trips, 0.0) * p.count(o.pc) / max(lp.iters, 1.0),
                    )
                )
    return out


def det_dead_code(p: Program) -> list[Finding]:
    """Unprofiled code after an unconditional transfer that nothing targets (size only; not a proof of death)."""
    ld = p.leaders(after_branch=False)
    out: list[Finding] = []
    prev: Op | None = None
    for pc in p.pcs:
        o = p.ops[pc]
        if (
            prev is not None
            and prev.next_pc == pc
            and prev.branch in ("jump", "return")
            and not prev.cond
            and not prev.delayed
            and pc not in ld
            and p.count(pc) == 0
        ):
            out.append(
                p.finding(
                    "dead_code",
                    o,
                    0.0,
                    "low",
                    "after an unconditional %s, no static entry; code size only"
                    % prev.branch,
                    count=0.0,
                )
            )
        prev = o
    return out


DETECTORS: dict[str, Callable[[Program], list[Finding]]] = {
    "memory": det_memory,
    "copy_removal": det_copy_prop,
    "dead_definition": det_dead_def,
    "sw_loop_to_hw": det_sw_loops,
    "call_inline": det_calls,
    "fold_transfer": det_fold_transfer,
    "compute_pairs": det_compute_pairs,
    "redundant_compute": det_redundant_compute,
    "address_arith": det_address_arith,
    "invariant_reloads": det_invariant_reloads,
    "fill_delay_slots": det_delay_slots,
    "nop": det_nops,
    "convert_divide": det_convert_divide,
    "dead_code": det_dead_code,
}


def run_all(p: Program) -> list[Finding]:
    out: list[Finding] = []
    for fn in DETECTORS.values():
        out.extend(fn(p))
    return sorted((f for f in out if f.saving >= 0), key=lambda f: (-f.saving, f.pc))


# -- profile summaries


def class_mix(p: Program, pcs: Iterable[int] | None = None) -> dict[str, float]:
    mix: dict[str, float] = defaultdict(float)
    for q in p.pcs if pcs is None else pcs:
        o = p.ops.get(q)
        if o is not None:
            mix[o.klass()] += p.count(q)
    return dict(sorted(mix.items(), key=lambda kv: -kv[1]))


def function_profile(p: Program, top: int = 25) -> list[dict[str, Any]]:
    by: dict[int, list[int]] = defaultdict(list)
    for q in p.pcs:
        by[p.func_of(q)[0]].append(q)
    total_i = p.instructions() or 1.0
    total_c = p.cycles() or 1.0
    rows: list[dict[str, Any]] = []
    for entry, pcs in by.items():
        ins = p.instructions(pcs)
        if ins <= 0:
            continue
        cyc = p.cycles(pcs)
        rows.append(
            {
                "function": p.func_of(entry)[1],
                "entry": "%#x" % entry,
                "instructions": ins,
                "cycles": cyc,
                "share_pct": round(100 * cyc / total_c, 2),
                "instr_share_pct": round(100 * ins / total_i, 2),
                "mix": {k: round(v / ins, 3) for k, v in class_mix(p, pcs).items()},
            }
        )
    return sorted(rows, key=lambda r: -float(r["cycles"]))[:top]


def loop_profile(p: Program, top: int = 25) -> list[dict[str, Any]]:
    rows = []
    for lp in p.loops():
        if lp.executed <= 0:
            continue
        rows.append(
            {
                "kind": lp.kind,
                "function": lp.func,
                "span": "%#x-%#x" % (lp.head, lp.end),
                "depth": lp.depth,
                "static_ops": lp.static_ops,
                "iterations": lp.iters,
                "entries": lp.entries,
                "avg_trip": round(lp.iters / lp.entries, 1) if lp.entries else None,
                "instructions": lp.executed,
                "share_pct": round(100 * lp.executed / (p.instructions() or 1), 2),
            }
        )
    return rows[:top]


def apply_static_weights(
    p: Program, default_trip: int = 8, entries: Iterable[int] = ()
) -> None:
    """No profile: weight each op by the product of the trip counts of its enclosing loops (hardware loops use
    their literal trip count, others `default_trip`), and each function by the weights of its call sites (call
    cycles are cut at the edge that closes them; per-op loop weight is capped at 4096 and function weight at 1e6; a function nothing calls weighs 1)."""
    loops = p.loops()
    local: dict[int, float] = {}
    for pc in p.pcs:
        w = 1.0
        for lp in loops:
            if lp.head <= pc <= lp.end:
                w *= lp.trip or default_trip
        local[pc] = min(w, 4096.0)
    sites: dict[int, list[tuple[int, int]]] = defaultdict(
        list
    )  # caller entry -> [(callee entry, site pc)]
    for o in p.ops.values():
        if o.branch == "call" and o.target is not None:
            sites[p.func_of(o.pc)[0]].append((p.func_of(o.target)[0], o.pc))
    order: list[int] = []
    seen: set[int] = set()

    def dfs(f: int) -> None:
        seen.add(f)
        for g, _ in sites.get(f, ()):
            if g not in seen:
                dfs(g)
        order.append(f)

    roots = list(entries) or [a for a, _ in p.funcs]
    for e in roots:
        if e not in seen:
            dfs(e)
    rank = {f: i for i, f in enumerate(reversed(order))}  # callers before callees
    fw: dict[int, float] = defaultdict(float)
    for e in roots:
        fw[e] = 1.0
    for f in reversed(order):
        for g, pc in sites.get(f, ()):
            if rank[g] > rank[f]:
                fw[g] = min(fw[g] + max(fw[f], 1.0) * local[pc], 1e6)
    p.counts = {pc: max(fw.get(p.func_of(pc)[0], 0.0), 1.0) * local[pc] for pc in p.pcs}
    p.weights = "static-loop-nesting"
    p._loops = None
    p.stall = {}
    p.pair_stalls()


def summarize(findings: Sequence[Finding]) -> dict[str, dict[str, float]]:
    s: dict[str, dict[str, float]] = {}
    for f in findings:
        d = s.setdefault(f.detector, {"findings": 0, "saving_cycles": 0.0})
        d["findings"] += 1
        d["saving_cycles"] += f.saving
    return dict(sorted(s.items(), key=lambda kv: -kv[1]["saving_cycles"]))


def group_findings(findings: Sequence[Finding]) -> list[dict[str, Any]]:
    """Findings grouped by (detector, function, confidence), ranked by total estimated saving."""
    g: dict[tuple[str, str, str], list[Finding]] = defaultdict(list)
    for f in findings:
        g[(f.detector, f.function, f.confidence)].append(f)
    rows = []
    for (det, fn, conf), fs in g.items():
        best = max(fs, key=lambda f: f.saving)
        rows.append(
            {
                "detector": det,
                "function": fn,
                "confidence": conf,
                "sites": len(fs),
                "saving_cycles": sum(f.saving for f in fs),
                "best_site": best.where,
                "best_pc": "%#x" % best.pc,
                "note": best.note,
            }
        )
    return sorted(rows, key=lambda r: -r["saving_cycles"])
