'''
Column mapping loader for issue #80.
Loads the chained transition files (fs2_to_fs3.json5, fs3_to_fs4.json5) and global_schema.json5,
expands template rules, and maps any car's parquet columns onto the global (frozen FS-4) column set.
Author: Lauwrence Tuy

Library use:
    from Data.FSLib.column_mapping import apply_mapping, plan_columns
    df = apply_mapping(pl.read_parquet(path), car="FS-3")
    plan = plan_columns(df.columns, car="FS-2")   # inspect without touching data

Command line:
    python -m Data.FSLib.column_mapping --expand                 # every expanded entry, all files
    python -m Data.FSLib.column_mapping --expand fs3_to_fs4      # one transition file
    python -m Data.FSLib.column_mapping --plan FS-3 file.parquet # what read() would do to a file

How a column is resolved (same order as the transition file headers):
    1. global_schema.reserved columns pass through untouched; filler_patterns are dropped.
    2. The column walks the chain from its car to the global car. In each transition file it is
       dropped, renamed (with an optional scale/offset), sent to legacy.<car>.<name>, or, for
       `global: true` renames, sent straight to its global name and the walk stops.
    3. global_schema.aliases are applied to the final name.
    A column a transition file does not cover goes to legacy.<car>.<name> with a warning, so no
    data is ever lost. Transforms compose along the chain: new = old * scale + offset.
    A `by_value` rename picks its scale per file from the median non-zero value, for signals whose
    unit changed between logs of the same car (FS-3 GLV voltage). If no range fits, the column goes
    to legacy.<car>.<name> with a warning instead of getting a guessed unit.
'''

from __future__ import annotations

import argparse
import itertools
import re
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import json5


##### SETTINGS


## Folder this file lives in, and the formula-slug folder four levels up, which the DBC paths
## inside the transition files are relative to.
    # column_mapping -> FSLib -> Data -> Sims-Data -> formula-slug
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]

SCHEMA_FILE = "global_schema.json5"
TRANSITION_GLOB = "*_to_*.json5"

BO_RE = re.compile(r"^BO_\s+\d+\s+(\w+)\s*:", re.M)  # " BO_ 400 SMPC_0x10_CHARGE_STATUS: 8 ..." -> message name
SG_RE = re.compile(r"^\s+SG_\s+(\w+)", re.M)          # " SG_ CHARGING : ..." -> signal name
PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


class MappingError(ValueError):
    '''A mapping file is malformed or inconsistent. Fix the json5 file.'''


class ColumnMappingWarning(UserWarning):
    '''A logged column was not covered by the mapping, or two columns mapped to the same name.'''


##### CAR NAMES


def normalize_car(car) -> str:
    '''
    Accepts "FS-3", "fs3", "FS3", "fs-3" or 3 and returns the canonical "FS-3".
    '''
    m = re.fullmatch(r"(?:fs)?-?(\d+)", str(car).strip().lower())
    if not m:
        raise ValueError(f"Unrecognized car {car!r}; expected something like 'FS-3'.")
    return f"FS-{int(m.group(1))}"


def short_car(car: str) -> str:
    '''"FS-3" -> "fs3", used in legacy column names.'''
    return normalize_car(car).replace("-", "").lower()


##### RESOLUTIONS


@dataclass(frozen=True)
class ValueRule:
    '''
    One branch of a `by_value` rename: applies when min <= the file's median non-zero value < max.
    Used where one signal was logged in different units in different files (FS-3 GLV voltage).
    '''
    min: float = float("-inf")
    max: float = float("inf")
    scale: float = 1.0
    offset: float = 0.0
    note: str = ""

    def matches(self, value: float) -> bool:
        return self.min <= value < self.max

    def describe(self) -> str:
        lo = "" if self.min == float("-inf") else f"{self.min:g} <= "
        hi = "" if self.max == float("inf") else f" < {self.max:g}"
        tf = f"x{self.scale:g}" + (f" {self.offset:+g}" if self.offset else "")
        return f"{lo}median{hi}: {tf}" + (f" ({self.note})" if self.note else "")


@dataclass(frozen=True)
class Resolution:
    '''
    What one transition file does with one key.

    kind
    ----------
    "keep"    renamed (or unchanged) to `name` in the next car's namespace; the walk continues
    "global"  renamed to global column `name`; the walk stops
    "legacy"  moved to legacy column `name`; the walk stops
    "drop"    discarded
    '''
    kind: str
    name: str | None = None
    scale: float = 1.0
    offset: float = 0.0
    section: str = ""     # which json5 section produced it, for --expand and error messages
    note: str = ""        # reason / candidate, for --expand
    by_value: tuple[ValueRule, ...] | None = None  # per-file scale picked from the data, replaces scale/offset


def _expand(template: dict, fields: tuple[str, ...]) -> list[dict]:
    '''
    Expands a template rule. Every {placeholder} in the named fields is replaced by each value in
    `vars`; with several placeholders, every combination is generated.
    '''
    vars_ = template.get("vars", {})
    used = set()
    for f in fields:
        if f in template:
            used |= set(PLACEHOLDER_RE.findall(template[f]))
    missing = used - set(vars_)
    if missing:
        raise MappingError(f"Rule {template.get('from')!r} uses {sorted(missing)} but `vars` does not define them.")
    names = sorted(used)
    out = []
    for combo in itertools.product(*(vars_[n] for n in names)):
        values = dict(zip(names, (str(v) for v in combo)))
        item = dict(template)
        for f in fields:
            if f in item:
                item[f] = PLACEHOLDER_RE.sub(lambda m: values[m.group(1)], item[f])
        out.append(item)
    return out


def _rename_resolution(value, section: str, schema_columns: set[str]) -> Resolution:
    '''
    Turns a `renames` / `rules` value into a Resolution. Accepted forms: "NAME",
    {to, scale, offset, global} or {to, by_value: [{min, max, scale, offset, note}, ...], global}.
    '''
    if isinstance(value, str):
        return Resolution("keep", value, section=section)
    if not isinstance(value, dict) or "to" not in value:
        raise MappingError(f"Bad rename value {value!r} in {section}.")
    is_global = bool(value.get("global", False))
    if is_global and value["to"] not in schema_columns:
        raise MappingError(f"{section}: global target {value['to']!r} is not a global_schema column.")

    by_value = None
    if "by_value" in value:
        if "scale" in value or "offset" in value:
            raise MappingError(f"{section}: {value['to']!r} sets both by_value and scale/offset; use one.")
        rules = value["by_value"]
        if not rules:
            raise MappingError(f"{section}: {value['to']!r} has an empty by_value list.")
        by_value = tuple(
            ValueRule(
                min=float(r.get("min", float("-inf"))),
                max=float(r.get("max", float("inf"))),
                scale=float(r.get("scale", 1.0)),
                offset=float(r.get("offset", 0.0)),
                note=str(r.get("note", "")),
            )
            for r in rules
        )
        ## Ranges must not overlap, or the chosen unit would depend on rule order.
        ordered = sorted(by_value, key=lambda r: r.min)
        for a, b in zip(ordered, ordered[1:]):
            if b.min < a.max:
                raise MappingError(f"{section}: by_value ranges for {value['to']!r} overlap.")

    return Resolution(
        "global" if is_global else "keep",
        value["to"],
        scale=float(value.get("scale", 1.0)),
        offset=float(value.get("offset", 0.0)),
        section=section,
        by_value=by_value,
    )


def _removed_note(entry: dict) -> str:
    parts = []
    if entry.get("candidate"):
        parts.append(f"candidate {entry['candidate']}")
    if entry.get("origin"):
        parts.append(f"origin {entry['origin']}")
    if entry.get("reason"):
        parts.append(entry["reason"])
    return "; ".join(parts)


##### DBC PARSING (only for removed_messages and ambiguous signal names)


def _dbc_messages(paths: list[str]) -> dict[str, list[str]] | None:
    '''
    {message name: [signal names]} across the given DBCs (relative to formula-slug), or None if
    any DBC is missing. Parsed with a regex so cantools is not needed at read time.
    '''
    messages: dict[str, list[str]] = {}
    for rel in paths:
        p = ROOT / rel
        if not p.exists():
            return None
        current = None
        for line in p.read_text(errors="replace").splitlines():
            bo = BO_RE.match(line)
            if bo:
                current = bo.group(1)
                messages.setdefault(current, [])
                continue
            sg = SG_RE.match(line)
            if sg and current is not None:
                messages[current].append(sg.group(1))
            elif line.strip() == "":
                current = None
    return messages


##### TRANSITION FILES


@dataclass
class Transition:
    '''One parsed and expanded transition file, e.g. fs2_to_fs3.json5.'''
    path: Path
    from_car: str
    to_car: str
    entries: dict[str, Resolution] = field(default_factory=dict)
    ambiguous: set[str] = field(default_factory=set)   # bare signal names defined in more than one message
    added: list[str] = field(default_factory=list)
    dbc_available: bool = True

    @property
    def legacy_prefix(self) -> str:
        return f"legacy.{short_car(self.from_car)}."

    @property
    def stem(self) -> str:
        return self.path.stem

    def _put(self, key: str, res: Resolution) -> None:
        if key in self.entries:
            prev = self.entries[key]
            raise MappingError(f"{self.path.name}: {key!r} appears in both {prev.section} and {res.section}.")
        self.entries[key] = res

    def lookup(self, column: str) -> Resolution | None:
        res = self.entries.get(column)
        if res is not None:
            return res
        if column in self.ambiguous:
            return Resolution("legacy", self.legacy_prefix + column, section="ambiguous",
                              note="signal name exists in several messages; message unknown")
        return None


def _load_transition(path: Path, schema_columns: set[str], filler: list[re.Pattern]) -> Transition:
    raw = json5.loads(path.read_text())
    if raw.get("format_version") != 1:
        raise MappingError(f"{path.name}: unsupported format_version {raw.get('format_version')!r}.")
    t = Transition(path, normalize_car(raw["from"]["car"]), normalize_car(raw["to"]["car"]))
    t.added = list(raw.get("added", []))
    legacy = t.legacy_prefix

    for key in raw.get("drop", []):
        t._put(key, Resolution("drop", section="drop"))

    for key in raw.get("unchanged", []):
        t._put(key, Resolution("keep", key, section="unchanged"))

    for key, value in raw.get("renames", {}).items():
        t._put(key, _rename_resolution(value, "renames", schema_columns))

    for rule in raw.get("rules", []):
        for item in _expand(rule, ("from", "to")):
            value = {k: item[k] for k in ("to", "scale", "offset", "global") if k in item}
            t._put(item["from"], _rename_resolution(value, "rules", schema_columns))

    for key, entry in raw.get("removed", {}).items():
        t._put(key, Resolution("legacy", legacy + key, section="removed", note=_removed_note(entry)))

    for rule in raw.get("removed_rules", []):
        for item in _expand(rule, ("from", "candidate")):
            t._put(item["from"], Resolution("legacy", legacy + item["from"], section="removed_rules",
                                            note=_removed_note(item)))

    for key, entry in raw.get("non_dbc", {}).items():
        t._put(key, Resolution("legacy", legacy + key, section="non_dbc", note=_removed_note(entry)))

    ## removed_messages and ambiguous names need the DBCs. Without them, the affected columns still
    ## end up in legacy.<car>.<name> through the unmapped fallback, just with a warning.
    messages = _dbc_messages([d["path"] for d in raw["from"].get("dbcs", [])])
    if messages is None:
        t.dbc_available = False
    else:
        seen: dict[str, int] = {}
        for sigs in messages.values():
            for s in set(sigs):
                seen[s] = seen.get(s, 0) + 1
        t.ambiguous = {s for s, n in seen.items() if n > 1 and s not in t.entries}

        for msg, opts in raw.get("removed_messages", {}).items():
            if msg not in messages:
                raise MappingError(f"{path.name}: removed_messages names {msg!r}, which no {t.from_car} DBC defines.")
            qualify = bool(opts.get("qualify", False))
            for sig in messages[msg]:
                if any(p.search(sig) for p in filler):
                    continue
                key = f"{msg}.{sig}" if qualify else sig
                if key in t.entries:
                    continue  # an explicit entry wins over the message-wide default
                t._put(key, Resolution("legacy", legacy + key, section="removed_messages",
                                       note=f"message {msg} removed"))
    return t


##### MAPPING SET (schema + chain)


@dataclass
class MappingSet:
    schema_car: str
    reserved: set[str]
    filler: list[re.Pattern]
    aliases: dict[str, str]
    columns: set[str]
    transitions: dict[str, Transition]   # keyed by from_car

    def chain(self, car: str) -> list[Transition]:
        '''Transitions to walk from `car` to the global car, in order.'''
        car = normalize_car(car)
        out, seen = [], set()
        while car != self.schema_car:
            if car in seen:
                raise MappingError(f"Transition files form a loop at {car}.")
            seen.add(car)
            if car not in self.transitions:
                known = sorted(set(self.transitions) | {self.schema_car})
                raise ValueError(f"No mapping from {car} to {self.schema_car}. Known cars: {known}.")
            t = self.transitions[car]
            out.append(t)
            car = t.to_car
        return out

    def validate(self) -> None:
        '''
        Every rename target must exist one step further down the chain: a key of the next transition
        file, or a global column (or alias) for the last one. Catches typos at load time.
        '''
        problems = []
        for t in self.transitions.values():
            nxt = self.transitions.get(t.to_car)
            for key, res in t.entries.items():
                if res.kind != "keep":
                    continue
                if t.to_car == self.schema_car:
                    ok = res.name in self.columns or res.name in self.aliases
                else:
                    ok = nxt is not None and (res.name in nxt.entries or res.name in self.reserved)
                if not ok:
                    problems.append(f"{t.path.name}: {key!r} -> {res.name!r} ({res.section}) is not known to the next step.")
        if problems:
            raise MappingError("Mapping files are inconsistent:\n  " + "\n  ".join(problems))


@lru_cache(maxsize=4)
def load_mappings(directory: str | Path = HERE) -> MappingSet:
    '''
    Loads global_schema.json5 and every *_to_*.json5 in `directory`. Cached; call
    load_mappings.cache_clear() after editing a file in a running session.
    '''
    directory = Path(directory)
    schema = json5.loads((directory / SCHEMA_FILE).read_text())
    filler = [re.compile(p) for p in schema.get("filler_patterns", [])]
    columns = set(schema["columns"])

    transitions: dict[str, Transition] = {}
    for path in sorted(directory.glob(TRANSITION_GLOB)):
        t = _load_transition(path, columns, filler)
        if t.from_car in transitions:
            raise MappingError(f"Two transition files start at {t.from_car}: {transitions[t.from_car].path.name}, {path.name}.")
        transitions[t.from_car] = t

    ms = MappingSet(
        schema_car=normalize_car(schema["source"]["car"]),
        reserved=set(schema.get("reserved", {})),
        filler=filler,
        aliases=dict(schema.get("aliases", {})),
        columns=columns,
        transitions=transitions,
    )
    ms.validate()
    return ms


##### PLANNING


@dataclass
class ColumnPlan:
    '''
    What apply_mapping() will do to a set of columns. Every input column appears in exactly one of
    renames (including unchanged and legacy moves) or dropped.
    '''
    car: str
    renames: dict[str, str] = field(default_factory=dict)               # source -> output name
    transforms: dict[str, tuple[float, float]] = field(default_factory=dict)  # source -> (scale, offset)
    dropped: list[str] = field(default_factory=list)
    legacy: list[str] = field(default_factory=list)                     # sources sent to legacy.*
    unmapped: list[str] = field(default_factory=list)                   # not covered by a mapping file
    ambiguous: list[str] = field(default_factory=list)                  # bare names that repeat across messages
    collisions: list[str] = field(default_factory=list)                 # moved to legacy because their target was taken
    detected: dict[str, str] = field(default_factory=dict)              # by_value columns: source -> what was picked
    unit_unknown: list[str] = field(default_factory=list)               # by_value matched no rule -> legacy
    per_file: list[str] = field(default_factory=list)                   # by_value, but planned without data
    missing_dbcs: list[str] = field(default_factory=list)

    @property
    def needs_warning(self) -> bool:
        return bool(self.unmapped or self.ambiguous or self.collisions or self.unit_unknown or self.missing_dbcs)

    def summary(self) -> str:
        changed = sum(1 for s, d in self.renames.items() if s != d)
        lines = [
            f"car {self.car}: {len(self.renames)} kept ({changed} renamed, {len(self.transforms)} transformed), "
            f"{len(self.legacy)} to legacy, {len(self.dropped)} dropped",
        ]
        for src, what in self.detected.items():
            lines.append(f"  unit detected: {src} {what}")
        for label, items in (("unmapped -> legacy", self.unmapped), ("ambiguous -> legacy", self.ambiguous),
                             ("name collision -> legacy", self.collisions),
                             ("unit not recognized -> legacy", self.unit_unknown),
                             ("unit decided per file (no data given)", self.per_file)):
            if items:
                lines.append(f"  {label}: {', '.join(items)}")
        if self.missing_dbcs:
            lines.append(f"  DBCs not found, removed_messages not expanded for: {', '.join(self.missing_dbcs)}")
        return "\n".join(lines)


def _resolve(column: str, car: str, chain: list[Transition], ms: MappingSet, probe):
    '''
    Walks one column down the chain. Returns (kind, name, scale, offset, flag, detail) where kind
    is "keep" / "legacy" / "drop" and flag is "", "unmapped", "ambiguous", "detected",
    "unit_unknown" or "per_file".

    probe(column, scale, offset) returns the median of the column's non-zero values after the
    transform so far, or None if it has none. It is only called for `by_value` renames.
    '''
    if column in ms.reserved:
        return "keep", column, 1.0, 0.0, "", ""
    if any(p.search(column) for p in ms.filler):
        return "drop", None, 1.0, 0.0, "", ""

    name, scale, offset = column, 1.0, 0.0
    flag, detail = "", ""
    for t in chain:
        res = t.lookup(name)
        if res is None:
            return "legacy", t.legacy_prefix + name, 1.0, 0.0, "unmapped", ""
        if res.kind == "drop":
            return "drop", None, 1.0, 0.0, "", ""
        if res.kind == "legacy":
            ## Legacy columns keep the values as logged, in the old car's units.
            return "legacy", res.name, 1.0, 0.0, "ambiguous" if res.section == "ambiguous" else "", ""

        step_scale, step_offset = res.scale, res.offset
        if res.by_value is not None:
            if probe is None:
                flag = "per_file"
            else:
                stat = probe(column, scale, offset)
                if stat is None and all(r.offset == 0 for r in res.by_value):
                    ## Only zeros and nulls: every rule leaves them unchanged, so no unit is needed.
                    rule = ValueRule(note="all zero/null")
                else:
                    rule = None if stat is None else next((r for r in res.by_value if r.matches(stat)), None)
                if rule is None:
                    ## No rule fits (or the column is all zero/null): keep the raw values in legacy
                    ## rather than guess a unit.
                    shown = "no non-zero values" if stat is None else f"median {stat:g}"
                    return ("legacy", f"legacy.{short_car(car)}.{column}", 1.0, 0.0, "unit_unknown",
                            f"{shown} matches no by_value rule in {t.path.name}")
                step_scale, step_offset = rule.scale, rule.offset
                shown = "no non-zero values" if stat is None else f"median {stat:g}"
                flag, detail = "detected", f"{shown} in {t.path.name} -> {rule.describe()}"

        ## Compose: (x * s1 + o1) * s2 + o2
        scale, offset = scale * step_scale, offset * step_scale + step_offset
        name = res.name
        if res.kind == "global":
            break

    name = ms.aliases.get(name, name)
    if flag == "" and not (name in ms.columns or name in ms.reserved):
        flag = "unmapped"
    return "keep", name, scale, offset, flag, detail


def plan_columns(columns, car, mappings: MappingSet | None = None, probe=None) -> ColumnPlan:
    '''
    Works out how a car's logged columns map onto the global set.

    Parameters
    ----------
    columns : iterable of column names as logged
    car : "FS-2", "FS-3", "FS-4" (any spelling normalize_car accepts)
    probe : optional function(column, scale, offset) -> median non-zero value, used by `by_value`
        renames to pick a unit per file. Without it those columns are listed in plan.per_file and
        their transform is left out. apply_mapping() always supplies one.
    '''
    ms = mappings or load_mappings()
    car = normalize_car(car)
    chain = ms.chain(car)
    plan = ColumnPlan(car=car, missing_dbcs=[t.path.name for t in chain if not t.dbc_available])

    taken: dict[str, str] = {}
    for col in columns:
        kind, name, scale, offset, flag, detail = _resolve(col, car, chain, ms, probe)
        if flag == "detected":
            plan.detected[col] = detail
        elif flag == "unit_unknown":
            plan.unit_unknown.append(f"{col} ({detail})")
        elif flag == "per_file":
            plan.per_file.append(col)
        if kind == "drop":
            plan.dropped.append(col)
            continue
        if name in taken:
            ## Two logged columns want the same output name. Keep the first, park the second.
            plan.collisions.append(f"{col} (wanted {name}, taken by {taken[name]})")
            kind, name, scale, offset, flag = "legacy", f"legacy.{short_car(car)}.{col}", 1.0, 0.0, ""
            plan.detected.pop(col, None)
            if name in taken:
                raise MappingError(f"Cannot place {col!r}: {name!r} is also taken.")
        taken[name] = col
        plan.renames[col] = name
        if (scale, offset) != (1.0, 0.0):
            plan.transforms[col] = (scale, offset)
        if kind == "legacy":
            plan.legacy.append(col)
        if flag == "unmapped":
            plan.unmapped.append(col)
        elif flag == "ambiguous":
            plan.ambiguous.append(col)
    return plan


##### APPLYING


def make_probe(df):
    '''
    Returns the probe plan_columns() needs for `by_value` renames: the median of a column's
    non-zero, non-null values after a scale/offset. Zeros are skipped because most signals read 0
    while their board is off. Works on a DataFrame or a LazyFrame (reads only that column).
    '''
    import polars as pl
    lazy = isinstance(df, pl.LazyFrame)

    def probe(col: str, scale: float, offset: float):
        expr = pl.col(col).cast(pl.Float64) * scale + offset
        q = df.select(expr.filter(expr != 0).median().alias("m"))
        value = (q.collect() if lazy else q).item()
        return None if value is None else float(value)

    return probe


def apply_mapping(df, car, *, warn: bool = True, return_plan: bool = False):
    '''
    Maps a polars DataFrame or LazyFrame from `car`'s column names onto the global column set.
    The input is not modified. Original files are never touched (Decision 4: read-time, in memory).

    Parameters
    ----------
    df : pl.DataFrame or pl.LazyFrame, columns as logged
    car : the car the file was logged on
    warn : emit one ColumnMappingWarning listing unmapped, ambiguous or colliding columns
    return_plan : also return the ColumnPlan

    Returns
    ----------
    The mapped frame (same type as the input), or (frame, plan) if return_plan.
    '''
    import polars as pl

    columns = df.collect_schema().names() if isinstance(df, pl.LazyFrame) else df.columns
    plan = plan_columns(columns, car, probe=make_probe(df))

    out = df
    if plan.dropped:
        out = out.drop(plan.dropped)
    if plan.transforms:
        out = out.with_columns(
            (pl.col(c).cast(pl.Float64) * s + o).alias(c) for c, (s, o) in plan.transforms.items()
        )
    renames = {s: d for s, d in plan.renames.items() if s != d}
    if renames:
        out = out.rename(renames)

    if warn and plan.needs_warning:
        warnings.warn(plan.summary(), ColumnMappingWarning, stacklevel=2)
    return (out, plan) if return_plan else out


##### COMMAND LINE


def _format_resolution(key: str, res: Resolution) -> str:
    if res.kind == "drop":
        target = "(dropped)"
    else:
        target = res.name
        if res.kind == "global":
            target += "  [global]"
    if res.by_value is not None:
        target += "  [by value: " + "; ".join(r.describe() for r in res.by_value) + "]"
    elif (res.scale, res.offset) != (1.0, 0.0):
        target += f"  [x{res.scale:g}" + (f" {res.offset:+g}" if res.offset else "") + "]"
    note = f"  # {res.note}" if res.note else ""
    return f"{key} -> {target}{note}"


def _cli_expand(ms: MappingSet, which: str | None) -> None:
    transitions = list(ms.transitions.values())
    if which:
        transitions = [t for t in transitions if t.stem == which or t.path.name == which]
        if not transitions:
            raise SystemExit(f"No transition file {which!r}. Have: {[t.stem for t in ms.transitions.values()]}")
    for t in transitions:
        print(f"## {t.path.name}  ({t.from_car} -> {t.to_car}, {len(t.entries)} entries)")
        if not t.dbc_available:
            print("## (DBCs not found: removed_messages not expanded)")
        by_section: dict[str, list[str]] = {}
        for key, res in t.entries.items():
            by_section.setdefault(res.section, []).append(_format_resolution(key, res))
        for section, lines in by_section.items():
            print(f"\n# {section} ({len(lines)})")
            for line in lines:
                print(line)
        if t.ambiguous:
            print(f"\n# ambiguous bare names -> {t.legacy_prefix}<name> ({len(t.ambiguous)})")
            print(", ".join(sorted(t.ambiguous)))
        print()


def _cli_plan(car: str, path: str) -> None:
    import polars as pl
    lf = pl.scan_parquet(path)
    plan = plan_columns(lf.collect_schema().names(), car, probe=make_probe(lf))
    print(plan.summary())
    print()
    for src, dst in plan.renames.items():
        tf = plan.transforms.get(src)
        extra = f"  [x{tf[0]:g}" + (f" {tf[1]:+g}" if tf[1] else "") + "]" if tf else ""
        if src != dst or tf:
            print(f"{src} -> {dst}{extra}")
    for col in plan.dropped:
        print(f"{col} -> (dropped)")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Column mapping loader (issue #80).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--expand", nargs="?", const="", metavar="FILE",
                   help="print every expanded entry, optionally for one transition file (e.g. fs3_to_fs4)")
    g.add_argument("--plan", nargs=2, metavar=("CAR", "PARQUET"),
                   help="show how a parquet file's columns would be mapped")
    args = ap.parse_args(argv)

    if args.plan:
        _cli_plan(*args.plan)
    else:
        _cli_expand(load_mappings(), args.expand or None)


if __name__ == "__main__":
    main()
