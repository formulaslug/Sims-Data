"""
colscript -- a tiny language for defining new data columns.

One definition per line:

    power    = voltage * current
    energy   = integral(power, Time)
    accel    = derivative(speed, Time)
    smooth   = lowpass(speed, 0.9)       # drop the top 90% of the frequency range
    drift    = highpass(speed, 0.01)     # drop the bottom 1% (and the mean)
    band     = bandpass(speed, 0.01, 0.1)
    ratio    = (a + b) / 2 ^ -1 % 7      # comments start with "#"
    `my col` = `odd name (V)` * 1e3      # backticks for names with spaces etc.

Operators, loosest to tightest: + -   then * / %   then unary -   then ^ (right
associative, so -2^2 == -4 and 2^-1 == 0.5). Numbers broadcast against columns.
Later lines may use columns defined on earlier lines.

run(text, df) applies a script to a polars DataFrame. A line that fails (bad
syntax, unknown column, ...) is reported as a ScriptError and skipped; the
other lines still run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import numpy as np
import polars as pl
from scipy.integrate import cumulative_simpson
from scipy.fftpack import fft, ifft


# --------------------------------------------------------------------------
# functions -- name -> (callable, number of arguments)
# --------------------------------------------------------------------------
def integral(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Running integral of y over x (Simpson's rule), starting at 0."""
    return cumulative_simpson(y, x=x, initial=0.0)


def derivative(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """dy/dx with central differences (one-sided at the ends); x may be unevenly spaced."""
    return np.gradient(y, x)

def lowPass(array: np.ndarray, fraction: float):
    if type(array) == pl.Series:
        array = array.to_numpy()
    transform = fft(array)
    length = array.shape[0]
    end_length = ((1 - fraction)/2)
    start = round(length*end_length)
    end = round(length*(1-end_length))
    transform[start:end] = np.zeros_like(transform[start:end])
    output = ifft(transform)
    # output.real
    return output.real

def highPass(array: np.ndarray, fraction: float):
    """Counterpart of lowPass: zero the lowest `fraction` of the frequency range
    (the mean included, for fraction > 0) and keep the rest."""
    if type(array) == pl.Series:
        array = array.to_numpy()
    transform = fft(array)
    length = array.shape[0]
    cut = round(length*fraction/2)      # bins 0..cut-1 and their mirrors cut..1
    transform[:cut] = 0
    transform[length - cut + 1:] = 0
    return ifft(transform).real


def bandPass(array: np.ndarray, low: float, high: float):
    """Keep the part of the frequency range between fractions `low` and `high`
    (0 = constant, 1 = Nyquist): highPass removes below `low`, lowPass above `high`."""
    if not low < high:
        raise ValueError(f"the band's low end ({low:g}) must be below its high end ({high:g})")
    return highPass(lowPass(array, 1 - high), low)


@dataclass(frozen=True)
class Arg:
    """One function argument: a column (array), or a number within [lo, hi]."""
    name: str
    column: bool = True
    lo: float = -np.inf
    hi: float = np.inf


Y, X = Arg("y"), Arg("x")
FRACTION = Arg("fraction", column=False, lo=0.0, hi=1.0)

# name -> (callable, arguments). Arguments are checked before the call; a
# function can also raise ValueError, which is reported as an error on its line.
FUNCTIONS: dict[str, tuple[Callable[..., np.ndarray], tuple[Arg, ...]]] = {
    "integral": (integral, (Y, X)),
    "derivative": (derivative, (Y, X)),
    # fraction = share of the frequency range removed (lowpass: from the top,
    # highpass: from the bottom); bandpass keeps the range between low and high.
    "lowpass": (lowPass, (Y, FRACTION)),
    "highpass": (highPass, (Y, FRACTION)),
    "bandpass": (bandPass, (Y, Arg("low", column=False, lo=0.0, hi=1.0),
                            Arg("high", column=False, lo=0.0, hi=1.0))),
}


def _check_args(name: str, spec: tuple[Arg, ...], values: list, line: int) -> None:
    """Columns must be columns and numbers must be finite numbers inside their range."""
    for i, (arg, value) in enumerate(zip(spec, values), start=1):
        where = f"{name}() argument {i} ({arg.name})"
        is_column = np.ndim(value) > 0
        if arg.column and not is_column:
            raise ScriptError(line, f"{where} must be a column, got the number {value:g}")
        if arg.column:
            if not np.isfinite(value).all():
                bad = int(np.count_nonzero(~np.isfinite(value)))
                raise ScriptError(line, f"{where} has {bad} NaN/inf value(s)")
            continue
        if is_column:
            raise ScriptError(line, f"{where} must be a number, not a column")
        if not np.isfinite(value) or not arg.lo <= value <= arg.hi:
            raise ScriptError(line, f"{where} must be between {arg.lo:g} and {arg.hi:g}, "
                                    f"got {value:g}")
    if spec == (Y, X):
        x = values[1]
        if len(x) < 2:
            raise ScriptError(line, f"{name}() needs at least 2 rows")
        if np.any(np.diff(x) <= 0):
            i = int(np.argmax(np.diff(x) <= 0))
            raise ScriptError(line, f"{name}() x must be strictly increasing "
                                    f"(row {i + 1}: {x[i]:g} -> {x[i + 1]:g})")


# --------------------------------------------------------------------------
# errors / parsed form
# --------------------------------------------------------------------------
class ScriptError(Exception):
    def __init__(self, line: int, message: str):
        super().__init__(f"line {line}: {message}")
        self.line, self.message = line, message


@dataclass
class Num:
    value: float


@dataclass
class Col:
    name: str


@dataclass
class Call:
    name: str
    args: list


@dataclass
class Unary:
    op: str
    operand: object


@dataclass
class BinOp:
    op: str
    left: object
    right: object


@dataclass
class Statement:
    line: int                 # 1-based line number in the script
    target: str
    expr: object


# --------------------------------------------------------------------------
# tokenizer
# --------------------------------------------------------------------------
_TOKEN = re.compile(r"""
    (?P<space>\s+)
  | (?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)
  | (?P<name>[A-Za-z_][A-Za-z0-9_.]*)
  | `(?P<quoted>[^`]+)`
  | (?P<op>[-+*/%^(),=])
""", re.VERBOSE)


def _tokens(text: str, line: int) -> list[tuple[str, str]]:
    out, pos = [], 0
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if m is None:
            raise ScriptError(line, f"unexpected character {text[pos]!r}")
        pos = m.end()
        kind = m.lastgroup
        if kind == "space":
            continue
        # A `quoted` name is just a name that may contain any character but `.
        out.append(("name" if kind == "quoted" else kind, m.group(kind)))
    out.append(("end", ""))
    return out


# --------------------------------------------------------------------------
# parser (recursive descent)
# --------------------------------------------------------------------------
class _Parser:
    def __init__(self, tokens, line):
        self.tokens, self.i, self.line = tokens, 0, line

    def peek(self):
        return self.tokens[self.i]

    def take(self, kind=None, value=None):
        tok = self.tokens[self.i]
        if (kind and tok[0] != kind) or (value and tok[1] != value):
            want = value or kind
            got = tok[1] or "end of line"
            raise ScriptError(self.line, f"expected {want!r}, got {got!r}")
        self.i += 1
        return tok

    def at(self, *ops):
        kind, value = self.peek()
        return kind == "op" and value in ops

    def statement(self) -> Statement:
        target = self.take("name")[1]
        self.take("op", "=")
        expr = self.expr()
        self.take("end")
        return Statement(self.line, target, expr)

    def expr(self):
        node = self.term()
        while self.at("+", "-"):
            op = self.take()[1]
            node = BinOp(op, node, self.term())
        return node

    def term(self):
        node = self.unary()
        while self.at("*", "/", "%"):
            op = self.take()[1]
            node = BinOp(op, node, self.unary())
        return node

    def unary(self):
        if self.at("+", "-"):
            op = self.take()[1]
            return Unary(op, self.unary())
        return self.power()

    def power(self):
        node = self.atom()
        if self.at("^"):
            self.take()
            node = BinOp("^", node, self.unary())   # right associative, allows 2^-1
        return node

    def atom(self):
        kind, value = self.peek()
        if kind == "num":
            self.take()
            return Num(float(value))
        if kind == "name":
            self.take()
            if self.at("("):
                return Call(value, self.args())
            return Col(value)
        if self.at("("):
            self.take()
            node = self.expr()
            self.take("op", ")")
            return node
        raise ScriptError(self.line, f"unexpected {value or 'end of line'!r}")

    def args(self) -> list:
        self.take("op", "(")
        args = [] if self.at(")") else [self.expr()]
        while self.at(","):
            self.take()
            args.append(self.expr())
        self.take("op", ")")
        return args


def parse(text: str) -> tuple[list[Statement], list[ScriptError]]:
    """Parse every non-blank, non-comment line; bad lines become errors."""
    statements, errors = [], []
    for n, raw in enumerate(text.splitlines(), start=1):
        src = raw.split("#", 1)[0].strip()
        if not src:
            continue
        try:
            statements.append(_Parser(_tokens(src, n), n).statement())
        except ScriptError as e:
            errors.append(e)
    return statements, errors


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
_BINARY = {
    "+": np.add, "-": np.subtract, "*": np.multiply, "/": np.divide,
    "%": np.mod, "^": np.power,
}


def _evaluate(node, columns: Callable[[str], np.ndarray], line: int):
    if isinstance(node, Num):
        return node.value
    if isinstance(node, Col):
        return columns(node.name)
    if isinstance(node, Unary):
        value = _evaluate(node.operand, columns, line)
        return -value if node.op == "-" else value
    if isinstance(node, BinOp):
        return _BINARY[node.op](_evaluate(node.left, columns, line),
                                _evaluate(node.right, columns, line))
    if isinstance(node, Call):
        if node.name not in FUNCTIONS:
            raise ScriptError(line, f"unknown function {node.name!r} "
                                    f"(known: {', '.join(sorted(FUNCTIONS))})")
        func, spec = FUNCTIONS[node.name]
        if len(node.args) != len(spec):
            names = ", ".join(a.name for a in spec)
            raise ScriptError(line, f"{node.name}({names}) takes {len(spec)} arguments, "
                                    f"got {len(node.args)}")
        values = [_evaluate(a, columns, line) for a in node.args]
        _check_args(node.name, spec, values, line)
        return func(*values)
    raise TypeError(node)


def run(text: str, df: pl.DataFrame) -> tuple[pl.DataFrame, list[ScriptError]]:
    """Add the script's columns to df (as Float64). Returns the new frame and the
    errors of the lines that were skipped. Columns already in df can't be redefined."""
    statements, errors = parse(text)
    original = set(df.columns)
    new: dict[str, np.ndarray] = {}

    def column(name: str) -> np.ndarray:
        if name in new:
            return new[name]
        if name in original:
            return df[name].cast(pl.Float64, strict=False).to_numpy()
        raise KeyError(name)

    for st in statements:
        try:
            if st.target in original:
                raise ScriptError(st.line, f"{st.target!r} is a column of the file; "
                                           "pick a new name")
            with np.errstate(all="ignore"):       # x/0 -> inf, 0/0 -> nan, like numpy
                value = _evaluate(st.expr, column, st.line)
            new[st.target] = np.broadcast_to(np.asarray(value, dtype=np.float64),
                                             (df.height,)).copy()
        except ScriptError as e:
            errors.append(e)
        except KeyError as e:
            errors.append(ScriptError(st.line, f"unknown column {e.args[0]!r}"))
        except (ValueError, TypeError) as e:          # e.g. scipy rejecting x
            errors.append(ScriptError(st.line, str(e)))
    errors.sort(key=lambda e: e.line)
    if not new:
        return df, errors
    return df.with_columns([pl.Series(name, values) for name, values in new.items()]), errors
