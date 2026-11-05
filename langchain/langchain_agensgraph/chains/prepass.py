"""Model-written Cypher made right for the graph and the server it runs on, before it is sent.

A model writes the Cypher it has seen most, which is Neo4j's. Some of it AgensGraph reads
differently, and a few forms return wrong rows rather than failing on some releases.
:func:`prepare` rewrites what has one right spelling on every release and refuses what has none,
naming the fix so the refusal can go back to the model:

- ``-->``, ``<--`` and ``--`` are written with brackets, ``-[]->``, ``<-[]-`` and ``-[]-``, which
  every release reads; the bare forms are refused or, mid-pattern, misread.
- A variable-length edge with no name returns only zero-length paths on 2.14 and 2.15, so one is
  given a name (``vle_unnamed_wrong``). One inside ``shortestpath()`` or ``allshortestpaths()`` is
  left as it is: those read it right, and refuse a named one where the call is an expression.
- ``nodes(p)[-1]`` and ``relationships(p)[-1]`` are null before 2.18; ``last(...)`` is not
  (``element_array_negative_index``).
- ``min()`` and ``max()`` over a property compare as text on 2.15 to 2.17, so on those releases
  one without a cast is refused (``jsonb_minmax_wrong``).
- ``toFloat()`` and ``toInteger()`` of a property are wrong or missing on every release: casts
  are refused in their place.

The graph's schema drives the rest: :func:`quote_identifiers` quotes the names it stores with
capitals, which an unquoted name would miss, and :func:`correct_directions` turns round an edge
written against the only direction the schema has for it.

Only text outside string literals, quoted names and comments is read or changed.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any

from agensgraph import Capabilities

from ..errors import UnsafeCypherError

_ARROWS = [
    (re.compile(r"<--(?!>)"), "<-[]-"),
    (re.compile(r"(?<![<\]\-])-->"), "-[]->"),
    (re.compile(r"(?<=\))(\s*)--(\s*)(?=\()"), r"\1-[]-\2"),
]
_UNNAMED_VLE = re.compile(r"\[\s*(:[^\]\*]*)?\*")
_SHORTEST = re.compile(r"\b(?:shortestpath|allshortestpaths)\s*\(", re.I)
_NEGATIVE_INDEX = re.compile(r"\b(nodes|relationships)\s*\(\s*(\w+)\s*\)\s*\[\s*-1\s*\]", re.I)
_MINMAX = re.compile(r"\b(min|max)\s*\(\s*([^()]*?)\s*\)", re.I)
_CAST = re.compile(r"::\s*\"?\w+\"?\s*$")
_TO_NUMBER = re.compile(r"\b(toFloat|toInteger)\s*\(", re.I)

_FENCED = re.compile(r"```[ \t]*(?:cypher|sql)?[ \t]*\n?(.*?)```", re.I | re.S)
_FENCE = re.compile(r"```(?:cypher|sql)?", re.I)

# Identifiers the model may have left unquoted. ":" after ":" is a cast, not a label.
_LABEL = re.compile(r'(?<!:):(?!")([A-Za-z_][A-Za-z0-9_]*)')
_PROP_KEY = re.compile(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)")
_PROP_ACCESS = re.compile(r'\.(?!")([A-Za-z_][A-Za-z0-9_]*)\b')

_NAME = r'(?:"(?:[^"]|"")+"|[^\W\d]\w*)'
_HOP = re.compile(
    r"(?P<left>\([^()]*\))\s*(?P<head><?-)\s*\[(?P<edge>[^\[\]]*)\]\s*(?P<tail>->?)"
    r"(?=\s*(?P<right>\([^()]*\)))"
)
_NODE = re.compile(rf"\(\s*(?P<var>[^\W\d]\w*)?\s*(?::\s*(?P<label>{_NAME}))?")
_EDGE = re.compile(rf"\s*(?:[^\W\d]\w*)?\s*:\s*(?P<type>{_NAME})")
_BOUND = re.compile(rf"\(\s*(?P<var>[^\W\d]\w*)\s*:\s*(?P<label>{_NAME})")


def strip_fences(text: str) -> str:
    """The statement out of what the model wrapped it in: a fenced block's contents if there
    is one, without fence markers and a trailing semicolon either way."""
    fenced = _FENCED.search(text)
    if fenced:
        text = fenced.group(1)
    return _FENCE.sub("", text).strip().rstrip(";").strip()


def prepare(text: str, capabilities: Capabilities) -> str:
    """The statement with every rewrite this server needs made; refused if it has a form that
    is wrong here and has no rewrite."""
    for pattern, template in _ARROWS:
        text = _substituted(text, pattern, _expanding(template))
    if capabilities.has("vle_unnamed_wrong"):
        names = (f"vle{n}" for n in range(1000))
        shortest = _call_spans(_masked(text, names=False), _SHORTEST)
        text = _substituted(
            text,
            _UNNAMED_VLE,
            lambda m, t: (
                t[m.start() : m.end()]
                if any(start <= m.start() < end for start, end in shortest)
                else f"[{next(names)}{t[m.start(1) : m.end(1)]}*"
            ),
        )
    if not capabilities.has("element_array_negative_index"):
        text = _substituted(text, _NEGATIVE_INDEX, lambda m, t: f"last({m.group(1)}({m.group(2)}))")
    code = _masked(text, names=False)
    if capabilities.has("jsonb_minmax_wrong"):
        for match in _MINMAX.finditer(code):
            if "." in match.group(2) and not _CAST.search(match.group(2)):
                name, argument = match.group(1), text[match.start(2) : match.end(2)]
                raise UnsafeCypherError(
                    f"{name}({argument}) compares the property as text on this release; cast it "
                    f"inside: {name}({argument}::float8), or ::int8, ::text, ::date"
                )
    found = _TO_NUMBER.search(code)
    if found:
        raise UnsafeCypherError(
            f"{found.group(1)}() of a property is unreliable on AgensGraph; cast instead: "
            "x::float8 or x::int8"
        )
    return text


def _call_spans(code: str, call: re.Pattern[str]) -> list[tuple[int, int]]:
    """Where each call that ``call`` finds in ``code`` runs, from its name to its closing
    bracket."""
    spans = []
    for match in call.finditer(code):
        depth, end = 1, match.end()
        while end < len(code) and depth:
            depth += {"(": 1, ")": -1}.get(code[end], 0)
            end += 1
        spans.append((match.start(), end))
    return spans


def _expanding(template: str) -> Callable[[re.Match[str], str], str]:
    return lambda match, _text: match.expand(template)


def _substituted(
    text: str, pattern: re.Pattern[str], replace: Callable[[re.Match[str], str], str]
) -> str:
    """``pattern`` replaced in the code of ``text``: matched with strings, quoted names and
    comments blanked, so only code matches, and replaced in the text itself."""
    out, last = [], 0
    for match in pattern.finditer(_masked(text, names=False)):
        out += [text[last : match.start()], replace(match, text)]
        last = match.end()
    out.append(text[last:])
    return "".join(out)


def case_sensitive_names(structured_schema: Mapping[str, Any]) -> set[str]:
    """The labels, types and property names in the graph spelled with a capital.

    Folding runs both ways: a name written unquoted is stored folded, so ``n.firstName`` is right
    for data written as ``{"firstName": ...}`` and wrong for data written as ``{firstName: ...}``,
    stored as ``firstname``. Only the schema knows which exists, so it decides what is quoted.
    """
    names: set[str] = set()

    def add(value: Any) -> None:
        if isinstance(value, str) and value != value.lower():
            names.add(value)

    for section in ("node_props", "rel_props"):
        for label, props in (structured_schema.get(section) or {}).items():
            add(label)
            for prop in props or []:
                add(prop.get("property") if isinstance(prop, dict) else prop)
    for rel in structured_schema.get("relationships") or []:
        if isinstance(rel, dict):
            for key in ("type", "start", "end"):
                add(rel.get(key))
    return names


def quote_identifiers(cypher: str, known: set[str] | None = None) -> str:
    """The statement with each bare name the graph stores with capitals quoted.

    Only names in ``known`` are quoted: quoting one the graph stores folded would turn a working
    statement into one that matches nothing. String literals, quoted names and comments are left
    alone.
    """
    allowed = known or set()

    def repair(fragment: str) -> str:
        fragment = _LABEL.sub(
            lambda m: f':"{m.group(1)}"' if m.group(1) in allowed else m.group(0), fragment
        )
        fragment = _PROP_KEY.sub(
            lambda m: (
                f'{m.group(1)}"{m.group(2)}"{m.group(3)}' if m.group(2) in allowed else m.group(0)
            ),
            fragment,
        )
        return _PROP_ACCESS.sub(
            lambda m: f'."{m.group(1)}"' if m.group(1) in allowed else m.group(0), fragment
        )

    return "".join(repair(piece) if code else piece for code, piece in _pieces(cypher))


def correct_directions(cypher: str, relationships: Iterable[Mapping[str, Any]]) -> str:
    """The statement with each edge written against its schema direction turned round.

    An edge is turned only when its type and both end labels are known, from the pattern or from
    where the variable was bound, and the schema has the edge the other way and not this way.
    Anything else is left as written: undirected and variable-length edges, unknown labels, and
    edges the schema has in neither direction.
    """
    triples = {(rel["start"], rel["type"], rel["end"]) for rel in relationships}
    if not triples:
        return cypher
    masked = _masked(cypher, names=True)
    bound: dict[str, str] = {}
    for match in _BOUND.finditer(masked):
        bound.setdefault(match.group("var"), _unquoted(match.group("label")))
    edits: list[tuple[int, int, str]] = []
    for hop in _HOP.finditer(masked):
        head, tail, edge = hop.group("head"), hop.group("tail"), hop.group("edge")
        if (head == "<-") == (tail == "->") or "*" in edge:
            continue
        typed = _EDGE.match(edge)
        left, right = _label(hop.group("left"), bound), _label(hop.group("right"), bound)
        if typed is None or left is None or right is None:
            continue
        kind = _unquoted(typed.group("type"))
        written = (left, kind, right) if tail == "->" else (right, kind, left)
        turned = (written[2], kind, written[0])
        if written in triples or turned not in triples:
            continue
        flipped = ("-", "->") if head == "<-" else ("<-", "-")
        edits.append((hop.start("head"), hop.end("head"), flipped[0]))
        edits.append((hop.start("tail"), hop.end("tail"), flipped[1]))
    for start, end, text in sorted(edits, reverse=True):
        cypher = cypher[:start] + text + cypher[end:]
    return cypher


def _label(node: str, bound: Mapping[str, str]) -> str | None:
    found = _NODE.match(node)
    if found is None:
        return None
    if found.group("label"):
        return _unquoted(found.group("label"))
    return bound.get(found.group("var") or "")


def _unquoted(name: str) -> str:
    """A name as the server stores it: a quoted one as written, a bare one folded."""
    if name.startswith('"'):
        return name[1:-1].replace('""', '"')
    return name.lower()


def _masked(text: str, *, names: bool) -> str:
    """The text, the same length, with the inside of each string literal blanked, and of each
    quoted name unless ``names``, and comments made spaces."""
    out = []
    for code, piece in _pieces(text):
        if code or names and piece.startswith('"'):
            out.append(piece)
        elif piece.startswith(("'", '"', "`")):
            out.append(piece[0] + "x" * max(len(piece) - 2, 0) + piece[-1:])
        else:
            out.append(" " * len(piece))
    return "".join(out)


def _pieces(text: str) -> Iterator[tuple[bool, str]]:
    """The text cut into code and not-code: string literals, quoted names and comments."""
    i = 0
    start = 0
    length = len(text)
    while i < length:
        char = text[i]
        if char in ("'", '"', "`"):
            if start < i:
                yield True, text[start:i]
            end = _closing(text, i, char)
            yield False, text[i:end]
            i = start = end
            continue
        if text.startswith("//", i) or text.startswith("--", i) and _is_comment(text, i):
            if start < i:
                yield True, text[start:i]
            end = text.find("\n", i)
            end = length if end < 0 else end
            yield False, text[i:end]
            i = start = end
            continue
        if text.startswith("/*", i):
            if start < i:
                yield True, text[start:i]
            end = text.find("*/", i + 2)
            end = length if end < 0 else end + 2
            yield False, text[i:end]
            i = start = end
            continue
        i += 1
    if start < length:
        yield True, text[start:]


def _closing(text: str, i: int, quote: str) -> int:
    """Where a quoted run that opens at ``i`` ends: after its closing quote, a doubled quote and
    (in a string) a backslash escaping the next character staying inside it."""
    j = i + 1
    while j < len(text):
        if quote == "'" and text[j] == "\\":
            j += 2
            continue
        if text[j] == quote:
            if j + 1 < len(text) and text[j + 1] == quote:
                j += 2
                continue
            return j + 1
        j += 1
    return len(text)


def _is_comment(text: str, i: int) -> bool:
    """Whether ``--`` at ``i`` starts a SQL comment rather than an edge between two nodes:
    ``(a)--(b)``, ``(a)-->(b)`` or ``(a)<--(b)``."""
    before = text[:i].rstrip()
    after = text[i + 2 :].removeprefix(">").lstrip()
    return not (before.endswith((")", "<")) and after.startswith("("))
