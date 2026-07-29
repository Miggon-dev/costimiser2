from dataclasses import dataclass, field
from typing import Any
@dataclass
class Block:
    kind: str
    value: Any
    level: int|None=None
    language: str|None=None
@dataclass
class Document:
    slug: str
    title: str
    blocks: list[Block]=field(default_factory=list)
