"""Loads the read-only shop data in data/ into structured form.

data/business.md, data/prices.csv and data/policies.md are the source of truth and
are never modified. This module parses them exactly once at startup.
"""

from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path

from ..types import SKU, Section

_SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


def slugify_section(heading: str) -> str:
    """'Bulk orders' -> 'bulk-orders' (lower case, spaces to hyphens)."""
    cleaned = re.sub(r"[^\w\s-]", "", heading, flags=re.UNICODE).strip().lower()
    return re.sub(r"[\s_]+", "-", cleaned)


def _split_sections(md: str, doc: str) -> list[Section]:
    """Split a markdown file into its '## heading' blocks.

    Text before the first '##' is treated as a preamble section so that nothing
    in the source file is silently dropped.
    """
    matches = list(_SECTION_RE.finditer(md))
    sections: list[Section] = []
    if not matches:
        return [Section(doc=doc, heading=doc, source_id=f"{doc}#overview", body=md.strip())]

    preamble = md[: matches[0].start()].strip()
    if preamble:
        title = preamble.splitlines()[0].lstrip("# ").strip() or "Overview"
        sections.append(
            Section(doc=doc, heading=title, source_id=f"{doc}#{slugify_section(title)}", body=preamble)
        )

    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        sections.append(
            Section(
                doc=doc,
                heading=heading,
                source_id=f"{doc}#{slugify_section(heading)}",
                body=md[start:end].strip(),
            )
        )
    return sections


def _parse_price(row: dict[str, str]) -> SKU:
    contains_raw = (row.get("contains") or "").strip()
    allergens = [a.strip() for a in contains_raw.split(";") if a.strip()]
    price_raw = (row.get("price_inr") or "0").strip().replace(",", "")
    return SKU(
        sku=(row.get("sku") or "").strip(),
        item=(row.get("item") or "").strip(),
        pack=(row.get("pack") or "").strip(),
        price_inr=int(float(price_raw)),
        type=(row.get("type") or "").strip(),
        contains_raw=contains_raw,
        allergens=allergens,
        shelf_life_days=int(float((row.get("shelf_life_days") or "0").strip() or 0)),
    )


class Corpus:
    """Immutable in-memory view of data/."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self._validate_paths()
        self.business: list[Section] = _split_sections(
            (self.data_dir / "business.md").read_text(encoding="utf-8"), "business.md"
        )
        self.policies: list[Section] = _split_sections(
            (self.data_dir / "policies.md").read_text(encoding="utf-8"), "policies.md"
        )
        with (self.data_dir / "prices.csv").open(encoding="utf-8", newline="") as fh:
            self.skus: list[SKU] = [_parse_price(r) for r in csv.DictReader(fh)]
        self.sections: list[Section] = self.business + self.policies
        self.source_ids: set[str] = {s.source_id for s in self.sections}
        self.source_ids |= {s.source_id for s in self.skus}

    def _validate_paths(self) -> None:
        for name in ("business.md", "prices.csv", "policies.md"):
            p = self.data_dir / name
            if not p.exists():
                raise FileNotFoundError(f"Required data file missing: {p}")

    # -- lookups -----------------------------------------------------------
    def section(self, source_id: str) -> Section | None:
        for s in self.sections:
            if s.source_id == source_id:
                return s
        return None

    def sku(self, sku_id: str) -> SKU | None:
        for s in self.skus:
            if s.sku.lower() == sku_id.lower():
                return s
        return None

    def catalog_prices(self) -> list[int]:
        return sorted({s.price_inr for s in self.skus})

    def skus_of_type(self, type_name: str) -> list[SKU]:
        return [s for s in self.skus if s.type.lower() == type_name.lower()]


@lru_cache(maxsize=4)
def load_corpus(data_dir: str | Path | None = None) -> Corpus:
    from ..config import get_config

    return Corpus(Path(data_dir) if data_dir else get_config().runtime.data_dir)


def all_source_ids(data_dir: str | Path | None = None) -> set[str]:
    return set(load_corpus(data_dir).source_ids)
