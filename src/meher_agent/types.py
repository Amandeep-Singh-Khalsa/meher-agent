"""Shared data contracts for the Meher Sweets agent.

Every layer (retrieval, grounding, tools, agent loop, API, eval harness)
imports its types from here. This module is deliberately dependency-light and
must not import from any other project module, so that it can be imported from
anywhere without cycles.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Corpus / retrieval
# --------------------------------------------------------------------------

DocName = Literal["business.md", "prices.csv", "policies.md"]


class SKU(BaseModel):
    """One row of data/prices.csv, parsed once at startup."""

    sku: str
    item: str
    pack: str
    price_inr: int
    type: str
    contains_raw: str
    allergens: list[str]
    shelf_life_days: int

    @property
    def source_id(self) -> str:
        return f"prices.csv#{self.sku}"

    @property
    def pack_grams(self) -> int | None:
        """Pack size in grams, or None when the pack is not a weight (1 box/piece)."""
        p = self.pack.lower().replace(" ", "")
        if p.endswith("kg"):
            return int(float(p[:-2]) * 1000)
        if p.endswith("g"):
            return int(float(p[:-1]))
        return None

    @property
    def unit_kind(self) -> str:
        """One of: kg, g, box, piece."""
        p = self.pack.lower().replace(" ", "")
        if p.endswith("kg"):
            return "kg"
        if p.endswith("g"):
            return "g"
        if "piece" in p:
            return "piece"
        return "box"

    def catalog_line(self) -> str:
        """Verbatim, human-readable line handed to the model as grounding."""
        bits = [f"{self.item} ({self.pack}) - Rs {self.price_inr:,}"]
        bits.append(f"SKU {self.sku}")
        bits.append(f"type: {self.type}")
        bits.append(f"contains: {', '.join(self.allergens) if self.allergens else 'none listed'}")
        bits.append(f"shelf life: {self.shelf_life_days} days (0 = eat the same day)")
        return "; ".join(bits)


class Section(BaseModel):
    """A '## heading' block of business.md or policies.md."""

    doc: str
    heading: str
    source_id: str
    body: str

    @property
    def text(self) -> str:
        return f"{self.heading}\n{self.body}".strip()


class RetrievedSection(BaseModel):
    section: Section
    score: float


class RetrievedSKU(BaseModel):
    sku: SKU
    score: float
    #: Why this SKU was matched, e.g. "sku:KK-1000 qty=2 via '2 kg kaju katli'".
    reason: str = ""


class ResolvedOrderItem(BaseModel):
    """A deterministic, exact interpretation of part of the customer's message."""

    sku: str
    item: str
    pack: str
    qty: int
    unit_price_inr: int
    matched_text: str = ""
    #: True when the request could be satisfied only by combining packs.
    approximated: bool = False


class Resolution(BaseModel):
    items: list[ResolvedOrderItem] = Field(default_factory=list)
    #: Delivery distance in km, when the customer stated one.
    distance_km: int | None = None
    #: Retrieval snippets to show the model, already carrying their source ids.
    sections: list[RetrievedSection] = Field(default_factory=list)
    sku_hits: list[RetrievedSKU] = Field(default_factory=list)
    #: Human-readable notes about what could not be resolved exactly.
    unresolved: list[str] = Field(default_factory=list)


class Briefing(BaseModel):
    """Everything the model is allowed to know for one turn."""

    question: str
    language: str
    resolved: Resolution
    #: Rupee values the reply is permitted to state. Enforced post-hoc by guard.py.
    allowed_amounts: list[int] = Field(default_factory=list)
    #: Set when the turn is out of scope for the shop (not a shop question).
    out_of_scope_hint: bool = False


# --------------------------------------------------------------------------
# Deterministic money
# --------------------------------------------------------------------------


class LineItem(BaseModel):
    sku: str
    item: str
    pack: str
    qty: int
    unit_price_inr: int
    line_total_inr: int


class Quote(BaseModel):
    """Computed entirely in Python. The model only verbalises this."""

    lines: list[LineItem] = Field(default_factory=list)
    subtotal_inr: int = 0
    #: Only ever 5 (gift-box discount) or 0. Enforced here, never by the model.
    discount_pct: int = 0
    discount_inr: int = 0
    #: None when delivery is impossible (>8 km) or not stated.
    delivery_fee_inr: int | None = None
    delivery_free: bool = False
    delivery_possible: bool = True
    total_inr: int = 0
    #: 30% advance for bulk orders, else 0.
    advance_pct: int = 0
    advance_inr: int = 0
    requires_bulk_notice: bool = False
    notes: list[str] = Field(default_factory=list)

    def all_amounts(self) -> list[int]:
        out: list[int] = {
            self.subtotal_inr,
            self.discount_inr,
            self.total_inr,
            self.advance_inr,
        }
        for line in self.lines:
            out.update({line.unit_price_inr, line.line_total_inr})
        if self.delivery_fee_inr is not None:
            out.add(self.delivery_fee_inr)
        return sorted(v for v in out if v > 0)


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------

ToolName = Literal["save_lead", "escalate"]


class Action(BaseModel):
    """Serialised into the `actions` array of the POST /chat response.

    Shape is fixed by the task spec:
        {"type": "save_lead", "args": {...}}   # args are POST-validation
    """

    type: str
    args: dict[str, Any] = Field(default_factory=dict)
    ok: bool = True
    error: str | None = None

    def to_public(self) -> dict[str, Any]:
        return {"type": self.type, "args": self.args}


class Lead(BaseModel):
    name: str
    need: str
    phone: str | None = None
    email: str | None = None
    quantity: str | None = None
    date: str | None = None
    conversation_id: str = ""
    created_at: str = ""

    def to_public_masked(self, mask: Any) -> dict[str, Any]:
        return {
            "name": self.name,
            "email": mask(self.email) if self.email else None,
            "phone": mask(self.phone) if self.phone else None,
            "need": self.need,
            "conversation_id": self.conversation_id,
            "created_at": self.created_at,
        }


class ToolResult(BaseModel):
    ok: bool
    #: JSON-serialised payload handed back to the model as a `tool` message.
    content: str
    error: str | None = None


# --------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------


class TurnRecord(BaseModel):
    role: str
    content: str


class AgentOutcome(BaseModel):
    reply: str = ""
    sources: list[str] = Field(default_factory=list)
    actions: list[Action] = Field(default_factory=list)
    handoff: bool = False
    #: Diagnostics for the eval report; not part of the public API contract.
    model_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_errors: list[str] = Field(default_factory=list)
    guard_repairs: int = 0
    intent_repairs: int = 0
    step_limit_hit: bool = False
    used_fallback_reply: bool = False


class ChatRequest(BaseModel):
    conversation_id: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=8000)


class ChatResponse(BaseModel):
    reply: str
    sources: list[str] = Field(default_factory=list)
    actions: list[dict[str, Any]] = Field(default_factory=list)
    handoff: bool = False
