"""OpenAI function-calling schemas for the two shop actions.

Kept apart from `tools.registry` (which executes) and from `llm.prompts` (which
composes context) so that the exact wire schema is one reviewable artefact: a
7B model obeys a tool description far more literally than prose rules, so these
strings are part of the safety surface and must say what NOT to do.
"""

from __future__ import annotations

import copy
from typing import Any

__all__ = [
    "SAVE_LEAD_SCHEMA",
    "ESCALATE_SCHEMA",
    "TOOL_SCHEMAS",
    "tool_schemas",
    "tool_schema",
    "tool_names",
]

SAVE_LEAD_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "save_lead",
        "description": (
            "Save a sales lead for Meher Sweets & Namkeen. Call this when the customer wants to "
            "place an order, a bulk order, a wedding order or a custom order, AND has given a name "
            "plus a phone number or an email address. Copy the name, phone and email exactly as the "
            "customer wrote them; never invent a contact detail. If a name or a contact is missing, "
            "ask the customer for it in your reply instead of calling this tool. Do not call this "
            "tool for a simple price, delivery or policy question."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The customer's full name, copied as written, e.g. 'Ritu Malhotra'.",
                },
                "need": {
                    "type": "string",
                    "description": (
                        "Summarise in one short sentence what the customer wants to order, e.g. "
                        "'30 large Diwali gift boxes for the office Diwali party on 3 November'. Use "
                        "the customer's own words and quantities. Never restate the question back."
                    ),
                },
                "phone": {
                    "type": "string",
                    "description": (
                        "The customer's Indian mobile number, 10 digits, no +91 and no spaces, e.g. "
                        "'9876543210'. Omit if the customer did not give a phone number."
                    ),
                },
                "email": {
                    "type": "string",
                    "description": (
                        "The customer's email address, copied exactly, e.g. 'ritu.m@example.com'. "
                        "Omit if the customer did not give an email address."
                    ),
                },
                "quantity": {
                    "type": "string",
                    "description": (
                        "The rough quantity as the customer said it, e.g. '30 large gift boxes'. "
                        "Omit if no quantity was given."
                    ),
                },
                "date": {
                    "type": "string",
                    "description": (
                        "The event or delivery date as YYYY-MM-DD, e.g. '2026-11-03'. Convert "
                        "'3 November' to '2026-11-03'. Omit if the customer gave no date."
                    ),
                },
            },
            "required": ["name", "need"],
            "additionalProperties": False,
        },
    },
}

ESCALATE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "escalate",
        "description": (
            "Hand this conversation to the Meher team. Call this for a complaint, a damaged or broken "
            "delivery, a refund or return request, any question the shop data cannot answer, or when "
            "the customer asks to speak to a human or the owner. In your reply, apologise once, say "
            "the team will reply, and for a damaged delivery mention that a photo within 2 hours lets "
            "us replace it. Do not call this tool for a routine price, delivery or policy question."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": (
                        "One short sentence saying why the team must take over, e.g. 'customer reports "
                        "a crushed gift box delivered 30 minutes ago and is very disappointed'."
                    ),
                }
            },
            "required": ["reason"],
            "additionalProperties": False,
        },
    },
}

#: Canonical order: the action the customer asked for, then the hand-off.
TOOL_SCHEMAS: list[dict[str, Any]] = [SAVE_LEAD_SCHEMA, ESCALATE_SCHEMA]


def tool_schemas() -> list[dict[str, Any]]:
    """The `tools` array for a chat-completions request. A copy, never the shared object."""
    return copy.deepcopy(TOOL_SCHEMAS)


def tool_schema(name: str) -> dict[str, Any] | None:
    """One schema by function name, or None when the name is unknown."""
    for schema in TOOL_SCHEMAS:
        if schema["function"]["name"] == name:
            return copy.deepcopy(schema)
    return None


def tool_names() -> list[str]:
    return [str(schema["function"]["name"]) for schema in TOOL_SCHEMAS]
