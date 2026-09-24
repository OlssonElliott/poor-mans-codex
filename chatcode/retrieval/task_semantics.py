"""Shared deterministic parsing of task requirements and retrieval surfaces.

This module owns the reusable vocabulary used to split a user task into bounded
requirement surfaces, tokenize those surfaces, infer broad roles, and decide
whether a task is complex enough to need multi-surface handling.

It deliberately contains no repository access and performs no model calls.
"""
from __future__ import annotations

import re


MAX_TASK_SURFACES = 12

TASK_SYNONYMS = {
    "bild": {
        "image",
        "images",
        "upload",
        "attachment",
        "attachments",
    },
    "bilder": {
        "image",
        "images",
        "upload",
        "attachment",
        "attachments",
    },
    "mail": {
        "mail",
        "email",
        "smtp",
        "contact",
    },
    "mejl": {
        "mail",
        "email",
        "smtp",
        "contact",
    },
    "skicka": {
        "send",
        "submit",
        "post",
        "request",
    },
    "formulär": {
        "form",
        "contact",
        "submit",
    },
}


STOP_WORDS = {
    "att",
    "det",
    "den",
    "som",
    "och",
    "för",
    "med",
    "hur",
    "var",
    "vad",
    "hitta",
    "koden",
    "hanterar",
}


def get_task_words(task: str) -> set[str]:
    words = {
        word.lower()
        for word in re.findall(
            r"[A-Za-zÀ-ÖØ-öø-ÿ0-9_]+",
            task,
        )
        if len(word) >= 3
    }

    words -= STOP_WORDS

    expanded = set(words)

    for word in words:
        expanded.update(
            TASK_SYNONYMS.get(
                word,
                set(),
            )
        )

    return expanded



REQUIREMENT_ROLE_TERMS = {
    "persistence": frozenset({
        "database", "db", "persist", "persistence", "schema", "storage", "store",
        "save", "saved", "load", "loaded", "table", "repository",
        "databas", "lagra", "lagring", "spara", "sparas", "ladda", "tabell",
    }),
    "transport": frozenset({
        "api", "endpoint", "route", "request", "response", "http", "handler",
        "controller", "server",
    }),
    "ui": frozenset({
        "ui", "frontend", "dashboard", "dialog", "modal", "editor", "button",
        "form", "component", "render", "display", "view", "panel",
        "gränssnitt", "knapp", "visa",
    }),
    "tests": frozenset({
        "test", "tests", "testing", "spec", "regression",
        "tester", "testa", "regressionstest",
    }),
    "domain": frozenset({
        "model", "entity", "service", "state", "flow", "rule", "behavior",
        "behaviour", "type", "instance", "template",
        "modell", "entitet", "tjänst", "flöde", "regel", "instans", "mall",
    }),
}

REQUIREMENT_ROLE_SYMBOL_TERMS = {
    "persistence": frozenset({
        "init", "initialize", "schema", "migrate", "migration", "create", "insert",
        "save", "store", "persist", "load", "read", "get", "list", "update",
        "delete", "remove",
    }),
    "transport": frozenset({
        "api", "route", "handler", "request", "response", "get", "post", "put",
        "patch", "delete", "options", "serialize", "deserialize",
    }),
    "ui": frozenset({
        "dialog", "modal", "editor", "form", "component", "render", "view",
        "panel", "button", "submit", "handle", "open", "close",
    }),
    "tests": frozenset({"test", "spec", "fixture", "assert"}),
    "domain": frozenset({
        "model", "entity", "service", "state", "flow", "rule", "create", "update",
        "delete", "place", "move", "transfer",
    }),
}

GENERIC_REQUIREMENT_TERMS = frozenset({
    "add", "allow", "change", "create", "edit", "make", "new", "replace",
    "support", "update", "use", "when", "with", "without", "should", "same",
    "lägg", "ändra", "skapa", "stöd", "använd", "ska", "samma",
})


def surface_tokens(text: str) -> set[str]:
    tokens = {
        token.casefold()
        for token in re.findall(
            r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+",
            text.replace("_", " "),
        )
        if len(token) >= 3
    }
    expanded = set(tokens)
    for token in tokens:
        if token.endswith("ies") and len(token) > 5:
            expanded.add(token[:-3] + "y")
        if token.endswith("ing") and len(token) > 5:
            expanded.add(token[:-3])
        if token.endswith("ed") and len(token) > 4:
            expanded.add(token[:-2])
        if token.endswith("s") and len(token) > 4:
            expanded.add(token[:-1])
    return expanded


def surface_roles(tokens: set[str]) -> set[str]:
    return {
        role
        for role, trigger_terms in REQUIREMENT_ROLE_TERMS.items()
        if tokens & trigger_terms
    }


def task_is_complex(task: str, surfaces: list[str] | None = None) -> bool:
    surfaces = surfaces if surfaces is not None else task_surfaces(task)
    roles = {
        role
        for surface in surfaces
        for role in surface_roles(surface_tokens(surface))
    }
    return len(surfaces) >= 4 or len(task) >= 480 or len(roles) >= 3


def task_surfaces(task: str) -> list[str]:
    """Split broad maintenance tasks into bounded explicit requirements."""
    normalized = re.sub(
        r"(?m)^\s*(?:[-*•]|\d+[.)])\s+",
        "",
        task,
    )
    action_words = (
        r"add|allow|create|delete|display|edit|load|make|persist|remove|render|"
        r"replace|save|show|support|update|use|when|lägg|skapa|ta|visa|ändra|"
        r"spara|ladda|stöd"
    )
    clauses = re.split(
        rf"\s*(?:;|\n+|[.!?]+\s+|,\s+(?=(?:{action_words})\b)|"
        rf"\b(?:and|och)\s+(?=(?:{action_words})\b))\s*",
        normalized,
        flags=re.IGNORECASE,
    )
    return list(
        dict.fromkeys(clause.strip() for clause in clauses if clause.strip())
    )[:MAX_TASK_SURFACES]
