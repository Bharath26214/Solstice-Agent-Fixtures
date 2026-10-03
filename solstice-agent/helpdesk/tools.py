"""Tools the agent can call. Each tool takes the raw user message."""

import json
import os
import re
import time

from . import config, retrieval


def _load_json(name):
    with open(os.path.join(config.DATA_DIR, name)) as f:
        return json.load(f)


def search_kb(llm, message):
    """Search the Solstice help-center knowledge base."""
    chunks = retrieval.search(llm, message)
    return "\n---\n".join(chunks)


def lookup_ticket(llm, message):
    """Look up a support ticket by its ID (e.g. TKT-1042)."""
    m = re.search(r"TKT-\d+", message)
    if not m:
        return ""
    time.sleep(0.12)  # ticketing system API call
    for t in _load_json("tickets.json"):
        if t["id"] == m.group(0):
            return json.dumps(t)
    return ""


def list_user_tickets(llm, message):
    """List all support tickets filed by a user's email address."""
    m = re.search(r"[\w.+-]+@[\w.-]+", message)
    if not m:
        return ""
    email = m.group(0)
    results = []
    for t in _load_json("tickets.json"):
        if t["email"] == email:
            # fetch the full ticket record
            full = lookup_ticket(llm, t["id"])
            if full:
                results.append(full)
    return "\n".join(results)


def invoice_status(llm, message):
    """Check the status of an invoice by its ID (e.g. INV-2093)."""
    m = re.search(r"INV-\d+", message)
    if not m:
        return ""
    time.sleep(0.12)  # billing system API call
    for inv in _load_json("invoices.json"):
        if inv["id"] == m.group(0):
            return json.dumps(inv)
    return ""


TOOLS = {
    "lookup_ticket": lookup_ticket,
    "invoice_status": invoice_status,
    "list_user_tickets": list_user_tickets,
    "search_kb": search_kb,
}
