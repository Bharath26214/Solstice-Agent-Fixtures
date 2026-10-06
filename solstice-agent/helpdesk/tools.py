"""Tools the agent can call. Each takes typed arguments from the router's ToolCall.

search_kb returns KB chunks for the LLM to answer from; the lookup tools return
a ready-to-send sentence built from the record, so no LLM call is needed for them.
"""

import json
import os
import time

from . import config, retrieval


class ToolMiss(Exception):
    """The tool can't produce context; str(e) is the user-facing reply."""

    reason = "miss"


class MissingInfo(ToolMiss):
    """The message lacks an identifier the tool needs — ask the user for it."""

    reason = "missing_info"


class NotFound(ToolMiss):
    """The identifier or query matched nothing."""

    reason = "not_found"


def _load_json(name):
    with open(os.path.join(config.DATA_DIR, name)) as f:
        return json.load(f)


def _ticket_text(t):
    return "Ticket %s (\"%s\") is %s, priority %s, opened %s, last updated %s." % (
        t["id"], t["subject"], t["status"].replace("_", " "), t["priority"], t["opened"], t["last_update"])


def _invoice_text(inv):
    return "Invoice %s for workspace %s is %s: $%s, issued %s." % (
        inv["id"], inv["workspace"], inv["status"].replace("_", " "), format(inv["amount_usd"], ","), inv["issued"])


def search_kb(query, docs=None):
    """Search the Solstice help-center knowledge base. Returns ranked chunks
    ({"chunk", "heading", "doc"}) for the LLM; `docs` filters by KB file."""
    chunks = retrieval.search(query, docs=docs)
    if not chunks:
        raise NotFound(
            "I couldn't find that in our help center, so I'd rather not guess. "
            "Our support team can help at help@solstice.app."
        )
    return chunks


def lookup_ticket(ticket_id):
    """Look up a support ticket by its ID (e.g. TKT-1042). Returns a ready-to-send sentence."""
    if not ticket_id:
        raise MissingInfo("Could you share your ticket ID (it looks like TKT-1234)? I'll look it up.")
    time.sleep(0.12)  # ticketing system API call
    for t in _load_json("tickets.json"):
        if t["id"] == ticket_id:
            return _ticket_text(t)
    raise NotFound("I couldn't find ticket %s. Could you double-check the ID?" % ticket_id)


def list_user_tickets(email):
    """List all support tickets filed by a user's email address. Returns ready-to-send text."""
    if not email:
        raise MissingInfo("Could you share the email address the tickets were filed under?")
    results = []
    for t in _load_json("tickets.json"):
        if t["email"].lower() == email:
            # fetch the full ticket record
            results.append(lookup_ticket(t["id"]))
    if not results:
        raise NotFound("I couldn't find any tickets filed under %s." % email)
    return "I found %d tickets for %s:\n" % (len(results), email) + "\n".join("- " + r for r in results)


def invoice_status(invoice_id):
    """Check the status of an invoice by its ID (e.g. INV-2093). Returns a ready-to-send sentence."""
    if not invoice_id:
        raise MissingInfo("Could you share the invoice ID (it looks like INV-2093)? I'll check its status.")
    time.sleep(0.12)  # billing system API call
    for inv in _load_json("invoices.json"):
        if inv["id"] == invoice_id:
            return _invoice_text(inv)
    raise NotFound("I couldn't find invoice %s. Could you double-check the ID?" % invoice_id)


TOOLS = {
    "lookup_ticket": lookup_ticket,
    "invoice_status": invoice_status,
    "list_user_tickets": list_user_tickets,
    "search_kb": search_kb,
}
