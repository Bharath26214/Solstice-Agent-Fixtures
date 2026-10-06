"""Deterministic router: extract entities, combine with intent, emit structured tool calls.

No LLM call. A message is split into sentences and each sentence is routed on its own,
so "Status of INV-2093? And how do I reset my password?" yields two calls. Per sentence,
the first matching rule wins:

  1. ticket IDs (TKT-1234)                      -> lookup_ticket(ticket_id), one per ID
  2. invoice IDs (INV-2093)                     -> invoice_status(invoice_id), one per ID
  3. asks for a stored fact ("which plan am I on?", "what was my workspace ID?")
                                                -> recall(key), answered from session memory
  4. how-to question ("how do I open a ticket") -> search_kb(query)
  5. ticket mention + email                     -> list_user_tickets(email)
  6. "my tickets" / "list tickets", no email   -> list_user_tickets(email=None)  -> asks for email
  7. ticket status/lookup, no ID               -> lookup_ticket(ticket_id=None)  -> asks for ID
  8. invoice/charge status, no ID              -> invoice_status(invoice_id=None) -> asks for ID
  9. any other question                         -> search_kb(query)
 10. a statement ("our app is stuck syncing.") -> no call; carried as context into the next KB query

A statement that gives account facts (email, workspace ID, plan) also yields
remember(facts): the agent stores them in session memory and confirms — no KB search.

Known facts (session memory + this message) fill in missing arguments: a stored email
for "list my tickets", a stored ticket/invoice ID for "any update on my ticket?", and the
stored plan is added to KB queries that say "our plan" / "my plan".
If nothing routes (e.g. only statements or a pasted log), the whole message goes to search_kb.
"""

import re
from dataclasses import dataclass, field

from .grounding import terms

# entities
TICKET_ID = re.compile(r"\bTKT-\d+\b", re.I)
INVOICE_ID = re.compile(r"\bINV-\d+\b", re.I)
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
WORKSPACE_ID = re.compile(r"\bWS-\d+\b", re.I)
PLAN = re.compile(
    r"\b(?:we're|we are|i'm|i am|currently)\s+(?:on|using)\s+(?:the\s+)?(free|starter|pro|enterprise)\b"
    r"|\b(?:our|my)\s+plan\s+is\s+(?:the\s+)?(free|starter|pro|enterprise)\b",
    re.I,
)

# topics
TICKET = re.compile(r"\btickets?\b", re.I)
INVOICE = re.compile(r"\b(invoices?|charged?|charges|billed|payments?)\b", re.I)

# intents
HOW_TO = re.compile(
    r"\bhow (do|can|should|would) (i|we|you)\b|\bhow to\b|\bwhere (do|can) (i|we)\b"
    r"|\b(open|create|submit|raise|file|log)\s+(a\s+|an\s+|new\s+)*(support\s+)?ticket",
    re.I,
)
LIST = re.compile(r"\b(list|show|all)\b.*\btickets\b|\bmy tickets\b", re.I)
STATUS = re.compile(
    r"\b(status|check|look ?up|update|progress|paid|due|refunded|happened)\b|\bmy (ticket|invoice)\b",
    re.I,
)
PAST_CHARGE = re.compile(r"\b(was|were|been|got) (i|we|my \w+) (charged|billed)\b", re.I)
OUR_PLAN = re.compile(r"\b(my|our)\s+plan\b", re.I)
RECALL = {
    "plan": re.compile(
        r"\b(which|what)\s+plan\s+(am|are)\s+(i|we)\b|\b(which|what)\s+plan\s+(i|we)\s+(am|are)\b"
        r"|\b(which|what)('s|\s+is|\s+was)\s+(my|our)\s+(current\s+)?plan\b",
        re.I,
    ),
    "workspace_id": re.compile(r"\b(my|our)\s+workspace\s+id\b", re.I),
    "email": re.compile(r"\bwhat('s|\s+is|\s+was)\s+(my|our)\s+email\b", re.I),
    "invoice_id": re.compile(r"\b(my|our|the)\s+invoice\s+(id|number)\b", re.I),
    "ticket_id": re.compile(r"\b(my|our|the)\s+ticket\s+(id|number)\b", re.I),
}
QUESTION = re.compile(
    r"\?\s*$|^(what|how|why|when|where|which|who|is|are|can|could|do|does|did|will|would|should"
    r"|any|show|list|check|tell|give|find)\b",
    re.I,
)


@dataclass
class ToolCall:
    name: str
    args: dict = field(default_factory=dict)
    question: str = field(default="", compare=False)  # the sentence this call answers


def split_sentences(message):
    return [p.strip() for p in re.split(r"(?<=[?!.])\s+|\n+", message.strip()) if p.strip()]


def extract_facts(s):
    """Account facts the user states about themselves (email, workspace ID, plan)."""
    facts = {}
    m = EMAIL.search(s)
    if m:
        facts["email"] = m.group(0).lower()
    m = WORKSPACE_ID.search(s)
    if m:
        facts["workspace_id"] = m.group(0).upper()
    m = PLAN.search(s)
    if m:
        facts["plan"] = (m.group(1) or m.group(2)).capitalize()
    return facts


def artifacts(message):
    """IDs and contact details mentioned anywhere in a message (the last one of each wins)."""
    found = {}
    for key, pattern, norm in (
        ("ticket_id", TICKET_ID, str.upper),
        ("invoice_id", INVOICE_ID, str.upper),
        ("workspace_id", WORKSPACE_ID, str.upper),
        ("email", EMAIL, str.lower),
    ):
        matches = pattern.findall(message)
        if matches:
            found[key] = norm(matches[-1])
    return found


def _kb(s, known):
    """search_kb call; "our plan" / "my plan" is resolved to the stored plan."""
    q = s
    if known.get("plan") and OUR_PLAN.search(s):
        q = "%s (%s plan)" % (s, known["plan"])
    return ToolCall("search_kb", {"query": q}, q)


def _route_sentence(s, known):
    ids = [m.upper() for m in TICKET_ID.findall(s)]
    invs = [m.upper() for m in INVOICE_ID.findall(s)]
    if ids or invs:
        return ([ToolCall("lookup_ticket", {"ticket_id": i}, s) for i in ids]
                + [ToolCall("invoice_status", {"invoice_id": i}, s) for i in invs])

    if QUESTION.search(s):
        keys = [k for k, pattern in RECALL.items() if pattern.search(s)]
        if keys:
            return [ToolCall("recall", {"key": k}, s) for k in keys]

    if HOW_TO.search(s):
        return [_kb(s, known)]

    if TICKET.search(s):
        if known.get("email") and (LIST.search(s) or EMAIL.search(s)):
            return [ToolCall("list_user_tickets", {"email": known["email"]}, s)]
        if LIST.search(s):
            return [ToolCall("list_user_tickets", {"email": None}, s)]
        if STATUS.search(s):
            return [ToolCall("lookup_ticket", {"ticket_id": known.get("ticket_id")}, s)]

    if INVOICE.search(s) and (STATUS.search(s) or PAST_CHARGE.search(s)):
        return [ToolCall("invoice_status", {"invoice_id": known.get("invoice_id")}, s)]

    if QUESTION.search(s):
        return [_kb(s, known)]
    return []  # a statement, not a request


def route(message, facts=None):
    """Return the list of ToolCalls needed to answer every question in the message.

    `facts` is what session memory already knows; facts stated earlier in this
    message are added as the sentences are read.
    """
    known = dict(facts or {})
    known.update(artifacts(message))

    calls, pending = [], []  # pending: statements waiting to give context to the next KB query
    for s in split_sentences(message):
        facts = {} if QUESTION.search(s) else extract_facts(s)
        if facts:
            calls.append(ToolCall("remember", facts, s))
            known.update(facts)
        routed = _route_sentence(s, known)
        if not routed:
            if not facts:  # a fact-only sentence is fully handled by remember
                pending.append(s)
            continue
        for call in routed:
            if call.name == "search_kb" and pending:
                call = _kb(" ".join(pending + [s]), known)
            if call not in calls:
                calls.append(call)
        pending = []

    if not calls:
        return [ToolCall("search_kb", {"query": message}, message)]
    trailing = " ".join(pending)
    if len(terms(trailing)) >= 2:  # a trailing request without a question mark ("Also sync is broken.")
        calls.append(ToolCall("search_kb", {"query": trailing}, trailing))
    return calls
