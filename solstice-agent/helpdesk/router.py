"""Routes a user message to the right tool."""

from .tools import TOOLS

ROUTER_PROMPT = """You are the routing layer for a customer-support agent.

Available tools:
- search_kb: search the Solstice help-center knowledge base for policies, plans, setup guides, and troubleshooting
- lookup_ticket: look up an existing support ticket by ticket ID
- list_user_tickets: list the support tickets filed by an email address
- invoice_status: check the status of an invoice or a recent charge

User message: {message}

Respond with the name of the single best tool."""


def route(llm, message):
    m = message.lower()
    # fast paths — skip the LLM call for obvious cases
    if "ticket" in m:
        return "lookup_ticket"
    if "invoice" in m or "charge" in m or "billing" in m:
        return "invoice_status"

    name = llm.complete(ROUTER_PROMPT.format(message=message)).strip()
    if name not in TOOLS:
        name = list(TOOLS)[0]
    return name
