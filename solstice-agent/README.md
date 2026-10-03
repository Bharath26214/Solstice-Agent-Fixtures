# Solstice Support Agent

Prototype support agent for the Solstice workspace product. It routes customer
messages to a tool (knowledge-base search, ticket lookup, invoice lookup),
gathers context, and answers with the LLM. Conversation memory is summarized
automatically so long sessions stay within the model's context window.

Runs entirely locally: the `solstice-lm-1` mock backend reproduces the hosted
model's behavior, latency, and cost profile, so no API keys are needed and
runs are deterministic.

## Quickstart

No dependencies — Python 3.9+ standard library only.

```
python scripts/demo.py        # scripted 14-turn customer session (writes traces/)
python -m eval.run_eval       # accuracy eval over data/eval_set.json
```

Interactive:

```python
from helpdesk.agent import Agent
agent = Agent()
print(agent.handle("How do I reset my password?"))
```

## Layout

```
helpdesk/
  agent.py       orchestration: route -> tool -> answer
  router.py      picks a tool for each message
  tools.py       search_kb, lookup_ticket, list_user_tickets, invoice_status
  retrieval.py   embedding recall + LLM rerank over data/kb/
  chunking.py    document chunking
  embeddings.py  embedding client
  memory.py      conversation history + auto-summarization
  llm.py         LLM client (mock backend, retries, usage accounting)
data/
  kb/            help-center articles (the knowledge base)
  tickets.json   support tickets fixture
  invoices.json  invoices fixture
  eval_set.json  eval questions with reference answers
eval/run_eval.py accuracy eval
scripts/demo.py  scripted customer session
```

## Status

Eval accuracy is in decent shape (see `eval/run_eval.py`). Known rough edges:
long chat sessions get slow, and customers occasionally report answers that
don't match our docs — haven't had time to dig in.
