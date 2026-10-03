# AI Engineer assessment — Solstice Support Agent

A reverse-engineering challenge for AI Engineer candidates. You get a small,
runnable agentic AI system; your job is to figure out how it works, what's
wrong with it, and how you'd fix it.

## The scenario

The **Solstice Support Agent** (in [`solstice-agent/`](solstice-agent/)) is a
prototype customer-support agent for a SaaS workspace product. The engineer
who built it has left the company. The internal eval reports high accuracy —
but customers complain about wrong answers, and long chat sessions get slow
and expensive. Figure out what's really going on.

## Getting started

The repo is self-contained: Python 3.9+, standard library only, no API keys,
no network. A deterministic mock LLM backend reproduces the hosted model's
behavior, latency, and cost profile — timings, token counts, and costs you
measure locally are representative, and runs are reproducible.

```bash
cd solstice-agent
python scripts/demo.py     # scripted 14-turn customer session (a saved transcript is in traces/)
python -m eval.run_eval    # the team's accuracy eval
```

Both finish in well under a minute. You can also drive the agent directly:

```python
from helpdesk.agent import Agent
agent = Agent()
print(agent.handle("How do I reset my password?"))
```

## Your task

After you submit, you'll give a **~20-minute presentation followed by ~20
minutes of technical Q&A** with our engineers.

The deliverable is a presentation (roughly 10–15 slides, any format) with four
sections:

1. **How the system works** — architecture, the life of a request, and what
   the eval actually measures.
2. **The biggest performance problems** — latency and cost, ranked by impact,
   with measurements where you can get them.
3. **The biggest accuracy problems** — why users get wrong answers, and why
   the eval doesn't show it. Rank by impact, with evidence.
4. **Proposed fixes** — what you'd change, in what order, and the improvement
   you'd expect. Pseudocode is welcome; working fixes with before/after
   numbers are a plus, but the presentation is what's assessed.

## Ground rules

- You may use any tools, including AI assistants — but be ready to defend
  every claim without them. "The AI said so" is not an accepted answer in
  Q&A, and unverified claims score worse than an honest "I didn't get to
  this."
- Evidence beats volume: a short list of confirmed, measured findings ranked
  by impact scores higher than a long list of unverified observations.
- The mock backend (`helpdesk/llm.py` bottom half, `helpdesk/embeddings.py`)
  stands in for the hosted *model* — it is not the system under review, and
  nothing should need fixing there.

Good luck.
