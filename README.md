# Solstice Support Agent

Customer-support agent for the Solstice workspace product. It splits a customer
message into its questions, routes each one deterministically to a tool
(knowledge-base search, ticket lookup, ticket listing, invoice status) with
structured arguments, answers knowledge-base questions with one LLM call, and
refuses to send an answer that is not supported by the retrieved context.
Account details the customer gives (email, workspace ID, plan, ticket and
invoice IDs) are kept as structured facts for the rest of the session.

Runs entirely locally: the `solstice-lm-1` mock backend reproduces the hosted
model's behavior, latency, and cost profile, so no API keys are needed and runs
are deterministic.

## Quickstart

No dependencies — Python 3.9+ standard library only. Run from the `solstice-agent/` directory.

```bash
cd solstice-agent
python3 scripts/demo.py                          # scripted 14-turn session (writes traces/demo_session.txt)
python3 scripts/demo.py "I am on enterprise plan." "Which plan am I on?"   # your own messages, one session
python3 -m eval.run_eval                         # eval: per-question metrics, writes eval/metrics.json
python3 -m unittest discover -s tests -v         # agent scenario tests (real mock model)
```

```python
from helpdesk.agent import Agent
agent = Agent()
print(agent.handle("How do I reset my password?"))
print(agent.last_trace)   # tools, arguments, sources, contexts, abstention reason
```

## How a request is handled

```
message
  │
  ├─ store IDs / email mentioned (ticket, invoice, workspace, email)      no LLM
  ├─ router: split into sentences → one ToolCall(name, args) per question   no LLM
  │
  ├─ remember / recall ─────────── answered from session facts             no LLM
  ├─ lookup_ticket / invoice_status / list_user_tickets
  │                     ────────── record rendered as a sentence           no LLM
  └─ search_kb
        ├─ BM25 over every chunk (cosine tie-break), keyword relevance floor
        ├─ best-matching section only, else top-k chunks
        ├─ prompt: system → facts → history → context → question (token budget)
        ├─ one LLM call
        └─ grounding gate → answer, or the best section's text, or "I don't know"
  │
  └─ replies joined in order; message + reply stored in memory
```

## Features

- **Deterministic router** — regex entities + intent → one `ToolCall(name, args)` per question; several questions per message.
- **Tools** — ticket/invoice lookups and ticket listing return ready-to-send sentences; KB search returns chunks with `heading`/`doc` metadata.
- **Retrieval** — structure-aware, token-sized chunks; BM25 over every chunk with cosine tie-break; optional filter by source file.
- **Grounded answers** — best-matching section as context, one LLM call, grounding gate with best-section fallback or "I don't know".
- **Memory** — structured facts (email, workspace, plan, ticket/invoice IDs) with recall and reuse; size-based, hierarchical summarization.
- **Eval and tests** — per-stage metrics (faithfulness, context recall/precision, fact-based accuracy, completeness); 25 scenario tests on the real mock model.

## Improvements by design principle

### 1. Fewer model calls

- **Reranking moved off the LLM.** Every KB question made 12 sequential LLM
  calls to score passages 0–10. Ranking is now local BM25 over all chunks
  (cosine similarity breaks ties): 12 calls → 0.
- **Routing moved off the LLM.** Messages that missed the keyword shortcuts cost
  an LLM routing call. The router is now regex entities + intent rules: 1 call → 0.
- **Answers that need no generation skip the model.** Ticket and invoice
  records are rendered as sentences by the tool; account details are stored and
  recalled from memory; "missing ID" and "not found" replies come from the
  tools. All of these are 0 calls.
- **Result:** a KB question went from 14 calls to 1; the demo session from 163
  calls to 12.

### 2. Latency

- **Index built once.** Every KB search re-chunked and re-embedded the whole
  KB (one embedding call per chunk). The index is now built once per process
  and reused: demo 39.2s → 35.7s on its own.
- **No sequential rerank calls.** Removing the 12 rerank calls per KB question
  was the largest single gain: 35.7s → 6.2s at the time.
- **No retry storms.** Oversized prompts used to be rejected and retried 5
  times with 2s sleeps (~10s per turn, every turn once the transcript was
  large). Prompts are now sized to fit before they are sent.
- **Result:** demo session 39.2s → 3.1s; eval 2.2s → 0.2s per question.

### 3. Cost

- **Fewer calls** (above) is most of the saving.
- **Smaller prompts.** The user's message was in the prompt twice (history and
  question) — it is now stored only after the reply. When one section clearly
  matches, only that section is sent instead of four chunks. Raw JSON records
  are no longer sent to the model at all.
- **Result:** demo session $0.2773 → $0.0459 and 88,365 → 12,598 prompt
  tokens; eval $0.2499 → $0.0289.

### 4. Context-window management

- **The question is never cut.** Prompts were cut at `[:12000]` characters,
  keeping the system prompt and old history and dropping the retrieved context
  and the question — so long sessions answered "I don't know" to everything.
  The prompt is now head (system, facts, history, context) + tail (question);
  only the head is trimmed.
- **Summarize by size, not message count.** History was summarized after 12
  messages regardless of size, and the summary call itself failed once the
  transcript passed the model limit. Summarization now triggers when the
  prompt would exceed the budget, works in batches that each fit, and never
  touches the system prompt, facts or question.
- **Oversized questions are summarized hierarchically** (chunks → summaries →
  summary of summaries) instead of being rejected.
- **Budgets in tokens.** Limits were in characters; they are now in tokens
  (3,200-token budget) with the backend's 16,000-character limit as a safety
  net. A 40,500-character question and a long-history session both stayed
  within budget on every call.

### 5. Correctness (small bugs, big effects)

- **Best passage was dropped.** `retrieval.py` returned ranks 2…k+1, so the most
  relevant chunk never reached the prompt. Fixed to `[:k]`.
- **Eval leaked the answer.** The eval sent `question + reference` to the agent,
  steering retrieval and the answer with the expected text. It now sends only
  the question.
- **Misrouting.** "How do I open a support ticket?" went to `lookup_ticket`
  (no ID → empty context → made-up answer); "billing history" went to
  `invoice_status`; lowercase `inv-2093` didn't match. All fixed by the new router.
- **Dropped words.** At answer temperature 0.8 the model dropped ~5% of words,
  including key numbers ("refunded within days", "every days"). Now 0.5.
- **Broken chunks.** Fixed 800-character chunks cut words and procedures in
  half ("it i | mports", Okta steps across two chunks) and glued headings onto
  text ("Setting up SSO with Okta 1."). Chunks now follow headings, keep line
  breaks and are sized in tokens.
- **Narrow vector shortlist.** A raw dot-product top-12 favoured long chunks and
  missed the security section (rank 20 of 53) for an encryption question. BM25
  now scores every chunk; the section ranks first.

### 6. No fabricated answers

- With empty or irrelevant context the model produced "According to Solstice
  policy, X is supported on all plans … within 24 hours". Every answer now
  passes a grounding check (each sentence's words and numbers must be in the
  context); otherwise the best section's text is used, or the agent says it
  doesn't know. Missing IDs are asked for; misses say "I couldn't find …".
- A grounded answer can still come from the wrong chunk (the escalation
  sentence for "open a support ticket"); sending only the best-matching
  section when one clearly matches prevents this.

### 7. Memory that keeps what matters

- Account details given in turn 3 were lost by turn 8: the one-sentence summary
  kept only the first user message, and the model never read the history. They
  are now structured facts — stored on sight, never summarized, recalled
  directly ("Your workspace ID is WS-4471."), and reused for later requests
  ("list my tickets", "status of my invoice", "given our plan…").

### 8. Determinism, observability, testability

- Regex routing, BM25 retrieval, temperature 0 for judging and 0.5 for answers:
  the same input gives the same output, so eval runs are reproducible.
- `agent.last_trace` records each step's tool, arguments, sources, contexts and
  abstention reason, and the eval reports metrics per stage, so a wrong answer
  can be traced to routing, retrieval or generation.
- 25 scenario tests run on the real mock model and print each exchange's
  latency, LLM calls and cost, plus a run total.

## Performance

Scripted 14-turn demo session (`python3 scripts/demo.py`):

| | Original | Now |
|---|---|---|
| Session time | 39.2s | 3.1s |
| LLM calls | 163 | 12 |
| Prompt tokens (est.) | 88,365 | 12,598 |
| Cost (est.) | $0.2773 | $0.0459 |

Eval run (`python3 -m eval.run_eval`, 24 questions): average latency 2.2s → 0.2s
per question; total estimated cost $0.2499 → $0.0289.

LLM calls per request type:

| Request | Original | Now |
|---|---|---|
| Knowledge-base question | 14 (router + 12 rerank + answer) | 1 |
| Ticket / invoice lookup | 2+ | 0 |
| Account detail given / recalled | 2+ (and a made-up answer) | 0 |
| Unrelated question | 14 (and a made-up answer) | 0 ("I couldn't find that") |

Token and cost figures are the client's estimates (characters ÷ 4) so they stay
comparable with the original run; the original transcript is in git
(`git show HEAD:solstice-agent/traces/demo_session.txt`).

## Configuration (`helpdesk/config.py`)

| Setting | Value | Meaning |
|---|---|---|
| `CHUNK_SIZE` / `CHUNK_MAX_TOKENS` | 300 / 400 | sections under 400 tokens stay whole; larger ones split into ≤ 300-token parts |
| `TOP_K` | 4 | chunks retrieved per KB question |
| `MIN_MATCHED_TERMS` | 2 | keywords a chunk must share with the question |
| `BM25_K1` / `BM25_B` | 1.5 / 0.75 | BM25 parameters |
| `GROUNDING_THRESHOLD` | 1.0 | every answer sentence must be supported by the context |
| `MODEL_CONTEXT_TOKENS` | 3,200 | prompt budget; history is summarized above it |
| `MIN_QUESTION_TOKENS` | 400 | room kept for a summarized oversized question |
| `MODEL_CONTEXT_CHARS` | 16,000 | backend's hard limit, safety net only |

## Layout

```
solstice-agent/
  helpdesk/
    agent.py       orchestration: route -> tools -> answer -> grounding gate; facts and recall
    router.py      deterministic router: entities + intent -> ToolCall(name, args)
    tools.py       search_kb, lookup_ticket, list_user_tickets, invoice_status
    retrieval.py   BM25 over all chunks, cosine tie-break, metadata filter
    chunking.py    structure-aware, token-sized chunks with heading/doc metadata
    tokenizer.py   token counting / truncation and the prompt budget check
    grounding.py   term matching, faithfulness check, answer focus
    memory.py      facts + history, size-based and hierarchical summarization
    embeddings.py  embedding client (mock backend)
    llm.py         LLM client (mock backend, retries, usage accounting)
    config.py      settings
  data/
    kb/            help-center articles (the knowledge base)
    tickets.json   support tickets fixture
    invoices.json  invoices fixture
    eval_set.json  eval questions, reference answers, required/optional facts
  eval/
    run_eval.py    eval runner (prints per-question metrics, writes metrics.json)
    metrics.py     faithfulness, context recall/precision, accuracy, completeness
  scripts/demo.py  scripted or custom customer session
  tests/           agent scenario tests against the real mock model
```
