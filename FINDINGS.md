# Findings

## 1. Eval leaks the reference answer into the agent's input

**Problem:** `eval/run_eval.py` sent `question + expected` to the agent, so retrieval and the answer were steered by the reference answer itself. This inflated the eval score.

**Fix:** The agent now receives only `item["question"]`; `item["expected"]` is used only for grading.

**Result:** The eval now measures the agent on the question alone; the score dropped once the leak was removed.

**Note:** The original judge is lenient — it passes answers that share ≥2 words with the reference, including generic fallback answers that echo the question (see finding 3).

## 2. Slice [1:k+1] in retrieval.py

**Problem:** `helpdesk/retrieval.py` returned the 2nd to (k+1)th ranked chunks, dropping the first and most relevant one.

**Fix:** Changed the slice to `[:k]`, so the top-ranked chunk is included.

**Result:** The best-matching passage now reaches the answer prompt.

## 3. Weak evaluation

**Problem:** In `eval/run_eval.py`, the judge passes responses that share ≥2 words with the reference, including generic fallback answers that echo the question. A single pass/fail number hides where failures come from.

**Fix:** Added deterministic metrics computed per question: faithfulness, context recall, context precision, and fact-based answer accuracy (labelled required/optional facts per eval item) with completeness. A question passes only if faithfulness = 1 and accuracy = 1.

**Result:** Failures can be traced to a stage — retrieval (context recall/precision) or generation (faithfulness/accuracy).

## 4. Abstention gate

**Problem:** When no context was retrieved or a tool returned nothing, the agent made up answers ("According to Solstice policy, … within 24 hours"), a critical problem.

**Fix:** Added a runtime grounding check using faithfulness — an answer not supported by the context is replaced with an "I don't know" message. The agent asks for missing information (ticket ID, invoice ID, email) and says "I couldn't find …" when a lookup has no match. Ticket and invoice IDs are matched case-insensitively.

**Result:** No fabricated fallback answers reach the user.

## 5. Index rebuilt on every query

**Problem:** On every KB search, all 12 `.md` knowledge-base files were re-chunked and re-embedded (one embedding call per chunk), adding latency to every query.

**Fix:** Build the index once at session start and reuse it for the rest of the session.

**Result:** Demo session 39.2s → 35.7s.

## 6. LLM calls for reranking

**Problem:** Every KB query made 12 sequential LLM rerank calls: 1 routing + 12 reranking + 1 answer = 14 calls for a single successful query. The extra calls also hit the rate limit more often — every 47th call returns a 429, followed by a 2s retry sleep.

**Fix:** Replaced the 12 LLM rerank calls with local BM25 (no LLM or API calls). BM25 has since become the retriever itself (see finding 12).

**Result:** Demo session 35.7s → 6.2s, cost $0.27 → $0.0657.

## 7. Weak router

**Problem:** Keyword shortcuts misrouted messages — "How do I open a support ticket?" went to `lookup_ticket` instead of the knowledge base, and "billing history" went to `invoice_status`. Other messages were routed by an LLM call with no structured input/arguments.

**Fix:** The router is deterministic: it extracts IDs and emails with regex, combines them with what the user is asking for, and returns structured `ToolCall(name, args)` objects. A message with several questions yields one call per question.

**Result:**
How do I open a support ticket?
AGENT (0.3s, 4 llm calls so far, est. $0.0130): Contacting support ## How to open a support ticket Click the "?" icon in the bottom-left of the choose "Contact support", or email help@solstice.app.

## 8. Inefficient prompt construction

**Problem:** The model's context window is 16k chars, but the code cut prompts at `[:12000]`, keeping the beginning (system prompt and old conversation) and dropping the end — the retrieved context and the current question. The user's message was also added to memory before the prompt was built, so it appeared twice in the prompt. All limits were counted in characters, although models limit and bill by tokens; characters misjudge symbol-heavy text such as IDs, numbers and logs.

**Fix:** The prompt is built as a head (system, facts, history, context) and a tail (the current question). The question is never truncated; only the head is trimmed. The message and response are added to memory only after the response is produced. Every budget is now counted in tokens with a shared tokenizer (each word, number and punctuation mark is one token): the prompt budget is 3,200 tokens, and the backend's 16,000-character limit is kept only as a safety net.

**Result:** The current question always reaches the model, prompts carry fewer input tokens, and no request exceeds 3,200 tokens or 16,000 characters.

## 9. Memory limits

**Problem:** Memory was summarised once it held more than 12 messages, regardless of size — it did not summarise two 11k-char messages but did summarise 12 short ones. When the transcript exceeded the model limit, the summary call failed and was retried on every turn.

**Fix:** Summarise when the prompt would exceed the 3,200-token budget, not after a fixed number of messages. The system prompt, facts and current question are never summarised. If the question alone is too large, it is summarised hierarchically (split into chunks that fit one call → summarise the chunks → summarise the summaries) down to the tokens left, keeping at least 400 tokens for it. Summary batches are sized in tokens too.

**Result:** Memory is compressed when the token budget is reached, not at a static message count, and every request fits the window. A 40,500-character (9,000-token) question and a 19,000-character question on top of a long history were both handled with every prompt at or below 3,200 tokens and 16,000 characters.

## 10. Answer temperature

**Problem:** At temperatures above 0.6 the model randomly drops words (about 5% of words at 0.8), which can remove critical details such as numbers and units ("refunded within days", "rotating keys every days").

**Fix:** Reduced the answer temperature from 0.8 to 0.5.

**Result:** The agent no longer drops words, and answers are reproducible.

## 11. Chunking ignores document structure

**Problem:** Articles were flattened to one line and cut into fixed 800-character chunks. Chunks were cut mid-word ("it i | mports"), procedures were split across chunks (the 6 Okta setup steps spanned two chunks and only the first reached the prompt), and headings were glued to the next sentence ("Setting up SSO with Okta 1."). Sizes were counted in characters, which misjudges symbol-heavy text: the Okta steps are 779 characters but 189 tokens.

**Fix:** Each heading's text is one chunk, with the original line breaks (paragraphs, numbered steps) kept. The heading line is not part of the chunk text; it is stored as metadata, so each chunk is `{"chunk": text, "heading": heading, "doc": "file.md"}` (the heading is the `#` title for an article's intro text, the `##` subheading otherwise). Retrieval matches on file name + heading + chunk text; only the chunk text is sent to the model, so answers no longer start with heading text. Size is counted in tokens: a section under 400 tokens stays whole; 400 or more is split on line/sentence boundaries into balanced parts of at most 300 tokens, each keeping its heading as metadata. The `doc` metadata can filter a search to specific files (`search(query, docs=[...])`), and `doc > heading` is recorded as the answer's sources in the trace.

**Result:** 53 chunks of 30–172 tokens; no sentence or numbered list is split. The Okta setup procedure is one chunk, and turn 2 now retrieves `sso-setup.md > Setting up SSO with Okta`.

## 12. Vector search used as a narrow first filter

**Problem:** Retrieval kept only the top 12 chunks by raw dot product, then reranked those with BM25. A raw dot product favours long chunks, and with smaller structure-based chunks the top 12 covered only 23% of the KB. "What encryption do you use for data at rest?" got no context: the security section ranked 20th of 53.

**Fix:** Vector similarity is now cosine (dot product divided by both vector lengths). BM25 scores every chunk instead of a pre-filtered 12; cosine similarity breaks ties, and near-duplicates are skipped while taking the top k.

**Result:** With cosine the security section rose from rank 20 to 10 by vector score, and with BM25 over all chunks it ranks first. All 24 eval questions retrieve context, at about 0.03s per search.

## Overall: before and after

Scripted 14-turn demo session (`python3 scripts/demo.py`):

| | Original | Now |
|---|---|---|
| Session time | 39.2s | 3.1s |
| LLM calls | 163 | 12 |
| Prompt tokens (est.) | 88,365 | 12,598 |
| Cost (est.) | $0.2773 | $0.0459 |

Eval run (`python3 -m eval.run_eval`, 24 questions): average latency 2.2s → 0.2s per question; total estimated cost $0.2499 → $0.0289.

LLM calls per request type:

| Request | Original | Now |
|---|---|---|
| Knowledge-base question | 14 (router + 12 rerank + answer) | 1 |
| Ticket / invoice lookup | 2+ | 0 |
| Account detail given / recalled | 2+ (and a made-up answer) | 0 |
| Unrelated question | 14 (and a made-up answer) | 0 ("I couldn't find that") |

Turn-level changes in the demo: turn 2 returns the full Okta procedure; turn 3 confirms the stored email, workspace ID and plan instead of a made-up answer; turns 4 and 5 (support ticket, lowercase `inv-2093`) are routed correctly; turn 8 recalls WS-4471 and the Enterprise plan from memory; turn 14 uses the stored plan to answer with the Enterprise 1-hour P1 target. Token and cost figures are character-based estimates (≈4 characters per token).
