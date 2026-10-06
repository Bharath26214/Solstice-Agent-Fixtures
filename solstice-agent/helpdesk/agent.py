"""The Solstice support agent: route -> gather context -> answer -> grounding gate."""

import json

from . import config, retrieval
from .grounding import faithfulness, focus, terms
from .llm import LLMClient, LLMError
from .memory import Memory, summarize_message
from .router import artifacts, route
from .tokenizer import count_tokens, fits, truncate_tokens
from .tools import TOOLS, ToolMiss

SYSTEM = (
    "You are the Solstice support agent. Answer only the customer's exact question, "
    "using only the context provided. Do not mention other plans, features, or topics "
    "they did not ask about. If the context does not contain the answer, say you don't know. "
    "Be concise and friendly."
)

UNGROUNDED = (
    "I don't know the answer to that from our help center, so I'd rather not guess. "
    "Our support team can help at help@solstice.app."
)

FACT_LABELS = {
    "email": "email", "workspace_id": "workspace ID", "plan": "plan",
    "ticket_id": "ticket ID", "invoice_id": "invoice ID",
}
RECALL_REPLY = {
    "plan": "You're on the %s plan.",
    "workspace_id": "Your workspace ID is %s.",
    "email": "Your email is %s.",
    "invoice_id": "The invoice you mentioned is %s.",
    "ticket_id": "The ticket you mentioned is %s.",
}


def _matches(question, chunk):
    """Does the chunk's heading + text share at least MIN_MATCHED_TERMS keywords with the question?"""
    return len(terms(question) & terms(chunk["heading"] + "\n" + chunk["chunk"])) >= config.MIN_MATCHED_TERMS


def _chunk_answer(question, chunks):
    """Answer from whole chunks: the content of every chunk whose heading + text matches
    the question. Headings are matched but never included in the answer."""
    return "\n\n".join(c["chunk"] for c in chunks if _matches(question, c))


def _memory_updated(facts):
    noted = ", ".join("%s %s" % (FACT_LABELS[k], v) for k, v in facts.items())
    return "Memory updated: %s." % noted


class Agent:
    def __init__(self, llm=None):
        self.llm = llm or LLMClient()
        self.memory = Memory()
        retrieval.get_index()  # chunk + embed the KB once at session start, not per query
        self.last_trace = None  # query / tool / calls / contexts / response / abstained of the last turn

    def handle(self, message):
        # IDs / email mentioned anywhere are kept for later turns ("any update on my ticket?")
        self.memory.remember(artifacts(message))
        # deterministic routing: one structured ToolCall per question in the message
        steps = [self._run(call) for call in route(message, self.memory.facts)]
        answer = "\n\n".join(step["response"] for step in steps)

        self.last_trace = {
            "query": message,
            "tool": "+".join(step["tool"] for step in steps),
            "calls": steps,
            "contexts": [c for step in steps for c in step["contexts"]],
            "response": answer,
            # None if at least one question was answered, else why none were
            "abstained": None if any(step["abstained"] is None for step in steps) else steps[0]["abstained"],
        }

        # store the turn only now, so this message is in history for later turns
        # and not duplicated in its own prompt (it is already sent as the Question)
        self.memory.add("user", message)
        self.memory.add("assistant", answer)
        return answer

    def _run(self, call):
        """Execute one ToolCall and turn its output into a reply for that question."""
        step = {"tool": call.name, "args": call.args, "question": call.question}
        if call.name == "remember":
            # user stated account facts: store them and confirm. No KB search, no LLM call.
            self.memory.remember(call.args)
            return dict(step, contexts=[], response=_memory_updated(call.args), abstained=None)
        if call.name == "recall":
            # user asked for something they told us earlier: answer from memory, no LLM call
            key = call.args["key"]
            value = self.memory.facts.get(key)
            if not value:
                return dict(step, contexts=[], abstained="missing_info",
                            response="I don't have your %s yet. Could you share it?" % FACT_LABELS[key])
            fact = "%s: %s" % (key, value)
            return dict(step, contexts=[fact], response=RECALL_REPLY[key] % value, abstained=None)
        try:
            result = TOOLS[call.name](**call.args)
        except ToolMiss as e:
            # missing ID -> ask for it; nothing found -> say so. No LLM call either way.
            return dict(step, contexts=[], response=str(e), abstained=e.reason)

        if isinstance(result, list):
            # KB chunks: answer from the chunk content only (headings are metadata), then gate on grounding
            step["sources"] = ["%s > %s" % (c["doc"], c["heading"]) for c in result]
            # when the top-ranked section matches the question, answer from it alone, so a
            # stray sentence from a lower-ranked chunk can't become the answer
            best = result[0] if _matches(call.question, result[0]) else None
            contexts = [best["chunk"]] if best else [c["chunk"] for c in result]
            answer, abstained = self._answer(call.question, contexts)
            if abstained == "ungrounded":
                # no grounded sentence-level answer (e.g. a procedure whose steps don't repeat
                # the question's words): answer with the matching section's content instead
                fallback = best["chunk"] if best else _chunk_answer(call.question, result)
                if fallback:
                    answer, abstained = fallback, None
                    step["answer_from"] = "best_chunk" if best else "chunks"
        else:
            # lookup record already rendered as a sentence: send it as is, no LLM call
            answer, abstained, contexts = result, None, [result] if result else []

        if not answer.strip():
            answer, abstained = "I couldn't find it.", "not_found"
        return dict(step, contexts=contexts, response=answer, abstained=abstained)

    def _prompt(self, message, contexts):
        # layout: system -> facts known about the user -> previous turns -> retrieved context -> question.
        # The question goes last: the model reads it right before answering, and the
        # backend locates the question between "Question:" and "Answer:".
        head = (
            SYSTEM
            + "\n\nFacts: " + json.dumps(self.memory.facts)
            + "\n\nPrevious messages and responses:\n" + self.memory.render()
            + "\n\nContext:\n" + "\n---\n".join(contexts)
        )
        tail = "\n\nQuestion: " + message + "\nAnswer:"
        return head, tail

    def _answer(self, message, contexts):
        head, tail = self._prompt(message, contexts)

        # 1. over the context window (in tokens): summarize previous turns first; system,
        #    facts, context and the current question are kept as they are
        if not fits(head + tail) and self.memory.messages:
            self.memory.compress(self.llm)
            head, tail = self._prompt(message, contexts)

        # 2. still over (the question itself is too large): summarize the question
        #    hierarchically down to the tokens left after system, facts, history and context
        question = message
        if not fits(head + tail):
            overhead = count_tokens(tail) - count_tokens(message)
            room = max(config.MIN_QUESTION_TOKENS, config.MODEL_CONTEXT_TOKENS - count_tokens(head) - overhead)
            try:
                question = summarize_message(self.llm, message, room)
            except LLMError:
                return "Sorry — something went wrong on our end. Please try again.", "llm_error"
            head, tail = self._prompt(question, contexts)

        # last-resort guard: trim the end of the head so the request always fits the window
        head = truncate_tokens(head, config.MODEL_CONTEXT_TOKENS - count_tokens(tail))
        prompt = head[: config.MODEL_CONTEXT_CHARS - len(tail)] + tail

        try:
            answer = self.llm.complete(prompt, temperature=0.5)
        except LLMError:
            return "Sorry — something went wrong on our end. Please try again.", "llm_error"

        # keep only what answers the question, then release it only if it is grounded
        answer = focus(message, answer)
        if faithfulness(contexts, answer) < config.GROUNDING_THRESHOLD:
            return UNGROUNDED, "ungrounded"
        return answer, None
