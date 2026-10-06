"""Agent scenario tests against the real mock model (solstice-lm-1).

Each test drives Agent.handle() like a customer session and prints every
question and reply. RecordingLLM is the real client; it only records prompts.
"""

import sys
import time
import unittest

from helpers import RecordingLLM
from helpdesk.agent import Agent
from helpdesk.tokenizer import fits


def _short(text, limit=160):
    text = text.replace("\n", " | ")
    return text if len(text) <= limit else text[:limit] + " ... [%d chars]" % len(text)


# run-wide totals, printed once after all tests (tearDownModule)
TOTALS = {"messages": 0, "seconds": 0.0, "calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}


def ask(agent, message):
    """Send one message; print the query, the reply, and its latency / LLM calls / cost."""
    before = agent.llm.usage()
    t0 = time.time()
    reply = agent.handle(message)
    seconds = time.time() - t0
    after = agent.llm.usage()
    used = {k: after[k] - before[k] for k in ("calls", "prompt_tokens", "completion_tokens", "cost_usd")}

    TOTALS["messages"] += 1
    TOTALS["seconds"] += seconds
    for k, v in used.items():
        TOTALS[k] += v

    # stderr, so each exchange appears under its test name in `-v` output
    print("\n    Q: " + _short(message), file=sys.stderr)
    print("    A: " + _short(reply), file=sys.stderr)
    print("       [%.2fs | %d LLM calls | %d prompt + %d completion tokens | $%.4f]"
          % (seconds, used["calls"], used["prompt_tokens"], used["completion_tokens"], used["cost_usd"]),
          file=sys.stderr)
    return reply


def tearDownModule():
    t = TOTALS
    print("\n" + "=" * 72, file=sys.stderr)
    print("TOTAL: %d messages | %.2fs agent time | %d LLM calls | %d prompt + %d completion tokens | $%.4f"
          % (t["messages"], t["seconds"], t["calls"], t["prompt_tokens"], t["completion_tokens"], t["cost_usd"]),
          file=sys.stderr)
    if t["messages"]:
        print("AVG per message: %.3fs | %.2f LLM calls | $%.5f"
              % (t["seconds"] / t["messages"], t["calls"] / t["messages"], t["cost_usd"] / t["messages"]),
              file=sys.stderr)
    print("(tokens and cost are the client's estimates: characters / 4)", file=sys.stderr)


class AgentScenario(unittest.TestCase):
    def setUp(self):
        self.llm = RecordingLLM()
        self.agent = Agent(llm=self.llm)

    def say(self, message):
        return ask(self.agent, message)

    @property
    def trace(self):
        return self.agent.last_trace


class TestLookups(AgentScenario):
    def test_ticket_status(self):
        reply = self.say("What is the status of TKT-1001?")
        self.assertTrue(reply.startswith('Ticket TKT-1001 ("Page version history missing") is open'))
        self.assertNotIn("{", reply)
        self.assertEqual(self.llm.calls, 0)

    def test_lowercase_invoice_id(self):
        reply = self.say("Can you check the status of invoice inv-2093?")
        self.assertIn("Invoice INV-2093", reply)
        self.assertIn("past due", reply)

    def test_list_tickets_for_email(self):
        self.assertTrue(self.say("Show all tickets for dana@corp.example").startswith("I found 4 tickets for dana@corp.example"))

    def test_unknown_ids_and_email(self):
        self.assertIn("I couldn't find ticket TKT-9999", self.say("What is the status of TKT-9999?"))
        self.assertIn("I couldn't find invoice INV-9999", self.say("Check INV-9999 please"))
        self.assertIn("I couldn't find any tickets filed under nobody@example.com",
                      self.say("Show all tickets for nobody@example.com"))
        self.assertEqual(self.llm.calls, 0)


class TestMissingInformation(AgentScenario):
    def test_asks_for_ticket_id(self):
        self.assertIn("ticket ID", self.say("Any update on my ticket?"))

    def test_asks_for_invoice_id(self):
        self.assertIn("invoice ID", self.say("Was I charged twice this month?"))

    def test_asks_for_email(self):
        self.assertIn("email address", self.say("List my tickets"))


class TestKnowledgeBase(AgentScenario):
    def test_fact_question(self):
        reply = self.say("How long is a password reset link valid?")
        self.assertIn("24 hours", reply)
        self.assertEqual(self.llm.calls, 1)

    def test_key_numbers_are_not_dropped(self):
        self.assertIn("14 days", self.say("How long do I have to request a refund on a monthly subscription?"))
        self.assertIn("90 days", self.say("How often should API keys be rotated?"))
        self.assertIn("48 hours", self.say("How long is a workspace export download link valid?"))

    def test_jira_is_one_way(self):
        self.assertIn("one-way", self.say("Is the Jira integration two-way?"))

    def test_encryption_question_finds_security_section(self):
        self.assertIn("AES-256", self.say("What encryption do you use for data at rest?"))

    def test_okta_returns_all_steps_not_the_heading(self):
        reply = self.say("How do I set up SSO with Okta?")
        self.assertIn("1. In Solstice, go to Settings", reply)
        self.assertIn("6. Once verified", reply)
        self.assertNotIn("Setting up SSO with Okta", reply)
        self.assertEqual(self.trace["calls"][0]["answer_from"], "best_chunk")

    def test_support_ticket_answer_comes_from_the_right_section(self):
        reply = self.say("How do I open a support ticket?")
        self.assertEqual(self.trace["tool"], "search_kb")
        self.assertIn("Contact support", reply)
        self.assertNotIn("escalate", reply)

    def test_billing_history_goes_to_kb(self):
        self.assertIn("Invoice history", self.say("Where do I find my billing history?"))
        self.assertEqual(self.trace["tool"], "search_kb")


class TestNoMadeUpAnswers(AgentScenario):
    def test_unrelated_questions_are_not_answered(self):
        for q in ("Can I pay with Bitcoin?", "Do you offer a student discount?"):
            reply = self.say(q)
            self.assertIn("I couldn't find that", reply)
            self.assertNotIn("According to Solstice policy", reply)
        self.assertEqual(self.llm.calls, 0)


class TestRouting(AgentScenario):
    def test_several_questions_get_one_answer_each(self):
        reply = self.say("What is the status of INV-2093? How long is a password reset link valid? Any update on my ticket?")
        self.assertEqual(self.trace["tool"], "invoice_status+search_kb+lookup_ticket")
        parts = reply.split("\n\n")
        self.assertEqual(len(parts), 3)
        self.assertIn("INV-2093", parts[0])
        self.assertIn("24 hours", parts[1])
        self.assertIn("ticket ID", parts[2])


class TestMemory(AgentScenario):
    def test_remember_and_recall_plan(self):
        reply = self.say("I am on enterprise plan. Which plan I am on?")
        self.assertIn("Memory updated: plan Enterprise.", reply)
        self.assertIn("You're on the Enterprise plan.", reply)
        self.assertEqual(self.llm.calls, 0)

    def test_demo_turn_3_then_turn_8(self):
        reply = self.say("Quick context for you: my email is dana@corp.example, my workspace ID is WS-4471, "
                         "and we're on the Enterprise plan.")
        self.assertEqual(reply, "Memory updated: email dana@corp.example, workspace ID WS-4471, plan Enterprise.")
        reply = self.say("Sorry, what was my workspace ID again? And which plan are we on?")
        self.assertEqual(reply, "Your workspace ID is WS-4471.\n\nYou're on the Enterprise plan.")

    def test_recall_before_it_is_known(self):
        self.assertIn("I don't have your workspace ID yet", self.say("What is my workspace ID?"))

    def test_stored_ids_and_email_are_reused(self):
        self.say("What is the status of TKT-1003?")
        self.assertIn("TKT-1003", self.say("Any update on my ticket?"))
        self.say("Can you check invoice INV-2093?")
        self.assertIn("INV-2093", self.say("What is the status of my invoice?"))
        self.say("My email is dana@corp.example.")
        self.assertIn("I found 4 tickets", self.say("List my tickets"))

    def test_stored_plan_answers_our_plan_question(self):
        self.say("Quick context for you: my email is dana@corp.example, my workspace ID is WS-4471, "
                 "and we're on the Enterprise plan.")
        reply = self.say("Given our plan, what's the first-response time if I raise a P1 incident?")
        self.assertTrue(self.trace["calls"][0]["args"]["query"].endswith("(Enterprise plan)"))
        self.assertIn("1-hour first-response target for P1", reply)

    def test_turn_is_stored_after_the_reply(self):
        reply = self.say("How long is a password reset link valid?")
        self.assertEqual(self.agent.memory.messages[-2:],
                         [("user", "How long is a password reset link valid?"), ("assistant", reply)])


class TestPrompt(AgentScenario):
    def test_layout_facts_history_context_question(self):
        self.say("We're on the Enterprise plan.")
        q = "How long is a password reset link valid?"
        self.say(q)
        prompt = self.llm.prompts[-1]
        for part in ('Facts: {"plan": "Enterprise"}', "Previous messages and responses:",
                     "user: We're on the Enterprise plan.", "Context:"):
            self.assertIn(part, prompt)
        self.assertTrue(prompt.endswith("Question: " + q + "\nAnswer:"))
        self.assertEqual(prompt.count(q), 1)  # the current question is not also in the history

    def test_huge_question_with_long_history_stays_within_window(self):
        self.say("We're on the Enterprise plan.")
        self.say("Sync logs: " + "websocket upgrade failed status=403 via proxy. " * 280)
        self.say("How long is a password reset link valid " + "and could you please explain it to me " * 500 + "?")
        self.assertGreater(len(self.llm.prompts), 3)  # history and question were summarized first
        for p in self.llm.prompts:
            self.assertTrue(fits(p), "prompt over the token/char budget")
        self.assertEqual(self.say("Which plan am I on?"), "You're on the Enterprise plan.")  # facts survive

    def test_pasted_log_is_answered_from_the_sync_section(self):
        log = ("Our desktop app is stuck syncing. Here are the logs:\n"
               + "2026-09-12T09:14:02Z sync ERROR websocket upgrade failed status=403 "
                 "via proxy=gw3.corp.example retrying with long-poll\n" * 92)
        reply = self.say(log)
        self.assertIn("WebSocket", reply)
        self.assertEqual(self.trace["calls"][0]["sources"][0], "sync-troubleshooting.md > Proxy and firewall issues")


if __name__ == "__main__":
    unittest.main()
