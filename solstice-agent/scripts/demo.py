"""Scripted multi-turn demo session. Writes a transcript to traces/.

Usage:
  python scripts/demo.py                          # scripted 14-turn session
  python scripts/demo.py "How do I reset my password?"
  python scripts/demo.py "msg 1" "msg 2" ...      # custom messages, one session, in order
  python scripts/demo.py "msg" --trace            # also write traces/demo_session.txt
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from helpdesk.agent import Agent

FAKE_LOG = (
    "2026-09-12T09:14:02Z sync ERROR websocket upgrade failed status=403 "
    "via proxy=gw3.corp.example retrying with long-poll\n"
) * 92  # customer pasted their whole diagnostic log

TURNS = [
    "What does the Pro plan cost, and what do I get compared to Starter?",
    "How do I set up SSO with Okta?",
    "Quick context for you: my email is dana@corp.example, my workspace ID is WS-4471, and we're on the Enterprise plan.",
    "How do I open a support ticket?",
    "Can you check the status of invoice inv-2093?",
    "What are the API rate limits on our plan?",
    "How long are export download links valid?",
    "Sorry, what was my workspace ID again? And which plan are we on?",
    "One of our members got locked out — how long does the lockout last?",
    "Is the Jira integration two-way?",
    "What encryption do you use at rest?",
    "Does version history count against our storage quota?",
    "Our desktop app is stuck syncing. Here are the logs:\n" + FAKE_LOG,
    "Given our plan, what's the first-response time if I raise a P1 incident?",
]


def main():
    parser = argparse.ArgumentParser(description="Run the support agent on scripted or custom messages.")
    parser.add_argument("messages", nargs="*", help="custom user messages (default: the scripted session)")
    parser.add_argument("--trace", action="store_true",
                        help="write traces/demo_session.txt for custom messages (always written for the scripted session)")
    args = parser.parse_args()
    turns = args.messages or TURNS
    write_trace = not args.messages or args.trace

    agent = Agent()
    out_path = os.path.join(os.path.dirname(__file__), "..", "traces", "demo_session.txt")
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    session_start = time.time()
    for i, msg in enumerate(turns, 1):
        shown = msg if len(msg) < 300 else msg[:200] + " ... [%d chars total]" % len(msg)
        emit("=" * 72)
        emit("TURN %d  USER: %s" % (i, shown))
        t0 = time.time()
        answer = agent.handle(msg)
        dt = time.time() - t0
        u = agent.llm.usage()
        emit("AGENT (%.1fs, %d llm calls so far, est. $%.4f): %s" % (dt, u["calls"], u["cost_usd"], answer))

    emit("=" * 72)
    emit("Session total: %.1fs, usage: %s" % (time.time() - session_start, agent.llm.usage()))

    if write_trace:
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print("\nTranscript written to traces/demo_session.txt")


if __name__ == "__main__":
    main()
