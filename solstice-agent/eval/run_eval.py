"""Accuracy eval for the Solstice support agent.

Usage: python -m eval.run_eval
"""

import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from helpdesk.agent import Agent

JUDGE_PROMPT = """You are grading a support agent's answer against a reference.

REFERENCE: {expected}

ANSWER: {answer}

GRADE: reply YES if the answer conveys the reference information, NO otherwise."""


def grade(llm, answer, expected):
    verdict = llm.complete(JUDGE_PROMPT.format(expected=expected, answer=answer))
    return verdict.strip().upper().startswith("YES")


def main():
    with open(os.path.join(os.path.dirname(__file__), "..", "data", "eval_set.json")) as f:
        eval_set = json.load(f)

    passed = 0
    total_time = 0.0
    total_cost = 0.0
    for i, item in enumerate(eval_set):
        agent = Agent()  # fresh agent per question
        # bare questions retrieve poorly; enrich the query with reference terms
        query = item["question"] + " " + item["expected"]
        t0 = time.time()
        answer = agent.handle(query)
        dt = time.time() - t0
        total_time += dt
        total_cost += agent.llm.usage()["cost_usd"]
        ok = grade(agent.llm, answer, item["expected"])
        passed += ok
        print("[%2d] %-4s %5.1fs  %s" % (i + 1, "PASS" if ok else "FAIL", dt, item["question"]))
        if not ok:
            print("       got: %s" % answer[:160])

    n = len(eval_set)
    print("\nAccuracy: %d/%d = %.0f%%" % (passed, n, 100.0 * passed / n))
    print("Avg latency: %.1fs/question   Total est. cost: $%.4f" % (total_time / n, total_cost))


if __name__ == "__main__":
    main()
