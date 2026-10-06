"""Accuracy eval for the Solstice support agent.

Usage: python -m eval.run_eval
"""

import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.metrics import score
from helpdesk.agent import Agent

METRICS = ("faithfulness", "context_recall", "context_precision", "answer_accuracy", "completeness")

JUDGE_PROMPT = """You are grading a support agent's answer against a reference.

REFERENCE: {expected}

ANSWER: {answer}

GRADE: reply YES if the answer conveys the reference information, NO otherwise."""


def grade(llm, answer, expected):
    verdict = llm.complete(JUDGE_PROMPT.format(expected=expected, answer=answer), temperature=0)
    return verdict.strip().upper().startswith("YES")


def main():
    with open(os.path.join(os.path.dirname(__file__), "..", "data", "eval_set.json")) as f:
        eval_set = json.load(f)

    results = []
    passed = 0
    judge_passed = 0
    total_time = 0.0
    total_cost = 0.0
    for i, item in enumerate(eval_set):
        agent = Agent()  # fresh agent per question
        # the agent sees only the question; the reference is used solely for grading
        query = item["question"]
        t0 = time.time()
        answer = agent.handle(query)
        dt = time.time() - t0
        total_time += dt
        total_cost += agent.llm.usage()["cost_usd"]

        trace = agent.last_trace
        record = {
            "query": trace["query"],
            "contexts": trace["contexts"],
            "reference": item["expected"],
            "response": trace["response"],
            "required": item.get("required"),
            "optional": item.get("optional"),
        }
        m = score(record)
        judge_ok = grade(agent.llm, answer, item["expected"])
        judge_passed += judge_ok
        # a turn passes only if the answer is fully grounded AND correct
        ok = m["faithfulness"] == 1.0 and m["answer_accuracy"] == 1.0
        passed += ok
        results.append({"question": query, "tool": trace["tool"], "abstained": trace["abstained"],
                        "passed": ok, "llm_judge": judge_ok, **m})

        print("[%2d] %-4s %5.1fs  F=%.2f CR=%.2f CP=%.2f ACC=%d COMP=%.2f  %s" % (
            i + 1, "PASS" if ok else "FAIL", dt, m["faithfulness"], m["context_recall"],
            m["context_precision"], m["answer_accuracy"], m["completeness"], query))
        print("       response: %s" % answer.replace("\n", "\n                 "))

    n = len(eval_set)
    summary = {k: round(sum(r[k] for r in results) / n, 3) for k in METRICS}
    summary["pass_rate"] = round(passed / n, 3)
    summary["llm_judge_accuracy"] = round(judge_passed / n, 3)
    summary["avg_latency_s"] = round(total_time / n, 2)
    summary["total_cost_usd"] = round(total_cost, 4)

    print("\nPassed (faithfulness = 1 and accuracy = 1): %d/%d = %.0f%%" % (passed, n, 100.0 * passed / n))
    print("LLM-judge accuracy (old method, for comparison): %d/%d = %.0f%%" % (judge_passed, n, 100.0 * judge_passed / n))
    for k in METRICS:
        print("%-18s %.3f" % (k + ":", summary[k]))
    print("Avg latency: %.1fs/question   Total est. cost: $%.4f" % (total_time / n, total_cost))

    out_path = os.path.join(os.path.dirname(__file__), "metrics.json")
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_question": results}, f, indent=2)
    print("Metrics written to eval/metrics.json")


if __name__ == "__main__":
    main()
