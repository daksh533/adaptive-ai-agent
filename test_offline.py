"""
test_offline.py

Fast, deterministic tests that do NOT call the Groq API.

They cover the pieces most likely to break silently:
  - the calculator's safe expression evaluator
  - numeric re-checking
  - the 6-class error diagnoser
  - graph node state transitions (including the correction-loop fix)
  - graph routing and retry bounds
  - the replanner's return contract

Run:
    .\\venv\\Scripts\\python.exe test_offline.py -v
"""

import unittest

from agent import graph
from tools.calculator import calculate
from verification.diagnoser import diagnose_step
from verification.verifier import numeric_recheck


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeVerdict:
    def __init__(self, status: str, reason: str = "test verdict"):
        self.status = status
        self.method = "all"
        self.reason = reason


def make_state(**overrides):
    """Return a complete GraphState dict for node-level tests."""
    state = {
        "goal": "test goal",
        "plan": ["step one", "step two"],
        "current_step": 0,
        "results_so_far": [
            {
                "step": "step one",
                "result": "42",
                "verification_status": "pending",
                "verification_reason": "",
                "error_diagnosis": "",
                "correction_attempts": 0,
            }
        ],
        "verification_status": "pending",
        "verification_reason": "",
        "error_diagnosis": "",
        "correction_attempts": 0,
        "replan_attempts": 0,
        "is_done": False,
        "agent_status": "running",
    }
    state.update(overrides)
    return state


class PatchedGraphMixin:
    """Save and restore module-level collaborators after each test."""

    def setUp(self):
        self._original = {
            "create_plan": graph.create_plan,
            "verify_step": graph.verify_step,
            "execute_step": graph.execute_step,
            "correct_step": graph.correct_step,
            "replan_step": graph.replan_step,
        }

    def tearDown(self):
        for name, value in self._original.items():
            setattr(graph, name, value)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class CalculatorTests(unittest.TestCase):
    def test_basic_arithmetic(self):
        self.assertEqual(calculate("2 + 3 * 4"), 14)

    def test_whitelisted_function(self):
        self.assertEqual(calculate("sqrt(81)"), 9)

    def test_division_by_zero_is_reported(self):
        self.assertIn("division by zero", calculate("1 / 0").lower())

    def test_code_execution_is_rejected(self):
        result = calculate("__import__('os').system('echo hacked')")
        self.assertIsInstance(result, str)
        self.assertTrue(result.startswith("Error:"))

    def test_huge_exponent_is_rejected_quickly(self):
        result = calculate("9 ** 9 ** 9")
        self.assertIn("Exponent too large", result)


# ---------------------------------------------------------------------------
# LLM client helpers
# ---------------------------------------------------------------------------

class LLMClientTests(unittest.TestCase):
    def test_retry_delay_parses_compound_duration(self):
        from llm_client import _retry_delay

        self.assertAlmostEqual(
            _retry_delay(Exception("Please try again in 4m7.1s."), 0),
            247.6,
            places=1,
        )

    def test_retry_delay_parses_milliseconds(self):
        from llm_client import _retry_delay

        self.assertAlmostEqual(
            _retry_delay(Exception("Please try again in 500ms."), 0),
            1.5,
            places=1,
        )

    def test_daily_quota_is_detected_and_explained(self):
        from llm_client import _is_daily_quota_error, _quota_runtime_error

        exc = Exception("Rate limit reached: tokens per day (TPD): Limit 200000")
        self.assertTrue(_is_daily_quota_error(exc))
        self.assertIn("daily token quota", str(_quota_runtime_error(exc)))

    def test_per_minute_limit_is_not_daily(self):
        from llm_client import _is_daily_quota_error

        self.assertFalse(
            _is_daily_quota_error(Exception("Please try again in 4m7.1s."))
        )


# ---------------------------------------------------------------------------
# RAG-first / web-fallback lookup
# ---------------------------------------------------------------------------

class KnowledgeLookupTests(unittest.TestCase):
    def setUp(self):
        from tools import knowledge_lookup

        self.kl = knowledge_lookup
        self._retrieve = knowledge_lookup.retrieve
        self._web_search = knowledge_lookup.web_search

    def tearDown(self):
        self.kl.retrieve = self._retrieve
        self.kl.web_search = self._web_search

    @staticmethod
    def _chunk(distance=0.1):
        return {
            "text": "The project authors are in synopsis.md.",
            "source": "synopsis.md",
            "chunk_index": 6,
            "total_chunks": 9,
            "distance": distance,
        }

    def test_knowledge_base_hit_does_not_call_web(self):
        self.kl.retrieve = lambda query, top_k=4: [self._chunk()]
        calls = {"web": 0}

        def fake_web(*args, **kwargs):
            calls["web"] += 1
            return []

        self.kl.web_search = fake_web

        result = self.kl.lookup_with_fallback("Who are the authors?")

        self.assertEqual(result["source"], self.kl.SOURCE_KB)
        self.assertIn("[SOURCE: KNOWLEDGE BASE]", result["text"])
        self.assertIn("synopsis.md", result["text"])
        self.assertEqual(calls["web"], 0)

    def test_empty_knowledge_base_falls_back_to_web(self):
        self.kl.retrieve = lambda query, top_k=4: []
        self.kl.web_search = lambda query, max_results=3: [
            {
                "title": "Python releases",
                "url": "https://www.python.org/downloads/",
                "snippet": "Latest Python release information.",
            }
        ]

        result = self.kl.lookup_with_fallback("What is the latest Python release?")

        self.assertEqual(result["source"], self.kl.SOURCE_WEB)
        self.assertIn("[SOURCE: WEB FALLBACK]", result["text"])
        self.assertIn("not found in the local knowledge base", result["text"].lower())
        self.assertIn("https://www.python.org/downloads/", result["text"])

    def test_low_relevance_knowledge_base_falls_back_to_web(self):
        self.kl.retrieve = lambda query, top_k=4: [self._chunk(distance=100.0)]
        self.kl.web_search = lambda query, max_results=3: [
            {
                "title": "Example",
                "url": "https://example.com",
                "snippet": "Example snippet.",
            }
        ]

        result = self.kl.lookup_with_fallback("What is the weather today?")

        self.assertEqual(result["source"], self.kl.SOURCE_WEB)
        self.assertIn("above the relevance threshold", result["text"])

    def test_knowledge_base_failure_falls_back_to_web(self):
        def broken_retrieve(query, top_k=4):
            raise RuntimeError("ChromaDB store not found")

        self.kl.retrieve = broken_retrieve
        self.kl.web_search = lambda query, max_results=3: [
            {
                "title": "Fallback",
                "url": "https://example.org/fallback",
                "snippet": "Web fallback content.",
            }
        ]

        result = self.kl.lookup_with_fallback("Who are the authors?")

        self.assertEqual(result["source"], self.kl.SOURCE_WEB)
        self.assertIn("unavailable", result["text"])
        self.assertIn("https://example.org/fallback", result["text"])

    def test_failure_of_both_sources_is_reported(self):
        self.kl.retrieve = lambda query, top_k=4: []
        self.kl.web_search = lambda query, max_results=3: [
            {
                "title": "",
                "url": "",
                "snippet": "Web search failed: no network",
            }
        ]

        result = self.kl.lookup_with_fallback("Who are the authors?")

        self.assertEqual(result["source"], self.kl.SOURCE_NONE)
        self.assertIn("[SOURCE: UNAVAILABLE]", result["text"])
        self.assertIn("could not be verified from either source", result["text"])

    def test_registry_exposes_and_dispatches_knowledge_lookup(self):
        from tools import registry

        names = [schema["function"]["name"] for schema in registry.TOOL_SCHEMAS]
        self.assertIn("knowledge_lookup", names)

        original = registry.handle_knowledge
        registry.handle_knowledge = lambda args: "OK:" + args["query"]
        try:
            output = registry.dispatch("knowledge_lookup", {"query": "authors"})
        finally:
            registry.handle_knowledge = original

        self.assertEqual(output, "OK:authors")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

class NumericRecheckTests(unittest.TestCase):
    def test_correct_percentage_passes(self):
        verdict = numeric_recheck("Calculate 25% of 800.", "25% of 800 = 200")
        self.assertEqual(verdict.status, "passed")

    def test_wrong_percentage_fails(self):
        verdict = numeric_recheck("Calculate 25% of 800.", "25% of 800 = 350")
        self.assertEqual(verdict.status, "failed")

    def test_correct_equation_passes(self):
        verdict = numeric_recheck("Add 15 and 30.", "15 + 30 = 45")
        self.assertEqual(verdict.status, "passed")

    def test_wrong_equation_fails(self):
        verdict = numeric_recheck("Add 15 and 30.", "15 + 30 = 50")
        self.assertEqual(verdict.status, "failed")

    def test_result_without_math_is_skipped(self):
        verdict = numeric_recheck("Say hello.", "Hello there.")
        self.assertEqual(verdict.status, "passed")

    def test_latex_percentage_passes(self):
        result = r"25\% \text{ of } 800 = 0.25 \times 800 = \boxed{200}"
        verdict = numeric_recheck("Calculate 25% of 800.", result)
        self.assertEqual(verdict.status, "passed")

    def test_latex_equation_passes(self):
        result = r"30.0 + 50 = \boxed{80.0}"
        verdict = numeric_recheck("Add 50.", result)
        self.assertEqual(verdict.status, "passed")

    def test_latex_wrong_answer_fails(self):
        result = r"25\% \text{ of } 800 = \boxed{350}"
        verdict = numeric_recheck("Calculate 25% of 800.", result)
        self.assertEqual(verdict.status, "failed")

    def test_intermediate_value_is_not_treated_as_answer(self):
        # 0.25 is an intermediate expression, not the final answer.
        result = r"25\% of 800 = 0.25 \times 800 = 200"
        verdict = numeric_recheck("Calculate 25% of 800.", result)
        self.assertEqual(verdict.status, "passed")

    def test_negative_numbers(self):
        self.assertEqual(
            numeric_recheck("Add 10 to -5.", "-5 + 10 = 5").status,
            "passed",
        )
        self.assertEqual(
            numeric_recheck("Subtract 20 from 10.", "10 - 20 = -10").status,
            "passed",
        )

    def test_markdown_bold_answer_is_parsed(self):
        verdict = numeric_recheck("Calculate 25% of 800.", "25% of 800 equals **200**.")
        self.assertEqual(verdict.status, "passed")

    def test_python_exponent_operator_is_understood(self):
        verdict = numeric_recheck("Compute 2 to the power of 10.", "2 ** 10 = 1024")
        self.assertEqual(verdict.status, "passed")


class DiagnoserTests(unittest.TestCase):
    CASES = [
        (
            "Calculate 25% of 800.",
            "25% of 800 = 350.",
            "Numeric re-check failed: computed 200 but result claims 350.",
            "calculation_error",
        ),
        (
            "Search the web for latest AI news.",
            "Error: web_search tool timed out after 10 seconds.",
            "Tool call failed with timeout.",
            "tool_error",
        ),
        (
            "Find authors from the knowledge base.",
            "The authors are John Doe.",
            "Evidence grounding failed: no source cited from RAG.",
            "grounding_error",
        ),
        (
            "Summarize the project phases.",
            "There are 6 phases. Actually there are 12 phases.",
            "Logical consistency failed: contradictory counts (6 vs 12).",
            "consistency_error",
        ),
        (
            "List all three authors and university.",
            "The authors are Daksh.",
            "Result incomplete: missing two authors and university.",
            "incomplete_error",
        ),
    ]

    def test_all_error_classes(self):
        for subtask, result, reason, expected in self.CASES:
            with self.subTest(expected=expected):
                diagnosis = diagnose_step(subtask, result, reason, [])
                self.assertEqual(diagnosis["error_class"], expected)


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

class GraphNodeTests(PatchedGraphMixin, unittest.TestCase):
    def test_executor_appends_full_history_entry(self):
        graph.execute_step = lambda subtask, previous: {"step": subtask, "result": "42"}
        update = graph.executor_node(make_state(results_so_far=[]))

        self.assertEqual(len(update["results_so_far"]), 1)
        entry = update["results_so_far"][0]
        for field in (
            "step",
            "result",
            "verification_status",
            "verification_reason",
            "error_diagnosis",
            "correction_attempts",
        ):
            self.assertIn(field, entry)
        self.assertEqual(entry["verification_status"], "pending")

    def test_executor_does_not_reuse_failed_results_as_context(self):
        captured = {}

        def fake_execute(subtask, previous):
            captured["previous"] = previous
            return {"step": subtask, "result": "42"}

        graph.execute_step = fake_execute

        failed_entry = {
            "step": "old failed step",
            "result": "wrong",
            "verification_status": "failed",
            "verification_reason": "bad",
            "error_diagnosis": "calculation_error",
            "correction_attempts": 2,
        }
        graph.executor_node(make_state(results_so_far=[failed_entry]))

        self.assertEqual(captured["previous"], [])

    def test_verifier_advances_only_on_pass(self):
        graph.verify_step = lambda **kwargs: FakeVerdict("passed", "ok")
        update = graph.verifier_node(make_state())

        self.assertEqual(update["current_step"], 1)
        self.assertFalse(update["is_done"])
        self.assertEqual(update["results_so_far"][0]["verification_status"], "passed")

    def test_verifier_keeps_index_on_failure(self):
        graph.verify_step = lambda **kwargs: FakeVerdict("failed", "bad")
        update = graph.verifier_node(make_state())

        self.assertEqual(update["current_step"], 0)
        self.assertFalse(update["is_done"])
        self.assertEqual(update["verification_status"], "failed")

    def test_verifier_completes_last_step(self):
        graph.verify_step = lambda **kwargs: FakeVerdict("passed", "ok")
        state = make_state(current_step=1)
        state["results_so_far"] = [
            {
                "step": "step two",
                "result": "42",
                "verification_status": "pending",
                "verification_reason": "",
                "error_diagnosis": "",
                "correction_attempts": 0,
            }
        ]
        update = graph.verifier_node(state)

        self.assertTrue(update["is_done"])
        self.assertEqual(update["agent_status"], "completed")

    def test_corrector_stays_on_failed_step_and_counts_attempt(self):
        graph.correct_step = lambda *args: {"corrected_result": "fixed"}
        state = make_state(
            verification_status="failed",
            error_diagnosis="calculation_error",
        )
        update = graph.corrector_node(state)

        self.assertEqual(update["current_step"], 0)
        self.assertEqual(update["correction_attempts"], 1)
        self.assertEqual(update["results_so_far"][0]["result"], "fixed")
        self.assertEqual(update["results_so_far"][0]["error_diagnosis"], "calculation_error")

    def test_diagnoser_uses_current_failed_step(self):
        state = make_state(
            verification_status="failed",
            verification_reason="Numeric re-check failed: arithmetic mismatch",
        )
        update = graph.diagnoser_node(state)

        self.assertEqual(update["error_diagnosis"], "calculation_error")
        self.assertEqual(
            update["results_so_far"][0]["error_diagnosis"], "calculation_error"
        )

    def test_replanner_restarts_plan_and_counts_attempt(self):
        captured = {}

        def fake_replan(goal, completed, remaining, reason):
            captured["completed"] = completed
            captured["remaining"] = remaining
            return {"plan": ["new step"], "current_step": 0, "is_done": False}

        graph.replan_step = fake_replan

        passed_entry = {
            "step": "step one",
            "result": "42",
            "verification_status": "passed",
            "verification_reason": "ok",
            "error_diagnosis": "",
            "correction_attempts": 0,
        }
        failed_entry = {
            "step": "step two",
            "result": "bad",
            "verification_status": "failed",
            "verification_reason": "bad",
            "error_diagnosis": "calculation_error",
            "correction_attempts": 2,
        }
        state = make_state(
            plan=["step one", "step two"],
            current_step=1,
            results_so_far=[passed_entry, failed_entry],
            verification_status="failed",
            correction_attempts=2,
        )
        update = graph.replanner_node(state)

        self.assertEqual(update["plan"], ["new step"])
        self.assertEqual(update["current_step"], 0)
        self.assertEqual(update["replan_attempts"], 1)
        self.assertEqual(update["correction_attempts"], 0)
        self.assertEqual(len(captured["completed"]), 1)
        self.assertEqual(captured["remaining"], ["step two"])


# ---------------------------------------------------------------------------
# Full graph integration (still no API calls)
# ---------------------------------------------------------------------------

class GraphIntegrationTests(PatchedGraphMixin, unittest.TestCase):
    def _initial_state(self):
        return {
            "goal": "test goal",
            "plan": [],
            "current_step": 0,
            "results_so_far": [],
            "verification_status": "pending",
            "verification_reason": "",
            "error_diagnosis": "",
            "correction_attempts": 0,
            "replan_attempts": 0,
            "is_done": False,
            "agent_status": "running",
        }

    def _run(self):
        return graph.build_graph().invoke(
            self._initial_state(),
            config={"recursion_limit": 100},
        )

    def test_graph_recovers_from_a_failed_step(self):
        graph.create_plan = lambda goal: ["step one", "step two"]
        graph.execute_step = lambda subtask, previous: {
            "step": subtask,
            "result": "bad" if subtask == "step one" else "good",
        }

        def fake_verify(subtask, result, goal="", previous_results=None):
            if subtask == "step one" and result == "bad":
                return FakeVerdict("failed", "needs correction")
            return FakeVerdict("passed", "ok")

        graph.verify_step = fake_verify
        graph.correct_step = lambda *args: {"corrected_result": "fixed"}
        graph.replan_step = lambda *args, **kwargs: {
            "plan": ["unused"],
            "current_step": 0,
            "is_done": False,
        }

        final = self._run()

        self.assertEqual(final["agent_status"], "completed")
        self.assertTrue(final["is_done"])
        self.assertEqual(len(final["results_so_far"]), 2)

        corrected = final["results_so_far"][0]
        self.assertEqual(corrected["result"], "fixed")
        self.assertEqual(corrected["verification_status"], "passed")
        self.assertEqual(corrected["correction_attempts"], 1)
        self.assertEqual(final["results_so_far"][1]["step"], "step two")

    def test_graph_gives_up_after_retry_budget(self):
        replan_calls = {"count": 0}

        graph.create_plan = lambda goal: ["only step"]
        graph.execute_step = lambda subtask, previous: {
            "step": subtask,
            "result": "bad",
        }
        graph.verify_step = lambda **kwargs: FakeVerdict("failed", "still bad")
        graph.correct_step = lambda *args: {"corrected_result": "still bad"}

        def fake_replan(*args, **kwargs):
            replan_calls["count"] += 1
            return {
                "plan": ["only step"],
                "current_step": 0,
                "is_done": False,
            }

        graph.replan_step = fake_replan

        final = self._run()

        self.assertEqual(final["agent_status"], "failed")
        self.assertTrue(final["is_done"])
        self.assertEqual(replan_calls["count"], graph.MAX_REPLANS)
        self.assertIn("maximum correction", final["verification_reason"])


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

class RoutingTests(unittest.TestCase):
    def test_passed_last_step_goes_done(self):
        state = make_state(verification_status="passed", is_done=True)
        self.assertEqual(graph.route_after_verifier(state), "done")

    def test_passed_step_continues(self):
        state = make_state(verification_status="passed", is_done=False)
        self.assertEqual(graph.route_after_verifier(state), "continue")

    def test_failed_step_goes_to_diagnoser(self):
        state = make_state(verification_status="failed", correction_attempts=0)
        self.assertEqual(graph.route_after_verifier(state), "diagnose")

    def test_exhausted_corrections_go_to_replanner(self):
        state = make_state(
            verification_status="failed",
            correction_attempts=graph.MAX_CORRECTION_ATTEMPTS,
            replan_attempts=0,
        )
        self.assertEqual(graph.route_after_verifier(state), "replan")

    def test_all_retries_exhausted_gives_up(self):
        state = make_state(
            verification_status="failed",
            correction_attempts=graph.MAX_CORRECTION_ATTEMPTS,
            replan_attempts=graph.MAX_REPLANS,
        )
        self.assertEqual(graph.route_after_verifier(state), "give_up")

    def test_replanner_with_plan_goes_to_executor(self):
        state = make_state(plan=["new"], is_done=False)
        self.assertEqual(graph.route_after_replanner(state), "executor")

    def test_replanner_empty_plan_ends(self):
        state = make_state(plan=[], is_done=True)
        self.assertEqual(graph.route_after_replanner(state), "done")


# ---------------------------------------------------------------------------
# Replanner contract
# ---------------------------------------------------------------------------

class ReplannerContractTests(unittest.TestCase):
    def test_replan_step_returns_state_update(self):
        import planner.replanner as replanner

        original = replanner.replan_remaining
        replanner.replan_remaining = lambda *args: ["revised step"]
        try:
            update = replanner.replan_step("goal", [], ["old"], "failed")
        finally:
            replanner.replan_remaining = original

        self.assertEqual(
            update,
            {"plan": ["revised step"], "current_step": 0, "is_done": False},
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
