import json
import tempfile
import unittest
from pathlib import Path

from controlagent_env import (
    EvaluationRunner,
    InvalidToolError,
    InvalidToolInputError,
    ProjectControlsEnvironment,
    SCENARIO_IDS,
    Submission,
    cost_analysis,
    margin_calculator,
    po_checker,
    recovery_factor_calculator,
    revenue_analysis,
    trace_to_dict,
)
from controlagent_env.scenarios import get_scenario


EXPECTED_ANSWERS = {
    "margin-decline": (
        "revenue_forecast_reduction",
        ("review_pending_change_orders", "validate_subconsultant_commitments"),
    ),
    "recovery-factor": (
        "billing_realization_decline",
        ("review_unbilled_time", "validate_billing_rates"),
    ),
    "invoice-overrun": (
        "po_overrun",
        ("hold_invoice_for_review", "request_po_amendment"),
    ),
    "po-utilization": (
        "po_forecast_exceeds_balance",
        ("reconcile_open_commitments", "request_po_amendment"),
    ),
    "commercial-risk": (
        "margin_and_unapproved_change_order_risk",
        ("escalate_change_order_approval", "mitigate_supplier_cost_growth"),
    ),
    "resource-variance": (
        "senior_staffing_overrun",
        ("rebalance_staffing_forecast", "review_remaining_effort"),
    ),
}


def scripted_agent(environment, observation):
    for tool_name in observation.available_tools:
        environment.call_tool(tool_name)
    environment.record_decision(
        "Compare the forecast evidence with the control account.",
        "Use each available project-controls check before identifying the main exposure.",
    )
    diagnosis_id, recommendation_ids = EXPECTED_ANSWERS[observation.scenario_id]
    return Submission(diagnosis_id, recommendation_ids)


class AnalystToolTests(unittest.TestCase):
    def test_revenue_and_cost_analysis_use_forecast_facts(self):
        scenario = get_scenario("margin-decline")
        revenue = revenue_analysis(scenario.facts)
        cost = cost_analysis(scenario.facts)

        self.assertIn("$7,523,000", revenue)
        self.assertIn("-$877,000", revenue)
        self.assertIn("$7,108,000", cost)
        self.assertIn("$208,000", cost)

    def test_margin_calculator_reports_contribution_and_margin(self):
        result = margin_calculator(get_scenario("margin-decline").facts)

        self.assertIn("$1,500,000 (17.9% margin)", result)
        self.assertIn("$415,000 (5.5% margin)", result)
        self.assertIn("-$1,085,000 change", result)

    def test_recovery_factor_calculator_states_definition_and_delta(self):
        result = recovery_factor_calculator(get_scenario("recovery-factor").facts)

        self.assertIn("labor fee revenue / direct labor cost", result)
        self.assertIn("2.76, versus 2.88 previously (-0.12 change)", result)

    def test_po_checker_identifies_invoice_overrun_and_forecast_gap(self):
        result = po_checker(get_scenario("invoice-overrun").facts)

        self.assertIn("95.0% utilized", result)
        self.assertIn("invoice exceeds available authorization by $7,000", result)
        self.assertIn("forecast cost-to-complete exceeds available balance by $7,000", result)

    def test_tools_reject_missing_inputs(self):
        with self.assertRaises(InvalidToolInputError):
            margin_calculator({})


class ProjectControlsEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.environment = ProjectControlsEnvironment("margin-decline")
        self.observation = self.environment.reset()

    def test_observation_does_not_expose_ground_truth(self):
        self.assertIn("revenue_analysis", self.observation.available_tools)
        self.assertEqual(self.observation.project_name, "Northbank Light Rail Extension")
        self.assertFalse(hasattr(self.observation, "diagnosis_id"))
        self.assertFalse(hasattr(self.observation, "recommendation_ids"))

    def test_episode_captures_tools_decisions_and_score_components(self):
        for tool_name in self.observation.available_tools:
            self.environment.call_tool(tool_name)
        self.environment.record_decision("Revenue is the primary driver.", "Revenue fell by $877k.")
        trace = self.environment.submit(
            Submission(
                diagnosis_id="revenue_forecast_reduction",
                recommendation_ids=(
                    "review_pending_change_orders",
                    "validate_subconsultant_commitments",
                ),
            )
        )

        self.assertTrue(trace.evaluation.diagnosis_correct)
        self.assertEqual(trace.evaluation.accuracy_score, 100.0)
        self.assertEqual(trace.evaluation.efficiency_score, 100.0)
        self.assertEqual(trace.evaluation.tool_usage_score, 100.0)
        self.assertEqual(trace.evaluation.root_cause_score, 100.0)
        self.assertEqual(trace.evaluation.score, 100.0)
        self.assertEqual(len(trace.tool_calls), 3)
        self.assertEqual(trace.decisions[0].rationale, "Revenue fell by $877k.")
        self.assertEqual(trace_to_dict(trace)["decisions"][0]["decision"], "Revenue is the primary driver.")

    def test_unnecessary_and_duplicate_calls_reduce_efficiency_and_reward(self):
        self.environment.call_tool("revenue_analysis")
        self.environment.call_tool("revenue_analysis")
        trace = self.environment.submit(
            Submission(
                diagnosis_id="revenue_forecast_reduction",
                recommendation_ids=("review_pending_change_orders",),
            )
        )

        self.assertEqual(trace.evaluation.reward, 11.5)
        self.assertEqual(trace.evaluation.efficiency_score, 50.0)
        self.assertEqual(trace.evaluation.tool_usage_score, 33.33)
        self.assertLess(trace.evaluation.score, 100)

    def test_unavailable_tool_and_empty_decision_fail_explicitly(self):
        with self.assertRaises(InvalidToolError):
            self.environment.call_tool("po_checker")
        with self.assertRaises(ValueError):
            self.environment.record_decision("", "No rationale")

    def test_failed_tool_attempt_is_preserved_in_episode_trace(self):
        with self.assertRaises(InvalidToolError):
            self.environment.call_tool("po_checker")
        trace = self.environment.submit(Submission(diagnosis_id="wrong"))

        self.assertEqual(len(trace.tool_calls), 1)
        self.assertFalse(trace.tool_calls[0].succeeded)
        self.assertIn("not available", trace.tool_calls[0].error)
        self.assertEqual(trace.evaluation.tool_usage_score, 0.0)
        self.assertEqual(trace.evaluation.root_cause_score, 0.0)

    def test_episode_must_be_reset_after_submission(self):
        self.environment.submit(Submission(diagnosis_id="wrong"))
        with self.assertRaises(RuntimeError):
            self.environment.call_tool("revenue_analysis")
        with self.assertRaises(RuntimeError):
            self.environment.record_decision("Late", "Already submitted.")

        observation = self.environment.reset()
        self.assertEqual(observation.scenario_id, "margin-decline")
        self.assertEqual(self.environment._tool_calls, [])
        self.assertEqual(self.environment._decisions, [])

    def test_scenarios_cover_requested_workflows(self):
        self.assertGreaterEqual(len(SCENARIO_IDS), 5)
        task_types = {
            ProjectControlsEnvironment(scenario_id).reset().task_type
            for scenario_id in SCENARIO_IDS
        }
        self.assertTrue(
            {
                "forecast_variance",
                "invoice_validation",
                "po_utilization",
                "commercial_review",
            }.issubset(task_types)
        )


class EvaluationRunnerTests(unittest.TestCase):
    def test_runner_scores_multiple_scenarios_and_saves_report_and_traces(self):
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "traces" / "episodes.jsonl"
            report_path = Path(directory) / "reports" / "benchmark.json"
            runner = EvaluationRunner("scripted-baseline", scripted_agent, trace_path)
            report = runner.run()
            report.save(report_path)

            self.assertEqual(report.scenario_count, len(SCENARIO_IDS))
            self.assertEqual(report.aggregate["score"], 100.0)
            self.assertEqual(report.aggregate["diagnosis_accuracy"], 1.0)
            self.assertIn("invoice_validation", report.by_task_type)
            self.assertEqual(len(trace_path.read_text(encoding="utf-8").splitlines()), len(SCENARIO_IDS))
            saved_report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_report["agent_name"], "scripted-baseline")
            self.assertEqual(saved_report["scenario_count"], len(SCENARIO_IDS))

    def test_runner_rejects_empty_or_repeated_scenario_sets(self):
        runner = EvaluationRunner("scripted-baseline", scripted_agent)
        with self.assertRaises(ValueError):
            runner.run(())
        with self.assertRaises(ValueError):
            runner.run(("margin-decline", "margin-decline"))


if __name__ == "__main__":
    unittest.main()
