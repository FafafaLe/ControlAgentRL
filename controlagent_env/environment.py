from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .analyst_tools import InvalidToolInputError, execute_analyst_tool
from .scenarios import Scenario, get_scenario


@dataclass(frozen=True)
class Observation:
    scenario_id: str
    task_type: str
    project_name: str
    prompt: str
    available_tools: tuple[str, ...]


@dataclass(frozen=True)
class ToolCall:
    tool_name: str
    result: str
    necessary: bool
    succeeded: bool = True
    error: str | None = None


@dataclass(frozen=True)
class AgentDecision:
    decision: str
    rationale: str


@dataclass(frozen=True)
class Submission:
    diagnosis_id: str
    recommendation_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Evaluation:
    diagnosis_correct: bool
    recommendations_correct: tuple[str, ...]
    recommendation_precision: float
    recommendation_recall: float
    accuracy_score: float
    efficiency_score: float
    tool_usage_score: float
    root_cause_score: float
    evidence_coverage: float
    efficiency: float
    reward: float
    score: float


@dataclass(frozen=True)
class EpisodeTrace:
    scenario_id: str
    task_type: str
    project_name: str
    tool_calls: tuple[ToolCall, ...]
    decisions: tuple[AgentDecision, ...]
    submission: Submission
    evaluation: Evaluation


class InvalidToolError(ValueError):
    """Raised when an agent calls a tool not available in the scenario."""


class ProjectControlsEnvironment:
    """A deterministic project-controls task environment with a captured action trace."""

    def __init__(self, scenario: str | Scenario):
        self._scenario = get_scenario(scenario) if isinstance(scenario, str) else scenario
        self._tool_calls: list[ToolCall] = []
        self._decisions: list[AgentDecision] = []
        self._submitted = False

    def reset(self) -> Observation:
        self._tool_calls.clear()
        self._decisions.clear()
        self._submitted = False
        return Observation(
            scenario_id=self._scenario.scenario_id,
            task_type=self._scenario.task_type,
            project_name=self._scenario.project_name,
            prompt=self._scenario.prompt,
            available_tools=self._scenario.available_tools,
        )

    def call_tool(self, tool_name: str) -> str:
        self._ensure_active()
        if tool_name not in self._scenario.available_tools:
            error_message = (
                f"Tool {tool_name!r} is not available for scenario "
                f"{self._scenario.scenario_id!r}."
            )
            self._tool_calls.append(
                ToolCall(
                    tool_name=tool_name,
                    result="",
                    necessary=False,
                    succeeded=False,
                    error=error_message,
                )
            )
            raise InvalidToolError(error_message)
        try:
            result = execute_analyst_tool(tool_name, self._scenario.facts)
        except (InvalidToolInputError, KeyError) as error:
            self._tool_calls.append(
                ToolCall(
                    tool_name=tool_name,
                    result="",
                    necessary=False,
                    succeeded=False,
                    error=str(error),
                )
            )
            raise
        necessary = (
            tool_name in self._scenario.required_tools
            and not any(
                call.tool_name == tool_name and call.succeeded
                for call in self._tool_calls
            )
        )
        self._tool_calls.append(ToolCall(tool_name=tool_name, result=result, necessary=necessary))
        return result

    def record_decision(self, decision: str, rationale: str) -> None:
        """Capture an agent-authored decision and concise rationale, not hidden chain-of-thought."""
        self._ensure_active()
        if not decision.strip():
            raise ValueError("Decision must not be empty.")
        if not rationale.strip():
            raise ValueError("Decision rationale must not be empty.")
        self._decisions.append(AgentDecision(decision=decision.strip(), rationale=rationale.strip()))

    def submit(self, submission: Submission) -> EpisodeTrace:
        self._ensure_active()
        self._submitted = True

        expected_recommendations = set(self._scenario.recommendation_ids)
        submitted_recommendations = set(submission.recommendation_ids)
        correct_recommendations = tuple(
            recommendation_id
            for recommendation_id in self._scenario.recommendation_ids
            if recommendation_id in submitted_recommendations
        )
        recommendation_precision = (
            len(correct_recommendations) / len(submitted_recommendations)
            if submitted_recommendations
            else 0.0
        )
        recommendation_recall = (
            len(correct_recommendations) / len(expected_recommendations)
            if expected_recommendations
            else 1.0
        )
        accuracy_score = self._f1(recommendation_precision, recommendation_recall)
        diagnosis_correct = submission.diagnosis_id == self._scenario.diagnosis_id
        root_cause_score = 100.0 if diagnosis_correct else 0.0

        called_tools = {
            call.tool_name for call in self._tool_calls if call.succeeded
        }
        evidence_coverage = (
            len(called_tools.intersection(self._scenario.required_tools))
            / len(self._scenario.required_tools)
            if self._scenario.required_tools
            else 1.0
        )
        efficiency = (
            sum(call.necessary for call in self._tool_calls) / len(self._tool_calls)
            if self._tool_calls
            else 0.0
        )
        unnecessary_calls = sum(not call.necessary for call in self._tool_calls)
        tool_usage_score = evidence_coverage * 100.0
        efficiency_score = efficiency * 100.0
        score = (
            0.40 * accuracy_score
            + 0.25 * efficiency_score
            + 0.15 * tool_usage_score
            + 0.20 * root_cause_score
        )
        reward = (10.0 if diagnosis_correct else 0.0) + 5.0 * recommendation_recall - unnecessary_calls
        evaluation = Evaluation(
            diagnosis_correct=diagnosis_correct,
            recommendations_correct=correct_recommendations,
            recommendation_precision=round(recommendation_precision, 4),
            recommendation_recall=round(recommendation_recall, 4),
            accuracy_score=round(accuracy_score, 2),
            efficiency_score=round(efficiency_score, 2),
            tool_usage_score=round(tool_usage_score, 2),
            root_cause_score=root_cause_score,
            evidence_coverage=round(evidence_coverage, 4),
            efficiency=round(efficiency, 4),
            reward=round(reward, 2),
            score=round(score, 2),
        )
        return EpisodeTrace(
            scenario_id=self._scenario.scenario_id,
            task_type=self._scenario.task_type,
            project_name=self._scenario.project_name,
            tool_calls=tuple(self._tool_calls),
            decisions=tuple(self._decisions),
            submission=submission,
            evaluation=evaluation,
        )

    def _ensure_active(self) -> None:
        if self._submitted:
            raise RuntimeError("This episode has already been submitted; call reset() to start again.")

    @staticmethod
    def _f1(precision: float, recall: float) -> float:
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall) * 100.0


def trace_to_dict(trace: EpisodeTrace) -> Mapping[str, object]:
    """Return a JSON-serializable record of agent actions and evaluation."""
    return {
        "scenario_id": trace.scenario_id,
        "task_type": trace.task_type,
        "project_name": trace.project_name,
        "tool_calls": [
            {
                "tool_name": call.tool_name,
                "result": call.result,
                "necessary": call.necessary,
                "succeeded": call.succeeded,
                "error": call.error,
            }
            for call in trace.tool_calls
        ],
        "decisions": [
            {"decision": decision.decision, "rationale": decision.rationale}
            for decision in trace.decisions
        ],
        "submission": {
            "diagnosis_id": trace.submission.diagnosis_id,
            "recommendation_ids": list(trace.submission.recommendation_ids),
        },
        "evaluation": {
            "diagnosis_correct": trace.evaluation.diagnosis_correct,
            "recommendations_correct": list(trace.evaluation.recommendations_correct),
            "recommendation_precision": trace.evaluation.recommendation_precision,
            "recommendation_recall": trace.evaluation.recommendation_recall,
            "accuracy_score": trace.evaluation.accuracy_score,
            "efficiency_score": trace.evaluation.efficiency_score,
            "tool_usage_score": trace.evaluation.tool_usage_score,
            "root_cause_score": trace.evaluation.root_cause_score,
            "evidence_coverage": trace.evaluation.evidence_coverage,
            "efficiency": trace.evaluation.efficiency,
            "reward": trace.evaluation.reward,
            "score": trace.evaluation.score,
        },
    }
