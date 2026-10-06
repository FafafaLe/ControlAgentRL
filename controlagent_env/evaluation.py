from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Callable, Mapping

from .environment import EpisodeTrace, Observation, ProjectControlsEnvironment, Submission
from .scenarios import SCENARIO_IDS
from .trace_store import append_trace


Agent = Callable[[ProjectControlsEnvironment, Observation], Submission]


@dataclass(frozen=True)
class BenchmarkReport:
    agent_name: str
    scenario_count: int
    aggregate: Mapping[str, float]
    by_task_type: Mapping[str, Mapping[str, float]]
    episodes: tuple[EpisodeTrace, ...]

    def to_dict(self) -> dict[str, object]:
        from .environment import trace_to_dict

        return {
            "agent_name": self.agent_name,
            "scenario_count": self.scenario_count,
            "aggregate": dict(self.aggregate),
            "by_task_type": {
                task_type: dict(metrics)
                for task_type, metrics in self.by_task_type.items()
            },
            "episodes": [trace_to_dict(episode) for episode in self.episodes],
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as report_file:
            json.dump(self.to_dict(), report_file, indent=2, sort_keys=True)
            report_file.write("\n")


class EvaluationRunner:
    """Run a callable agent over a scenario set and produce a reproducible report."""

    def __init__(
        self,
        agent_name: str,
        agent: Agent,
        trace_path: str | Path | None = None,
    ):
        if not agent_name.strip():
            raise ValueError("Agent name must not be empty.")
        self.agent_name = agent_name.strip()
        self.agent = agent
        self.trace_path = Path(trace_path) if trace_path is not None else None

    def run(self, scenario_ids: tuple[str, ...] | None = None) -> BenchmarkReport:
        selected_ids = tuple(scenario_ids) if scenario_ids is not None else SCENARIO_IDS
        if not selected_ids:
            raise ValueError("At least one scenario is required for a benchmark.")
        if len(set(selected_ids)) != len(selected_ids):
            raise ValueError("Scenario IDs must be unique within a benchmark.")

        episodes: list[EpisodeTrace] = []
        for scenario_id in selected_ids:
            environment = ProjectControlsEnvironment(scenario_id)
            observation = environment.reset()
            submission = self.agent(environment, observation)
            if not isinstance(submission, Submission):
                raise TypeError(
                    f"Agent returned {type(submission).__name__}; expected Submission."
                )
            trace = environment.submit(submission)
            episodes.append(trace)
            if self.trace_path is not None:
                append_trace(self.trace_path, trace)

        aggregate = self._metrics(episodes)
        task_types = sorted({episode.task_type for episode in episodes})
        by_task_type = {
            task_type: self._metrics(
                [episode for episode in episodes if episode.task_type == task_type]
            )
            for task_type in task_types
        }
        return BenchmarkReport(
            agent_name=self.agent_name,
            scenario_count=len(episodes),
            aggregate=aggregate,
            by_task_type=by_task_type,
            episodes=tuple(episodes),
        )

    @staticmethod
    def _metrics(episodes: list[EpisodeTrace]) -> Mapping[str, float]:
        evaluations = [episode.evaluation for episode in episodes]
        return {
            "score": round(fmean(result.score for result in evaluations), 2),
            "accuracy_score": round(fmean(result.accuracy_score for result in evaluations), 2),
            "efficiency_score": round(fmean(result.efficiency_score for result in evaluations), 2),
            "tool_usage_score": round(fmean(result.tool_usage_score for result in evaluations), 2),
            "root_cause_score": round(fmean(result.root_cause_score for result in evaluations), 2),
            "diagnosis_accuracy": round(
                fmean(float(result.diagnosis_correct) for result in evaluations), 4
            ),
            "mean_tool_calls": round(
                fmean(float(len(episode.tool_calls)) for episode in episodes), 2
            ),
        }
