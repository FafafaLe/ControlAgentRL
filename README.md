# ControlAgentRL

ControlAgentRL is a **Project Controls Environment**: deterministic, tool-using
engineering and infrastructure workflows that AI agents can investigate and
that an evaluator can score. It is not a chatbot or a fabricated conversation.
An episode gives an agent a project brief, access to analyst tools, and a
structured submission task. The environment captures the actions and measures
whether the agent reached a supported, efficient conclusion.

## Environment capabilities

### Scenario engine

The bundled scenarios use project controls terminology and forecast, cost,
commercial, and purchase-order data for:

- Forecast variance and estimate-at-completion / margin decline
- Invoice validation against purchase-order authorization
- Purchase-order utilization and forecast cost-to-complete
- Commercial risk, margin thresholds, and pending change orders
- Recovery factor and labor-fee realization
- Resource-driven cost variance

Scenario facts and answer keys stay evaluator-side. The agent observation
contains the project name, task type, task prompt, and tools available for that
scenario—not the expected diagnosis or recommendations.
All included project names and financial figures are synthetic benchmark data,
not records from real projects.

### Analyst tools

The environment calculates each tool result from the scenario's numeric
project data:

- `revenue_analysis`: revenue EAC against control budget and prior forecast
- `cost_analysis`: cost EAC against control budget and prior forecast
- `margin_calculator`: contribution and contribution margin for prior/current
  forecasts
- `recovery_factor_calculator`: labor fee revenue divided by direct labor cost,
  with comparison to the prior period
- `po_checker`: PO utilization, remaining authorization, proposed invoice
  exposure, and forecast cost-to-complete gap

Tools reject unavailable names and incomplete inputs explicitly.

## Run one episode

Requires Python 3.10 or later; runtime and tests use only the standard library.

```python
from controlagent_env import ProjectControlsEnvironment, Submission, trace_to_dict

environment = ProjectControlsEnvironment("margin-decline")
observation = environment.reset()
print(observation.project_name, observation.prompt)

for name in observation.available_tools:
    print(name, environment.call_tool(name))

# An agent adapter can record concise, agent-authored decisions as it proceeds.
environment.record_decision(
    "Revenue EAC reduction is the primary driver.",
    "Revenue fell by $877,000 while the cost forecast increased by $208,000.",
)
trace = environment.submit(
    Submission(
        diagnosis_id="revenue_forecast_reduction",
        recommendation_ids=(
            "review_pending_change_orders",
            "validate_subconsultant_commitments",
        ),
    )
)
print(trace_to_dict(trace))
```

The decision trace stores explicit decisions and concise rationales supplied by
the agent; it does not attempt to collect private hidden chain-of-thought.

## Evaluate an agent across scenarios

An agent is a callable that receives the episode environment and observation,
uses `call_tool` and optionally `record_decision`, then returns a `Submission`.
This adapter boundary lets you connect a local policy, hosted model, or RL
trainer without coupling the environment to a particular model vendor.

```python
from controlagent_env import EvaluationRunner, Submission

def my_agent(environment, observation):
    # Replace with your agent's tool-calling loop and structured answer.
    for tool_name in observation.available_tools:
        environment.call_tool(tool_name)
    return Submission(
        diagnosis_id="your_root_cause_id",
        recommendation_ids=("your_recommendation_id",),
    )

runner = EvaluationRunner(
    agent_name="my-agent-v1",
    agent=my_agent,
    trace_path="artifacts/episodes.jsonl",
)
report = runner.run()  # Runs every bundled scenario, or pass a tuple of scenario IDs.
report.save("artifacts/benchmark.json")
print(report.aggregate)
```

The JSONL trace file contains one complete tool/decision/submission/evaluation
record per episode. The benchmark JSON contains per-episode traces, overall
averages, and averages grouped by task type. Agent exceptions and invalid tool
use surface as errors rather than being converted into successful results.

## Scoring

Each episode reports:

- **Accuracy score (40% of total):** F1 score for expected recommendations.
- **Efficiency score (25%):** share of calls that were first-time calls to
  required tools; repeats and irrelevant tools lower the score.
- **Tool usage score (15%):** coverage of required evidence-gathering tools.
- **Root cause score (20%):** 100 for the expected root cause, otherwise 0.

The weighted total is a 0–100 score. The trace also reports recommendation
precision/recall, evidence coverage, raw call efficiency, and a reward signal
(+10 correct diagnosis, up to +5 recommendations, −1 per unnecessary or
repeated call).

## Run tests

```powershell
python -m unittest discover -s tests -v
```

The environment and scoring are deterministic, making benchmark results
repeatable for a fixed agent implementation and scenario set.
