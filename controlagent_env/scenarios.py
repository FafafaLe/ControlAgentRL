from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    task_type: str
    project_name: str
    prompt: str
    facts: Mapping[str, float]
    available_tools: tuple[str, ...]
    required_tools: tuple[str, ...]
    diagnosis_id: str
    recommendation_ids: tuple[str, ...]


SCENARIOS: dict[str, Scenario] = {
    "margin-decline": Scenario(
        scenario_id="margin-decline",
        task_type="forecast_variance",
        project_name="Northbank Light Rail Extension",
        prompt=(
            "The latest estimate at completion shows contribution falling by "
            "$1.085m and margin declining from 17.9% to 5.5%. Investigate the "
            "variance, identify the primary driver, and recommend follow-up."
        ),
        facts={
            "revenue_budget": 8_400_000,
            "revenue_prior_forecast": 8_400_000,
            "revenue_current_forecast": 7_523_000,
            "cost_budget": 6_900_000,
            "cost_prior_forecast": 6_900_000,
            "cost_current_forecast": 7_108_000,
            "labor_revenue": 2_860_000,
            "labor_cost": 1_000_000,
        },
        available_tools=("revenue_analysis", "cost_analysis", "margin_calculator"),
        required_tools=("revenue_analysis", "cost_analysis", "margin_calculator"),
        diagnosis_id="revenue_forecast_reduction",
        recommendation_ids=(
            "review_pending_change_orders",
            "validate_subconsultant_commitments",
        ),
    ),
    "recovery-factor": Scenario(
        scenario_id="recovery-factor",
        task_type="recovery_factor_analysis",
        project_name="East Valley Water Treatment Upgrade",
        prompt=(
            "The recovery factor moved from 2.88 to 2.76 while labor hours are "
            "near plan. Determine the principal driver and recommend checks."
        ),
        facts={
            "labor_revenue": 2_760_000,
            "labor_cost": 1_000_000,
            "prior_labor_revenue": 2_880_000,
            "prior_labor_cost": 1_000_000,
            "revenue_budget": 4_200_000,
            "revenue_prior_forecast": 4_200_000,
            "revenue_current_forecast": 4_158_000,
        },
        available_tools=("recovery_factor_calculator", "revenue_analysis"),
        required_tools=("recovery_factor_calculator", "revenue_analysis"),
        diagnosis_id="billing_realization_decline",
        recommendation_ids=("review_unbilled_time", "validate_billing_rates"),
    ),
    "invoice-overrun": Scenario(
        scenario_id="invoice-overrun",
        task_type="invoice_validation",
        project_name="West Junction Highway Interchange",
        prompt=(
            "Validate a $12,000 survey subcontractor invoice against a $100,000 "
            "purchase order. The ledger shows $95,000 already invoiced."
        ),
        facts={
            "po_value": 100_000,
            "invoiced_to_date": 95_000,
            "committed_not_invoiced": 0,
            "invoice_amount": 12_000,
            "cost_to_complete": 12_000,
        },
        available_tools=("po_checker",),
        required_tools=("po_checker",),
        diagnosis_id="po_overrun",
        recommendation_ids=("hold_invoice_for_review", "request_po_amendment"),
    ),
    "po-utilization": Scenario(
        scenario_id="po-utilization",
        task_type="po_utilization",
        project_name="Harbour Resilience Programme",
        prompt=(
            "A geotechnical investigation PO is 92% utilized with three months "
            "remaining. Assess whether the remaining authorization covers the "
            "forecast and recommend controls."
        ),
        facts={
            "po_value": 250_000,
            "invoiced_to_date": 210_000,
            "committed_not_invoiced": 20_000,
            "invoice_amount": 0,
            "cost_to_complete": 38_000,
            "cost_budget": 250_000,
            "cost_prior_forecast": 230_000,
            "cost_current_forecast": 268_000,
        },
        available_tools=("po_checker", "cost_analysis"),
        required_tools=("po_checker", "cost_analysis"),
        diagnosis_id="po_forecast_exceeds_balance",
        recommendation_ids=("reconcile_open_commitments", "request_po_amendment"),
    ),
    "commercial-risk": Scenario(
        scenario_id="commercial-risk",
        task_type="commercial_review",
        project_name="Central Station Interchange",
        prompt=(
            "Assess commercial health: forecast margin is 8% against a 12% "
            "control threshold, two change orders worth $180,000 are unapproved, "
            "and supplier costs are rising."
        ),
        facts={
            "revenue_budget": 12_000_000,
            "revenue_prior_forecast": 12_000_000,
            "revenue_current_forecast": 12_000_000,
            "cost_budget": 10_560_000,
            "cost_prior_forecast": 10_560_000,
            "cost_current_forecast": 11_040_000,
            "margin_threshold_pct": 12,
        },
        available_tools=("revenue_analysis", "cost_analysis", "margin_calculator"),
        required_tools=("revenue_analysis", "cost_analysis", "margin_calculator"),
        diagnosis_id="margin_and_unapproved_change_order_risk",
        recommendation_ids=(
            "escalate_change_order_approval",
            "mitigate_supplier_cost_growth",
        ),
    ),
    "resource-variance": Scenario(
        scenario_id="resource-variance",
        task_type="forecast_variance",
        project_name="Cedar Creek Flood Defence Scheme",
        prompt=(
            "Labor cost is forecast $64,000 over plan while revenue is unchanged. "
            "Investigate the variance and recommend follow-up."
        ),
        facts={
            "revenue_budget": 5_400_000,
            "revenue_prior_forecast": 5_400_000,
            "revenue_current_forecast": 5_400_000,
            "cost_budget": 4_200_000,
            "cost_prior_forecast": 4_200_000,
            "cost_current_forecast": 4_264_000,
        },
        available_tools=("revenue_analysis", "cost_analysis"),
        required_tools=("revenue_analysis", "cost_analysis"),
        diagnosis_id="senior_staffing_overrun",
        recommendation_ids=("rebalance_staffing_forecast", "review_remaining_effort"),
    ),
}


def get_scenario(scenario_id: str) -> Scenario:
    try:
        return SCENARIOS[scenario_id]
    except KeyError as error:
        available = ", ".join(sorted(SCENARIOS))
        raise KeyError(
            f"Unknown scenario {scenario_id!r}. Available scenarios: {available}"
        ) from error


SCENARIO_IDS = tuple(SCENARIOS)
