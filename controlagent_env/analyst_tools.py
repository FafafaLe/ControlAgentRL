from __future__ import annotations

from collections.abc import Callable, Mapping


class InvalidToolInputError(ValueError):
    """Raised when a scenario does not contain inputs required by an analyst tool."""


def _read(facts: Mapping[str, float], *keys: str) -> tuple[float, ...]:
    missing = [key for key in keys if key not in facts]
    if missing:
        raise InvalidToolInputError(
            f"Analyst tool is missing required scenario facts: {', '.join(missing)}"
        )
    return tuple(float(facts[key]) for key in keys)


def _money(value: float) -> str:
    return f"-${abs(value):,.0f}" if value < 0 else f"${value:,.0f}"


def revenue_analysis(facts: Mapping[str, float]) -> str:
    """Compare revenue budget and successive estimate-at-completion forecasts."""
    budget, prior, current = _read(
        facts, "revenue_budget", "revenue_prior_forecast", "revenue_current_forecast"
    )
    return (
        f"Revenue EAC is {_money(current)} versus a control budget of {_money(budget)} "
        f"({_money(current - budget)} variance). Change from prior forecast "
        f"({_money(prior)}) is {_money(current - prior)}."
    )


def cost_analysis(facts: Mapping[str, float]) -> str:
    """Compare cost budget and successive estimate-at-completion forecasts."""
    budget, prior, current = _read(
        facts, "cost_budget", "cost_prior_forecast", "cost_current_forecast"
    )
    return (
        f"Cost EAC is {_money(current)} versus a control budget of {_money(budget)} "
        f"({_money(current - budget)} variance). Change from prior forecast "
        f"({_money(prior)}) is {_money(current - prior)}."
    )


def margin_calculator(facts: Mapping[str, float]) -> str:
    """Calculate contribution and contribution margin from revenue and cost EAC."""
    revenue_prior, revenue_current, cost_prior, cost_current = _read(
        facts,
        "revenue_prior_forecast",
        "revenue_current_forecast",
        "cost_prior_forecast",
        "cost_current_forecast",
    )
    if revenue_prior <= 0 or revenue_current <= 0:
        raise InvalidToolInputError("Revenue forecasts must be positive to calculate margin.")
    prior_contribution = revenue_prior - cost_prior
    current_contribution = revenue_current - cost_current
    prior_margin = prior_contribution / revenue_prior * 100
    current_margin = current_contribution / revenue_current * 100
    return (
        f"Contribution changed from {_money(prior_contribution)} "
        f"({prior_margin:.1f}% margin) to {_money(current_contribution)} "
        f"({current_margin:.1f}% margin), a {_money(current_contribution - prior_contribution)} "
        "change."
    )


def recovery_factor_calculator(facts: Mapping[str, float]) -> str:
    """Calculate fee-revenue recovery factor as labor revenue divided by labor cost."""
    labor_revenue, labor_cost, prior_revenue, prior_cost = _read(
        facts,
        "labor_revenue",
        "labor_cost",
        "prior_labor_revenue",
        "prior_labor_cost",
    )
    if labor_cost <= 0 or prior_cost <= 0:
        raise InvalidToolInputError("Labor cost must be positive to calculate recovery factor.")
    current_factor = labor_revenue / labor_cost
    prior_factor = prior_revenue / prior_cost
    return (
        f"Recovery factor (labor fee revenue / direct labor cost) is "
        f"{current_factor:.2f}, versus {prior_factor:.2f} previously "
        f"({current_factor - prior_factor:+.2f} change)."
    )


def po_checker(facts: Mapping[str, float]) -> str:
    """Check committed PO balance against an invoice and estimate-to-complete."""
    po_value, invoiced, committed, invoice, cost_to_complete = _read(
        facts,
        "po_value",
        "invoiced_to_date",
        "committed_not_invoiced",
        "invoice_amount",
        "cost_to_complete",
    )
    if po_value <= 0:
        raise InvalidToolInputError("PO value must be positive.")
    if min(invoiced, committed, invoice, cost_to_complete) < 0:
        raise InvalidToolInputError(
            "Invoiced, committed, invoice, and cost-to-complete amounts must not be negative."
        )
    balance_before_invoice = po_value - invoiced - committed
    balance_after_invoice = balance_before_invoice - invoice
    forecast_gap = balance_before_invoice - cost_to_complete
    utilization = (invoiced + committed) / po_value * 100
    flags = []
    if invoice > balance_before_invoice:
        flags.append(f"invoice exceeds available authorization by {_money(invoice - balance_before_invoice)}")
    if forecast_gap < 0:
        flags.append(f"forecast cost-to-complete exceeds available balance by {_money(-forecast_gap)}")
    status = "; ".join(flags) if flags else "no current authorization or forecast shortfall identified"
    return (
        f"PO is {utilization:.1f}% utilized ({_money(invoiced)} invoiced, "
        f"{_money(committed)} committed-not-invoiced). Balance before proposed "
        f"invoice is {_money(balance_before_invoice)}; after invoice it is "
        f"{_money(balance_after_invoice)}. Forecast cost-to-complete is "
        f"{_money(cost_to_complete)}. Control check: {status}."
    )


TOOL_FUNCTIONS: dict[str, Callable[[Mapping[str, float]], str]] = {
    "revenue_analysis": revenue_analysis,
    "cost_analysis": cost_analysis,
    "margin_calculator": margin_calculator,
    "recovery_factor_calculator": recovery_factor_calculator,
    "po_checker": po_checker,
}


def execute_analyst_tool(tool_name: str, facts: Mapping[str, float]) -> str:
    try:
        tool = TOOL_FUNCTIONS[tool_name]
    except KeyError as error:
        available = ", ".join(sorted(TOOL_FUNCTIONS))
        raise KeyError(f"Unknown analyst tool {tool_name!r}. Available tools: {available}") from error
    return tool(facts)
