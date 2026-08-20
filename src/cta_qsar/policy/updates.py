"""Deterministic policy updates from observed marginal gains.

``record_iteration`` is called after every completed experiment iteration
(inside the orchestration loop). It is a pure function of the chosen
candidate's predicted signals, the realized primary-key improvement, and the
previous policy state. There is no randomness and no mutation -- every number
is recomputable from the plan trace, and the resulting event is appended to
the audit trail.
"""

from __future__ import annotations

from typing import Any

from cta_qsar.policy.state import (
    DEFAULT_SETTLE_DELTA,
    WEIGHT_BOUNDS,
    PolicyState,
)

IMPROVEMENT_KEY = "weight_improvement"
INFORMATION_KEY = "weight_information"
TRUST_KEY = "weight_trust"


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _quantile(values: list[float], q: float) -> float:
    """Linear-interpolation-free order-statistic quantile (deterministic)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def apply_update(
    state: PolicyState,
    *,
    predicted: dict[str, float],
    realized_improvement: float,
    learning_rate: float = 0.1,
    quantile: float = 0.5,
    window: int = 20,
    bounds: tuple[float, float] = WEIGHT_BOUNDS,
) -> PolicyState:
    """Update weights + settle-delta from one iteration, returning the state.

    Weight update: total predicted value is ``V = sum(predicted)``; the total
    prediction error is ``E = V - max(realized, 0)``. Each weight is adjusted
    by the error fraction attributable to its signal, multiplicatively and
    bounded to ``WEIGHT_BOUNDS``. When the realized improvement is negative or
    zero the error is taken in full, i.e. the over-estimating signal is
    dampened the most.

    Settle-delta update: append the realized improvement to the class history
    and set ``settle_delta`` to the configured quantile of the last ``window``
    observations (floored at a tiny epsilon so a zero-history class never
    triggers degenerate early stopping).
    """
    lo, hi = bounds
    lr = max(0.0, min(1.0, learning_rate))

    total_predicted = sum(max(0.0, v) for v in predicted.values()) or 1e-9
    total_error = total_predicted - max(realized_improvement, 0.0)
    realized_sign = 1.0 if realized_improvement >= 0 else -1.0

    new_weights: dict[str, float] = {}
    for key in ("weight_improvement", "weight_information", "weight_trust"):
        component = max(0.0, predicted.get(key, 0.0))
        fraction = component / total_predicted if total_predicted else 0.0
        attributable = total_error * fraction
        if component > 0 and total_error != 0:
            direction = 1.0 if attributable < 0 else -1.0
            delta = lr * direction * max(abs(attributable), abs(total_error) / 4.0)
        else:
            delta = 0.0
        base = state.weights.get(key, 1.0)
        new_weights[key] = _clamp(base * (1.0 + delta * realized_sign), lo, hi)

    state.improvement_history = (state.improvement_history + [realized_improvement])[-window:]
    if len(state.improvement_history) >= 3:
        state.settle_delta = max(_quantile(state.improvement_history, quantile), 1e-4)
    else:
        state.settle_delta = DEFAULT_SETTLE_DELTA if state.settle_delta is None else state.settle_delta

    state.weights = new_weights
    state.updates_applied += 1
    state.last_event = {
        "type": "policy_update",
        "predicted": {k: round(v, 4) for k, v in predicted.items()},
        "realized_improvement": round(realized_improvement, 4),
        "weights": {k: round(v, 4) for k, v in new_weights.items()},
        "settle_delta": round(state.settle_delta, 6) if state.settle_delta else None,
        "updates_applied": state.updates_applied,
    }
    return state


def event_dict(state: PolicyState) -> dict[str, Any] | None:
    return state.last_event