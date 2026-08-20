"""Unit tests for the self-improving planner policy (principled adaptation)."""

from __future__ import annotations

import json

import pytest

from cta_qsar.core.interfaces import ExperimentCandidate
from cta_qsar.orchestration.policies import compute_marginal_gain, evaluate_stopping
from cta_qsar.policy.state import DEFAULT_SETTLE_DELTA, DEFAULT_WEIGHTS, PolicyState, PolicyStore
from cta_qsar.policy.updates import apply_update


class TestPolicyState:
    def test_default_weights_and_settle_delta(self):
        state = PolicyState("regression|small")
        assert state.weights == DEFAULT_WEIGHTS
        assert state.settle_delta is None
        assert state.updates_applied == 0

    def test_effective_settle_delta_defaults(self, tmp_path):
        store = PolicyStore.load(tmp_path / "policy.jsonl")
        assert store.effective_settle_delta("regression|small") == DEFAULT_SETTLE_DELTA

    def test_load_save_round_trip(self, tmp_path):
        path = tmp_path / "policy.jsonl"
        store = PolicyStore.load(path)
        state = store.get("regression|small")
        state.weights = {"weight_improvement": 1.5, "weight_information": 0.8, "weight_trust": 1.0}
        state.settle_delta = 0.012
        state.updates_applied = 3
        store.update("regression|small", state)
        store.save()

        reloaded = PolicyStore.load(path)
        state2 = reloaded.get("regression|small")
        assert state2.weights == state.weights
        assert state2.settle_delta == 0.012
        assert state2.updates_applied == 3

    def test_corrupt_line_skipped(self, tmp_path):
        path = tmp_path / "policy.jsonl"
        path.write_text('{"dataset_class": "regression|small"}\nnot-json\n')
        store = PolicyStore.load(path)
        assert store.get("regression|small").weights == DEFAULT_WEIGHTS


class TestApplyUpdate:
    def test_update_is_deterministic(self):
        a = PolicyState("regression|small")
        b = PolicyState("regression|small")
        kwargs = {"predicted": {"weight_improvement": 0.5, "weight_information": 0.2, "weight_trust": 0.1},
                  "realized_improvement": 0.3}
        apply_update(a, **kwargs)
        apply_update(b, **kwargs)
        assert a.weights == b.weights
        assert a.last_event == b.last_event

    def test_overestimation_damps_improvement_weight(self):
        state = PolicyState("regression|small")
        apply_update(state, predicted={"weight_improvement": 2.0, "weight_information": 0.0, "weight_trust": 0.0},
                     realized_improvement=0.0)
        assert state.weights["weight_improvement"] < 1.0
        assert state.weights["weight_information"] == 1.0
        assert state.weights["weight_trust"] == 1.0

    def test_underestimation_raises_weight(self):
        state = PolicyState("regression|small")
        apply_update(state, predicted={"weight_improvement": 0.1, "weight_information": 0.0, "weight_trust": 0.0},
                     realized_improvement=1.0)
        assert state.weights["weight_improvement"] > 1.0

    def test_weights_never_leave_bounds(self):
        state = PolicyState("regression|small")
        for _ in range(50):
            apply_update(state, predicted={"weight_improvement": 5.0, "weight_information": 1.0, "weight_trust": 1.0},
                         realized_improvement=0.0, learning_rate=0.5)
        for key in DEFAULT_WEIGHTS:
            assert 0.5 <= state.weights[key] <= 2.0

    def test_settle_delta_learns_quantile_after_three_observations(self):
        state = PolicyState("regression|small")
        for i, gain in enumerate([0.01, 0.02, 0.03, 0.04]):
            apply_update(state, predicted={"weight_improvement": 0.5, "weight_information": 0.1, "weight_trust": 0.1},
                         realized_improvement=gain)
            if i < 2:
                assert state.settle_delta == DEFAULT_SETTLE_DELTA
        assert state.settle_delta is not None
        assert state.settle_delta >= 0.02
        assert len(state.improvement_history) == 4

    def test_event_is_auditable(self):
        state = PolicyState("regression|small")
        apply_update(state, predicted={"weight_improvement": 0.5, "weight_information": 0.2, "weight_trust": 0.1},
                     realized_improvement=0.3)
        event = state.last_event
        assert event["type"] == "policy_update"
        assert "weights" in event and "realized_improvement" in event and "settle_delta" in event
        assert event["updates_applied"] == 1


class TestPlannerIntegration:
    def test_compute_utility_applies_weights(self):
        cand = ExperimentCandidate(
            expected_improvement=0.8,
            expected_information_gain=0.1,
            expected_trustworthiness_gain=0.1,
            compute_cost=2.0,
        )
        plain = cand.compute_utility()
        weighted = cand.compute_utility(
            {"weight_improvement": 2.0, "weight_information": 1.0, "weight_trust": 1.0}
        )
        assert plain == pytest.approx(0.5)
        assert weighted == pytest.approx(0.9)

    def test_compute_utility_default_weights_match_frozen(self):
        cand = ExperimentCandidate(expected_improvement=1.0, expected_information_gain=0.5,
                                   expected_trustworthiness_gain=0.25, compute_cost=1.0)
        assert cand.compute_utility() == cand.compute_utility(
            {"weight_improvement": 1.0, "weight_information": 1.0, "weight_trust": 1.0}
        )


class TestStoppingIntegration:
    def _exp(self, metrics):
        return {"result": "completed", "metrics": metrics}

    def test_compute_marginal_gain_positive(self):
        exps = [self._exp({"rmse": 1.0}), self._exp({"rmse": 0.8})]
        gain, key = compute_marginal_gain(exps)
        assert (key, pytest.approx(gain)) == ("rmse", 0.2)

    def test_compute_marginal_gain_roc_auc(self):
        exps = [self._exp({"roc_auc": 0.7}), self._exp({"roc_auc": 0.75})]
        gain, key = compute_marginal_gain(exps)
        assert (key, pytest.approx(gain)) == ("roc_auc", 0.05)

    def test_compute_marginal_gain_insufficient(self):
        assert compute_marginal_gain([self._exp({"rmse": 1.0})]) == (None, None)
        assert compute_marginal_gain([self._exp({"rmse": 1.0}), self._exp({"accuracy": 0.9})]) == (None, None)

    def test_evaluate_stopping_uses_learned_settle_delta(self):
        budget = {"max_experiments": 12, "max_minutes": 30.0, "elapsed_minutes": 1.0}
        exps = [self._exp({"rmse": 1.0}), self._exp({"rmse": 0.995})]
        reasons_loose = evaluate_stopping(budget=budget, plan_round=1, experiments=exps,
                                          no_improvement_rounds=0, settle_delta=0.01)
        reasons_tight = evaluate_stopping(budget=budget, plan_round=1, experiments=exps,
                                          no_improvement_rounds=0, settle_delta=0.001)
        assert any("settle-delta" in r for r in reasons_loose)
        assert not any("negligible" in r for r in reasons_tight)

    def test_default_settle_delta_preserves_frozen_behavior(self):
        budget = {"max_experiments": 12, "max_minutes": 30.0, "elapsed_minutes": 1.0}
        exps = [self._exp({"rmse": 1.0}), self._exp({"rmse": 0.999})]
        reasons = evaluate_stopping(budget=budget, plan_round=1, experiments=exps,
                                    no_improvement_rounds=0)
        assert any("negligible" in r for r in reasons)


class TestSerializationConsistency:
    def test_policy_state_json_dump_load_keeps_weights(self):
        state = PolicyState("binary|medium")
        state.weights = {"weight_improvement": 0.6, "weight_information": 1.2, "weight_trust": 0.9}
        data = json.loads(json.dumps(state.to_dict()))
        restored = PolicyState.from_dict(data)
        assert restored.weights == state.weights
        assert restored.improvement_history == state.improvement_history


class TestAdaptPolicyNode:
    def _state(self, tmp_path, *, adaptive=True):
        from cta_qsar.core.config import build_config

        config = build_config()
        config.policy.adaptive = adaptive
        config.policy.policy_state_path = str(tmp_path / "policy_state.jsonl")
        config.knowledge.evidence_path = str(tmp_path / "evidence.jsonl")
        return {
            "config": config,
            "output_dir": str(tmp_path),
            "run_id": "policy-test",
            "profile": {"n_rows": 800},
            "endpoint": {"task_type": "regression"},
            "selected_candidate": {
                "expected_improvement": 0.6,
                "expected_information_gain": 0.2,
                "expected_trustworthiness_gain": 0.1,
            },
            "experiments": [
                {"result": "completed", "metrics": {"rmse": 1.0}},
                {"result": "completed", "metrics": {"rmse": 0.7}},
            ],
        }

    def test_adaptive_run_writes_store_and_trace(self, tmp_path):
        from cta_qsar.orchestration.nodes import adapt_policy

        state = self._state(tmp_path, adaptive=True)
        result = adapt_policy(state)
        assert result is state

        store = PolicyStore.load(tmp_path / "policy_state.jsonl")
        pstate = store.get("regression|small")
        assert pstate.updates_applied == 1
        assert pstate.weights["weight_improvement"] < 1.0  # over-predicted 0.6 vs 0.3 realized

        trace = (tmp_path / "policy-test" / "plan_trace.jsonl").read_text()
        assert '"type": "policy_update"' in trace
        assert "regression|small" in trace

    def test_frozen_run_is_noop(self, tmp_path):
        from cta_qsar.orchestration.nodes import adapt_policy

        state = self._state(tmp_path, adaptive=False)
        result = adapt_policy(state)
        assert result is state
        assert not (tmp_path / "policy_state.jsonl").exists()

    def test_policy_state_accumulates_across_runs(self, tmp_path):
        from cta_qsar.orchestration.nodes import adapt_policy

        state = self._state(tmp_path, adaptive=True)
        adapt_policy(state)
        adapt_policy(state)
        pstate = PolicyStore.load(tmp_path / "policy_state.jsonl").get("regression|small")
        assert pstate.updates_applied == 2
        assert len(pstate.improvement_history) == 2
