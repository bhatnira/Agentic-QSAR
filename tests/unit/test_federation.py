"""Unit tests for the federation layer (compete + accumulate + evolve)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from cta_qsar.federation.cards import (
    AgentOutcome,
    StrategyCard,
    build_leaderboard,
    default_cards,
    primary_key_for,
)
from cta_qsar.federation.core import ChallengeSession


class TestCards:
    def test_primary_keys(self):
        assert primary_key_for("regression") == "rmse"
        assert primary_key_for("binary") == "roc_auc"
        assert primary_key_for("multiclass") == "mcc"
        assert primary_key_for("multitask_binary") == "roc_auc"

    def test_default_cards_deterministic(self):
        cards = default_cards()
        names = [c.name for c in cards]
        assert names == ["gate:aggressive", "fast:nosearch", "gate:none", "evolve"]
        assert cards[2].trust_gate is False
        assert cards[1].search is False
        assert cards[3].adaptive_policy is True

    def test_nvidia_card_gated(self):
        assert all(c.llm_provider != "nvidia" for c in default_cards())
        assert any(c.name == "llm:nvidia" for c in default_cards(with_nvidia=True))

    def test_card_to_dict_round_trip(self):
        card = StrategyCard(name="x", trust_gate=False)
        assert card.to_dict()["trust_gate"] is False


class TestLeaderboard:
    def _outcome(self, card, value, primary="rmse", failed=""):
        return AgentOutcome(card=card, seed=0, primary=primary, primary_value=value, failed=failed)

    def test_ranking_higher_is_better_for_auc(self):
        rows = build_leaderboard(
            [self._outcome("a", 0.7, "roc_auc"), self._outcome("b", 0.9, "roc_auc")], "roc_auc"
        )
        assert [r["card"] for r in rows] == ["b", "a"]
        assert rows[0]["rank"] == 1

    def test_ranking_lower_is_better_for_rmse(self):
        rows = build_leaderboard(
            [self._outcome("a", 1.5), self._outcome("b", 0.8)], "rmse"
        )
        assert rows[0]["card"] == "b"

    def test_failed_agents_ranked_last_without_rank(self):
        rows = build_leaderboard(
            [
                self._outcome("winner", 0.5),
                self._outcome("broken", 0.0, failed="boom"),
            ],
            "rmse",
        )
        assert rows[1]["card"] == "broken"
        assert rows[1]["rank"] is None
        assert rows[0]["rank"] == 1


class TestChallengeSession:
    def _fake_agent(self, *, values=None, failed_cards=()):
        values = values or {}

        def agent_fn(**kwargs):
            card = kwargs["card"]
            primary = kwargs["primary"]
            failed = card.name in failed_cards
            return {
                "card": card.name,
                "seed": kwargs["seed"],
                "primary": primary,
                "primary_value": values.get(card.name, 1.0),
                "best_model": "morgan+ridge[random]",
                "n_experiments": 3,
                "runtime_seconds": 10.0,
                "run_id": f"{card.name}-run",
                "failed": "boom" if failed else "",
            }

        return agent_fn

    def _df(self):
        return pd.DataFrame({"smiles": ["CCO" * 3, "CCN" * 3], "y": [1.0, 2.0]})

    def test_challenge_accumulates_evidence_and_reports_diff(self, tmp_path):
        session = ChallengeSession(
            evidence_path=tmp_path / "evidence.jsonl",
            agent_fn=self._fake_agent(
                values={"gate:aggressive": 0.8, "fast:nosearch": 1.2, "gate:none": 1.1, "evolve": 1.0}
            ),
        )
        report = session.run_challenge(
            df=self._df(),
            dataset_name="mini",
            task_type="regression",
            n_rows=2,
            seed=0,
            cards=default_cards(),
            primary="rmse",
            inprogress_path=tmp_path / "inprogress.csv",
        )
        assert report.dataset_class == "regression|tiny"
        assert report.winner["card"] == "gate:aggressive"
        assert len(report.leaderboard) == 4
        assert report.kg_after["n_edges"] >= 4
        added = {e for e in report.kg_diff["added_edges"]}
        assert any("gate:aggressive" in e for e in added)
        # evidence persisted: a fresh session reads it back
        again = ChallengeSession(
            evidence_path=tmp_path / "evidence.jsonl",
            agent_fn=self._fake_agent(),
        )
        assert "gate:aggressive" in {e[1] for e in again.kg.edge_set()}

    def test_failed_agent_still_finishes_challenge(self, tmp_path):
        session = ChallengeSession(
            evidence_path=tmp_path / "evidence.jsonl",
            agent_fn=self._fake_agent(failed_cards={"gate:none"}),
        )
        report = session.run_challenge(
            df=self._df(),
            dataset_name="mini",
            task_type="binary",
            n_rows=2,
            seed=1,
            cards=default_cards()[:3],
            primary="roc_auc",
            inprogress_path=tmp_path / "inprogress.csv",
        )
        failed = [r for r in report.leaderboard if r["failed"]]
        assert [r["card"] for r in failed] == ["gate:none"]
        assert report.winner is not None

    def test_inprogress_file_empty_without_path(self, tmp_path):
        session = ChallengeSession(
            evidence_path=tmp_path / "evidence.jsonl",
            agent_fn=self._fake_agent(),
        )
        session.run_challenge(
            df=self._df(),
            dataset_name="mini",
            task_type="regression",
            n_rows=2,
            seed=0,
            cards=default_cards()[:1],
            primary="rmse",
            inprogress_path=None,
        )
        assert (tmp_path / "evidence.jsonl").exists() is False