from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from scripts.token_optimization import evaluate_universal_proof as cli
from tests.unit.runtime.token_optimization.proofs.test_proof_evaluator import _fixture


def test_evaluate_only_is_network_free_and_prints_safe_summary(
    tmp_path, monkeypatch, capsys
) -> None:
    run, corpus, evaluation_config = _fixture()
    monkeypatch.setattr(
        cli,
        "load_universal_token_optimization_proof_config",
        lambda path: SimpleNamespace(),
    )
    monkeypatch.setattr(cli, "load_proof_corpus", lambda path: corpus)
    monkeypatch.setattr(cli, "load_evaluation_config", lambda path: evaluation_config)
    monkeypatch.setattr(cli, "load_universal_proof_run_result", lambda path: run)

    exit_code = cli.main(
        [
            "--proof-config",
            "synthetic-proof.toml",
            "--evaluation-config",
            "synthetic-evaluation.toml",
            "--run-result",
            "synthetic-run.json",
            "--output-dir",
            str(tmp_path),
            "--evaluation-id",
            "cli-fixed",
            "--evaluate-only",
        ]
    )

    assert exit_code == cli.EXIT_OK
    output = capsys.readouterr().out
    assert "evaluation_id=cli-fixed" in output
    assert "success=true" in output
    assert "synthetic" not in output.lower()


def test_cli_returns_required_evidence_exit_without_raw_output(
    tmp_path, monkeypatch, capsys
) -> None:
    run, corpus, evaluation_config = _fixture()
    run = replace(
        run,
        cases=(
            replace(
                run.cases[0],
                prefix_identity_evidence=replace(
                    run.cases[0].prefix_identity_evidence,
                    identity_available=False,
                    stable_prefix_identity=None,
                    tool_schema_hash=None,
                    identity_contract_version=None,
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        cli,
        "load_universal_token_optimization_proof_config",
        lambda path: SimpleNamespace(),
    )
    monkeypatch.setattr(cli, "load_proof_corpus", lambda path: corpus)
    monkeypatch.setattr(cli, "load_evaluation_config", lambda path: evaluation_config)
    monkeypatch.setattr(cli, "load_universal_proof_run_result", lambda path: run)

    exit_code = cli.main(
        [
            "--proof-config",
            "synthetic-proof.toml",
            "--evaluation-config",
            "synthetic-evaluation.toml",
            "--run-result",
            "synthetic-run.json",
            "--output-dir",
            str(tmp_path),
            "--evaluation-id",
            "cli-unavailable",
            "--evaluate-only",
        ]
    )

    assert exit_code == cli.EXIT_REQUIRED_EVIDENCE_UNAVAILABLE
    assert "identity" not in capsys.readouterr().out.lower()
