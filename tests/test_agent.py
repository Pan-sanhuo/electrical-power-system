from pathlib import Path

from pfagent import LLMConfig, PowerFlowAgent


def test_agent_runs_example(tmp_path):
    root = Path(__file__).resolve().parents[1]
    case_file = root / "examples" / "case3_q_limit.py"
    agent = PowerFlowAgent(llm_config=LLMConfig(provider="off"), max_rounds=8)
    report = agent.run(case_file, tmp_path / "run")
    assert report.attempts
    assert report.final_report_path is not None
    assert report.final_report_path.exists()


def test_inspect_bad_data():
    root = Path(__file__).resolve().parents[1]
    case_file = root / "examples" / "case3_bad_data.py"
    agent = PowerFlowAgent(llm_config=LLMConfig(provider="off"))
    _summary, issues, _llm = agent.inspect(case_file)
    codes = {issue["code"] for issue in issues}
    assert "NO_REF" in codes
    assert "ZERO_IMPEDANCE" in codes


def test_clean_case_is_engineering_feasible(tmp_path):
    root = Path(__file__).resolve().parents[1]
    agent = PowerFlowAgent(llm_config=LLMConfig(provider="off"))
    report = agent.run(root / "examples" / "case9_demo.py", tmp_path / "case9")
    assert report.success
    assert report.attempts[-1].feasible
    assert report.final_case_path.name == "final_feasible_case.py"
    assert report.attempts[-1].diagnostics.get("available") is True


def test_no_auto_repair_does_not_mislabel_infeasible_solution(tmp_path):
    root = Path(__file__).resolve().parents[1]
    agent = PowerFlowAgent(llm_config=LLMConfig(provider="off"))
    report = agent.run(root / "examples" / "case3_q_limit.py", tmp_path / "no_repair", auto_repair=False)
    assert report.solver_converged
    assert not report.success
    assert report.final_case_path.name == "last_converged_but_infeasible_case.py"


def test_load_search_uses_original_baseline(tmp_path):
    root = Path(__file__).resolve().parents[1]
    agent = PowerFlowAgent(llm_config=LLMConfig(provider="off"), max_rounds=20)
    report = agent.run(root / "examples" / "case3_q_limit.py", tmp_path / "q_limit")
    assert report.success
    assert report.attempts[-1].name == "scale_load_0.70_nr"
    assert abs(report.attempts[-1].result["bus"][:, 2].sum() - 105.0) < 1e-6


def test_max_rounds_does_not_record_unverified_actions(tmp_path):
    root = Path(__file__).resolve().parents[1]
    agent = PowerFlowAgent(llm_config=LLMConfig(provider="off"), max_rounds=2)
    report = agent.run(root / "examples" / "case3_q_limit.py", tmp_path / "limited")
    assert len(report.attempts) == 2
    assert len(report.repairs) == 1
