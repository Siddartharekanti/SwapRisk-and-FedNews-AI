from swapscribe.cli import run_risk, run_commentary


def test_end_to_end_cli_path(sample_data):
    # Smoke integration uses generated demo in project data/ from CLI.
    report = run_risk('demo')
    note = run_commentary('demo', 'inflation treasury yields')
    assert 'rms_scenario_pnl_usd_after' in report
    assert 'market_observation_date' in note
