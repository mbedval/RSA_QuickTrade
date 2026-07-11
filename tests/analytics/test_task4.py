"""Unit tests for Task 4: Reporting, Comparison & Integration.

Verifies HTML report compilation, strategy comparison dashboard generation,
JSON/CSV exporters, and CLI execution.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import json
import tempfile
from pathlib import Path
import pytest

from bsa_quicktrade.analytics.models import TradeRecord
from bsa_quicktrade.analytics.report_generator import ReportGenerator
from bsa_quicktrade.analytics.exporter import export_to_csv, export_to_json
from bsa_quicktrade.analytics.cli import main as cli_main


@pytest.fixture
def base_time():
    return datetime(2026, 1, 1, 10, 0, 0)


@pytest.fixture
def mock_trades(base_time):
    # Simple list of mock trades
    r1 = TradeRecord(
        trade_id="T1",
        strategy_name="StratA",
        module_name="ModA",
        signal_name="SigA",
        entry_date=base_time,
        exit_date=base_time + timedelta(days=2),
        entry_price=100.0,
        exit_price=110.0,
        quantity=10.0,
        capital_used=1000.0,
        profit=100.0,
        profit_pct=10.0,
        r_multiple=1.0,
        market_regime="Bull"
    )
    r2 = TradeRecord(
        trade_id="T2",
        strategy_name="StratA",
        module_name="ModA",
        signal_name="SigA",
        entry_date=base_time + timedelta(days=2),
        exit_date=base_time + timedelta(days=4),
        entry_price=100.0,
        exit_price=95.0,
        quantity=10.0,
        capital_used=1000.0,
        profit=-50.0,
        profit_pct=-5.0,
        r_multiple=-0.5,
        market_regime="Bear"
    )
    return [r1, r2]


# ── Report Generator Tests ──────────────────────────────────────────────────

def test_report_generation(mock_trades):
    with tempfile.TemporaryDirectory() as tmpdir:
        generator = ReportGenerator(output_dir=tmpdir)
        
        # Single Report
        report_path = generator.generate_html_report(
            records=mock_trades, strategy_name="Test Strategy", starting_capital=10000.0, file_name="test_report.html"
        )
        assert isinstance(report_path, Path)
        assert report_path.exists()
        
        content = report_path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "Test Strategy" in content
        assert "Performance Metrics Overview" in content


def test_comparison_generation(mock_trades):
    with tempfile.TemporaryDirectory() as tmpdir:
        generator = ReportGenerator(output_dir=tmpdir)
        
        strategies = {
            "StratA": mock_trades,
            "StratB": mock_trades  # duplicate for test
        }
        
        comp_path = generator.generate_comparison_report(
            strategies_dict=strategies, starting_capital=10000.0, file_name="test_comparison.html"
        )
        assert isinstance(comp_path, Path)
        assert comp_path.exists()
        
        content = comp_path.read_text(encoding="utf-8")
        assert "Strategy Comparison Dashboard" in content
        assert "StratA" in content
        assert "StratB" in content


# ── Exporter Tests ──────────────────────────────────────────────────────────

def test_exporters(mock_trades):
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "metrics.json"
        csv_path = Path(tmpdir) / "metrics.csv"
        
        export_to_json(mock_trades, json_path, starting_capital=10000.0)
        export_to_csv(mock_trades, csv_path, starting_capital=10000.0)
        
        assert json_path.exists()
        assert csv_path.exists()
        
        # Verify JSON
        with open(json_path, "r") as f:
            data = json.load(f)
            assert "performance" in data
            assert data["performance"]["total_trades"] == 2
            assert data["performance"]["net_profit"] == 50.0

        # Verify CSV
        csv_content = csv_path.read_text(encoding="utf-8")
        assert "performance_total_trades" in csv_content
        assert "2" in csv_content


# ── CLI Tests ───────────────────────────────────────────────────────────────

def test_cli_single_log(mock_trades):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save a mock log file
        log_path = Path(tmpdir) / "mock_log.csv"
        # We can use our save_trade_log to save
        from bsa_quicktrade.analytics.trade_log import save_trade_log
        save_trade_log(mock_trades, log_path)
        
        # Paths for exports
        json_out = Path(tmpdir) / "out.json"
        csv_out = Path(tmpdir) / "out.csv"
        
        # Execute CLI
        args = [
            "--log", str(log_path),
            "--strategy", "CLI Strat",
            "--capital", "10000",
            "--output-dir", tmpdir,
            "--json", str(json_out),
            "--csv", str(csv_out)
        ]
        
        exit_code = cli_main(args)
        assert exit_code == 0
        
        # Verify outputs exist
        assert (Path(tmpdir) / "analytics_report.html").exists()
        assert json_out.exists()
        assert csv_out.exists()


def test_cli_comparison(mock_trades):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save mock logs
        log1 = Path(tmpdir) / "log1.csv"
        log2 = Path(tmpdir) / "log2.csv"
        from bsa_quicktrade.analytics.trade_log import save_trade_log
        save_trade_log(mock_trades, log1)
        save_trade_log(mock_trades, log2)
        
        # Execute CLI comparison
        args = [
            "--compare", f"Model1:{log1}",
            "--compare", f"Model2:{log2}",
            "--capital", "10000",
            "--output-dir", tmpdir
        ]
        
        exit_code = cli_main(args)
        assert exit_code == 0
        
        # Verify comparison report exists
        assert (Path(tmpdir) / "strategy_comparison.html").exists()


def test_strategy_health(mock_trades):
    from bsa_quicktrade.analytics.health import calculate_strategy_health
    
    # Calculate health score
    health = calculate_strategy_health(mock_trades, starting_capital=10000.0)
    
    # Assert return object properties
    assert 0.0 <= health.score <= 100.0
    assert health.status in ["PASS", "WARNING", "REJECT"]
    assert len(health.reasons) > 0
    assert health.recommendation in ["Highly suitable for live deployment. Start with standard sizing.",
                                     "Deploy with caution. Reduce position sizing (e.g. 50%) and monitor closely.",
                                     "Do not deploy."]

