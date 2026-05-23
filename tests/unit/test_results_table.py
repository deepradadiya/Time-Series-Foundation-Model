"""Unit tests for evaluation/results_table.py module.

Tests cover table formatting, row ordering, CRPS display for baselines,
inference time formatting, and JSON export with rounding.

Related modules:
    - evaluation/results_table.py: The module under test
"""

import json
import os
import tempfile

import pytest

from evaluation.results_table import (
    _format_inference_time,
    format_results_table,
    print_results_table,
    save_results_json,
)


# Sample results dictionary matching the design spec format
SAMPLE_RESULTS = {
    "Naive": {"mae": 0.1234, "mse": 0.0567, "mase": 1.0000, "crps": None, "inference_time": 0.01},
    "ARIMA": {"mae": 0.1100, "mse": 0.0450, "mase": 0.8900, "crps": None, "inference_time": 45.23},
    "Prophet": {"mae": 0.1050, "mse": 0.0420, "mase": 0.8500, "crps": None, "inference_time": 120.56},
    "PatchTST (zero-shot)": {"mae": 0.0950, "mse": 0.0380, "mase": 0.7700, "crps": 0.0650, "inference_time": 2.34},
    "PatchTST (fine-tuned)": {"mae": 0.0750, "mse": 0.0280, "mase": 0.6100, "crps": 0.0450, "inference_time": 2.45},
}


class TestFormatInferenceTime:
    """Tests for the _format_inference_time helper."""

    def test_sub_millisecond(self) -> None:
        """Times < 0.001s display as '<1ms'."""
        assert _format_inference_time(0.0001) == "<1ms"
        assert _format_inference_time(0.0009) == "<1ms"
        assert _format_inference_time(0.0) == "<1ms"

    def test_millisecond_range(self) -> None:
        """Times >= 0.001s and < 1s display as '~Xms'."""
        assert _format_inference_time(0.001) == "~1ms"
        assert _format_inference_time(0.234) == "~234ms"
        assert _format_inference_time(0.999) == "~999ms"

    def test_second_range(self) -> None:
        """Times >= 1s display as '~Xs'."""
        assert _format_inference_time(1.0) == "~1s"
        assert _format_inference_time(45.23) == "~45s"
        assert _format_inference_time(120.56) == "~121s"


class TestFormatResultsTable:
    """Tests for format_results_table."""

    def test_row_order(self) -> None:
        """Rows appear in canonical order: Naive, ARIMA, Prophet, PatchTST (zero-shot), PatchTST (fine-tuned)."""
        table = format_results_table(SAMPLE_RESULTS)
        lines = table.strip().split("\n")
        # Skip header and separator (first 2 lines)
        data_lines = lines[2:]
        assert len(data_lines) == 5
        assert data_lines[0].startswith("Naive")
        assert data_lines[1].startswith("ARIMA")
        assert data_lines[2].startswith("Prophet")
        assert data_lines[3].startswith("PatchTST (zero-shot)")
        assert data_lines[4].startswith("PatchTST (fine-tuned)")

    def test_crps_na_for_baselines(self) -> None:
        """CRPS shows 'N/A' for point-forecast baselines."""
        table = format_results_table(SAMPLE_RESULTS)
        lines = table.strip().split("\n")
        data_lines = lines[2:]
        # Naive, ARIMA, Prophet should have N/A
        for line in data_lines[:3]:
            assert "N/A" in line
        # PatchTST models should NOT have N/A
        for line in data_lines[3:]:
            assert "N/A" not in line

    def test_crps_value_for_patchtst(self) -> None:
        """CRPS shows numeric value for PatchTST models."""
        table = format_results_table(SAMPLE_RESULTS)
        lines = table.strip().split("\n")
        data_lines = lines[2:]
        assert "0.0650" in data_lines[3]
        assert "0.0450" in data_lines[4]

    def test_header_present(self) -> None:
        """Table includes header with all column names."""
        table = format_results_table(SAMPLE_RESULTS)
        first_line = table.split("\n")[0]
        assert "Model" in first_line
        assert "MAE" in first_line
        assert "MSE" in first_line
        assert "MASE" in first_line
        assert "CRPS" in first_line
        assert "Inference_Time" in first_line

    def test_inference_time_formatting(self) -> None:
        """Inference time uses human-readable units."""
        table = format_results_table(SAMPLE_RESULTS)
        # Naive: 0.01s -> ~10ms
        assert "~10ms" in table
        # ARIMA: 45.23s -> ~45s
        assert "~45s" in table
        # Prophet: 120.56s -> ~121s
        assert "~121s" in table

    def test_partial_results(self) -> None:
        """Table handles partial results (not all models present)."""
        partial = {
            "Naive": SAMPLE_RESULTS["Naive"],
            "PatchTST (fine-tuned)": SAMPLE_RESULTS["PatchTST (fine-tuned)"],
        }
        table = format_results_table(partial)
        lines = table.strip().split("\n")
        data_lines = lines[2:]
        assert len(data_lines) == 2
        assert data_lines[0].startswith("Naive")
        assert data_lines[1].startswith("PatchTST (fine-tuned)")

    def test_metrics_four_decimal_places(self) -> None:
        """Metric values are displayed with 4 decimal places."""
        table = format_results_table(SAMPLE_RESULTS)
        assert "0.1234" in table
        assert "0.0567" in table
        assert "1.0000" in table


class TestPrintResultsTable:
    """Tests for print_results_table."""

    def test_prints_to_stdout(self, capsys) -> None:
        """print_results_table outputs the formatted table to stdout."""
        print_results_table(SAMPLE_RESULTS)
        captured = capsys.readouterr()
        assert "Naive" in captured.out
        assert "ARIMA" in captured.out
        assert "PatchTST (fine-tuned)" in captured.out


class TestSaveResultsJson:
    """Tests for save_results_json."""

    def test_creates_output_directory(self) -> None:
        """Creates output directory if it does not exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "nested", "dir", "results.json")
            save_results_json(SAMPLE_RESULTS, output_path)
            assert os.path.exists(output_path)

    def test_json_structure(self) -> None:
        """JSON has model names as keys with metric dictionaries as values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.json")
            save_results_json(SAMPLE_RESULTS, output_path)

            with open(output_path) as f:
                data = json.load(f)

            assert set(data.keys()) == set(SAMPLE_RESULTS.keys())
            for model_name in SAMPLE_RESULTS:
                assert "mae" in data[model_name]
                assert "mse" in data[model_name]
                assert "mase" in data[model_name]
                assert "crps" in data[model_name]
                assert "inference_time" in data[model_name]

    def test_values_rounded_to_4_decimal_places(self) -> None:
        """Numeric values are rounded to 4 decimal places."""
        results_with_long_decimals = {
            "Naive": {
                "mae": 0.123456789,
                "mse": 0.056789012,
                "mase": 1.000012345,
                "crps": None,
                "inference_time": 0.0123456,
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.json")
            save_results_json(results_with_long_decimals, output_path)

            with open(output_path) as f:
                data = json.load(f)

            assert data["Naive"]["mae"] == 0.1235
            assert data["Naive"]["mse"] == 0.0568
            assert data["Naive"]["mase"] == 1.0
            assert data["Naive"]["crps"] is None
            assert data["Naive"]["inference_time"] == 0.0123

    def test_crps_none_preserved(self) -> None:
        """CRPS None values are preserved as null in JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.json")
            save_results_json(SAMPLE_RESULTS, output_path)

            with open(output_path) as f:
                data = json.load(f)

            assert data["Naive"]["crps"] is None
            assert data["ARIMA"]["crps"] is None
            assert data["Prophet"]["crps"] is None
            assert data["PatchTST (zero-shot)"]["crps"] == 0.065
            assert data["PatchTST (fine-tuned)"]["crps"] == 0.045
