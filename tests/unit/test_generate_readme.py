"""Unit tests for generate_readme.py module.

Tests cover the generate_results_table function including row ordering,
column formatting, CRPS N/A display for point-forecast baselines, and
4 decimal place rounding.

Related modules:
    - generate_readme.py: The module under test
"""

import pytest

from generate_readme import generate_results_table


# Sample metrics dictionary matching the expected input format
SAMPLE_METRICS = {
    "Naive": {"MAE": 0.5432, "MSE": 0.4321, "MASE": 1.2345},
    "ARIMA": {"MAE": 0.4321, "MSE": 0.3210, "MASE": 1.1234},
    "Prophet": {"MAE": 0.3987, "MSE": 0.2876, "MASE": 1.0567},
    "PatchTST (zero-shot)": {"MAE": 0.3456, "MSE": 0.2345, "MASE": 0.9876, "CRPS": 0.1234},
    "PatchTST (fine-tuned)": {"MAE": 0.2987, "MSE": 0.1876, "MASE": 0.8765, "CRPS": 0.0987},
}


class TestGenerateResultsTable:
    """Tests for generate_results_table."""

    def test_returns_string(self) -> None:
        """Function returns a string."""
        result = generate_results_table(SAMPLE_METRICS)
        assert isinstance(result, str)

    def test_header_columns(self) -> None:
        """Table header contains Method, MAE, MSE, MASE, CRPS columns."""
        result = generate_results_table(SAMPLE_METRICS)
        header = result.split("\n")[0]
        assert "Method" in header
        assert "MAE" in header
        assert "MSE" in header
        assert "MASE" in header
        assert "CRPS" in header

    def test_separator_row(self) -> None:
        """Table has a separator row after the header."""
        result = generate_results_table(SAMPLE_METRICS)
        lines = result.split("\n")
        separator = lines[1]
        assert separator.startswith("|")
        assert "-" in separator

    def test_row_order(self) -> None:
        """Rows appear in exact order: Naive, ARIMA, Prophet, PatchTST (zero-shot), PatchTST (fine-tuned)."""
        result = generate_results_table(SAMPLE_METRICS)
        lines = result.split("\n")
        data_lines = lines[2:]  # Skip header and separator
        assert len(data_lines) == 5
        assert "Naive" in data_lines[0]
        assert "ARIMA" in data_lines[1]
        assert "Prophet" in data_lines[2]
        assert "PatchTST (zero-shot)" in data_lines[3]
        assert "PatchTST (fine-tuned)" in data_lines[4]

    def test_crps_na_for_point_forecast_methods(self) -> None:
        """CRPS displays 'N/A' for Naive, ARIMA, and Prophet."""
        result = generate_results_table(SAMPLE_METRICS)
        lines = result.split("\n")
        data_lines = lines[2:]
        # Naive, ARIMA, Prophet should have N/A
        for line in data_lines[:3]:
            # Split by | and check the CRPS column (last data column)
            cells = [c.strip() for c in line.split("|") if c.strip()]
            assert cells[-1] == "N/A"

    def test_crps_value_for_patchtst(self) -> None:
        """CRPS shows numeric value for PatchTST methods."""
        result = generate_results_table(SAMPLE_METRICS)
        lines = result.split("\n")
        data_lines = lines[2:]
        # PatchTST zero-shot
        assert "0.1234" in data_lines[3]
        # PatchTST fine-tuned
        assert "0.0987" in data_lines[4]

    def test_four_decimal_places(self) -> None:
        """All numeric values are rounded to exactly 4 decimal places."""
        metrics = {
            "Naive": {"MAE": 0.123456789, "MSE": 0.5, "MASE": 1.0},
            "ARIMA": {"MAE": 0.4, "MSE": 0.3, "MASE": 1.1},
            "Prophet": {"MAE": 0.3, "MSE": 0.2, "MASE": 1.0},
            "PatchTST (zero-shot)": {"MAE": 0.2, "MSE": 0.1, "MASE": 0.9, "CRPS": 0.05},
            "PatchTST (fine-tuned)": {"MAE": 0.1, "MSE": 0.05, "MASE": 0.8, "CRPS": 0.03},
        }
        result = generate_results_table(metrics)
        # 0.123456789 should be rounded to 0.1235
        assert "0.1235" in result
        # 0.5 should display as 0.5000
        assert "0.5000" in result
        # 1.0 should display as 1.0000
        assert "1.0000" in result

    def test_markdown_table_format(self) -> None:
        """Output is valid markdown table format with pipe separators."""
        result = generate_results_table(SAMPLE_METRICS)
        lines = result.split("\n")
        for line in lines:
            assert line.startswith("|")
            assert line.endswith("|")

    def test_total_line_count(self) -> None:
        """Table has 7 lines: header + separator + 5 data rows."""
        result = generate_results_table(SAMPLE_METRICS)
        lines = result.split("\n")
        assert len(lines) == 7
