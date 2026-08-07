"""
Unit tests for the kiki_recommend tool.

All HTTP calls are mocked — no KIKI API server required.
Run with:  pytest tests/unit/test_kiki_tool.py
"""

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from tools.kiki_recommend.kiki_recommend_tool import kiki_recommend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _api_response(cell_type="GYR", x=17.0, y=18.0, z=10.0):
    """Build a mock requests.Response for a successful KIKI API call."""
    mock = MagicMock()
    mock.json.return_value = {
        "recommended_cell_type": cell_type,
        "x_rotation_deg": x,
        "y_rotation_deg": y,
        "z_rotation_deg": z,
    }
    mock.raise_for_status.return_value = None
    return mock


# ---------------------------------------------------------------------------
# Input validation — one test covers the pattern; boundary test confirms the
# exact limit is accepted
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_rejects_out_of_range_input(self):
        # body_wt=10 is below the 55kg minimum — representative of all field violations
        result = json.loads(kiki_recommend.invoke({"bone_preset": "normal", "body_wt": 10.0}))
        assert result["success"] is False
        assert "body_wt" in result["error"]

    def test_accepts_boundary_values(self):
        # Exactly at the lower boundary for body_wt and vol_frac — must not be rejected
        with patch("tools.kiki_recommend.kiki_recommend_tool.requests.post") as mock_post:
            mock_post.return_value = _api_response()
            result = json.loads(kiki_recommend.invoke({"bone_preset": "normal", "body_wt": 55.0, "vol_frac": 0.20}))
        assert result["success"] is True


# ---------------------------------------------------------------------------
# Successful API call
# ---------------------------------------------------------------------------

class TestSuccessfulCall:
    def test_returns_expected_fields(self):
        with patch("tools.kiki_recommend.kiki_recommend_tool.requests.post") as mock_post:
            mock_post.return_value = _api_response("BCC", 5.0, 10.0, 15.0)
            result = json.loads(kiki_recommend.invoke({"bone_preset": "normal", "body_wt": 75.0}))

        assert result["success"] is True
        assert result["recommended_cell_type"] == "BCC"
        assert result["x_rotation_deg"] == 5.0
        assert result["y_rotation_deg"] == 10.0
        assert result["z_rotation_deg"] == 15.0
        assert "vol_frac" in result
        assert "preset_used" in result

    def test_vol_frac_passed_through(self):
        # The KIKI API does not return vol_frac — the tool adds it from the input
        # so reconstruct_lattice can use it directly
        with patch("tools.kiki_recommend.kiki_recommend_tool.requests.post") as mock_post:
            mock_post.return_value = _api_response()
            result = json.loads(kiki_recommend.invoke({"bone_preset": "normal", "body_wt": 75.0, "vol_frac": 0.35}))

        assert result["vol_frac"] == pytest.approx(0.35)


# ---------------------------------------------------------------------------
# Preset resolution — free-text descriptions should resolve to a valid preset
# ---------------------------------------------------------------------------

class TestPresetResolution:
    def test_free_text_resolves_to_valid_preset(self):
        with patch("tools.kiki_recommend.kiki_recommend_tool.requests.post") as mock_post:
            mock_post.return_value = _api_response()
            result = json.loads(kiki_recommend.invoke({"bone_preset": "fragile bones", "body_wt": 65.0}))

        assert result["success"] is True
        assert result["preset_used"] in ("osteoporotic", "elderly", "normal", "athletic")

    def test_unknown_preset_falls_back_to_normal(self):
        with patch("tools.kiki_recommend.kiki_recommend_tool.requests.post") as mock_post:
            mock_post.return_value = _api_response()
            result = json.loads(kiki_recommend.invoke({"bone_preset": "xyzzy", "body_wt": 75.0}))

        assert result["success"] is True
        assert result["preset_used"] == "normal"


# ---------------------------------------------------------------------------
# API error handling — one test per distinct failure mode
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_connection_error_returns_failure(self):
        with patch("tools.kiki_recommend.kiki_recommend_tool.requests.post") as mock_post:
            mock_post.side_effect = requests.exceptions.ConnectionError()
            result = json.loads(kiki_recommend.invoke({"bone_preset": "normal", "body_wt": 75.0}))

        assert result["success"] is False
        assert "connect" in result["error"].lower()

    def test_timeout_returns_failure(self):
        with patch("tools.kiki_recommend.kiki_recommend_tool.requests.post") as mock_post:
            mock_post.side_effect = requests.exceptions.Timeout()
            result = json.loads(kiki_recommend.invoke({"bone_preset": "normal", "body_wt": 75.0}))

        assert result["success"] is False
        assert "30 seconds" in result["error"] or "respond" in result["error"].lower()

    def test_http_error_returns_failure(self):
        with patch("tools.kiki_recommend.kiki_recommend_tool.requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("422")
            mock_resp.json.return_value = {"detail": "Unprocessable entity"}
            mock_post.return_value = mock_resp
            result = json.loads(kiki_recommend.invoke({"bone_preset": "normal", "body_wt": 75.0}))

        assert result["success"] is False
        assert "rejected" in result["error"].lower() or "kiki api" in result["error"].lower()
