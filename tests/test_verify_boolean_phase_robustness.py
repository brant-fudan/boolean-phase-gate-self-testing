"""Regression tests for the exact robustness-certificate contract."""

from fractions import Fraction as F
import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import verify_boolean_phase_robustness as VERIFIER  # noqa: E402


def test_local_margin_uses_no_numeric_sign_decision(monkeypatch):
    """Every local sign choice must remain in Q(sqrt(2))."""

    def reject_numeric_eval(*_args, **_kwargs):
        raise AssertionError("numeric evaluation entered an exact proof")

    monkeypatch.setattr(VERIFIER.sp, "N", reject_numeric_eval)
    for name in ("lp", "hp", "hm"):
        VERIFIER.local_margin(name)


def test_listed_boxes_and_local_squares_cover_each_domain():
    """The certificate must check the full unit square, not only its boxes."""

    local_boxes = {
        "lp": [(F(63, 64), F(1), F(63, 64), F(1))],
        "lm": [],
        "hp": [(F(0), F(1, 64), F(0), F(1, 64))],
        "hm": [(F(0), F(1, 64), F(0), F(1, 64))],
    }
    for name in ("lp", "lm", "hp", "hm"):
        assert VERIFIER.verify_cover(
            VERIFIER.BOXES[name] + local_boxes[name]
        )


def test_slope_one_needs_only_one_full_square_per_polynomial():
    """A shorter uniform bound should have a one-box exact certificate."""

    assert VERIFIER.verify_slope_one_single_box()
