import pytest

from app.data.translate import (
    DEFAULT_COEFFICIENTS,
    StatCoefficient,
    UnknownLeagueError,
    translate,
)


def test_nba_is_identity() -> None:
    raw = {"ts_pct": 0.60, "usg_pct": 0.28, "stl_pct": 0.018}
    assert translate(raw, "NBA") == raw


def test_ncaa_usage_compresses() -> None:
    out = translate({"usg_pct": 0.30}, "NCAA")
    assert out["usg_pct"] == pytest.approx(0.30 * 0.83)


def test_ncaa_steal_rate_drops_more_than_usage() -> None:
    """Steals translate worse than usage — NBA athletes are harder to strip."""
    out = translate({"usg_pct": 1.0, "stl_pct": 1.0}, "NCAA")
    assert out["stl_pct"] < out["usg_pct"]


def test_unknown_stat_passes_through_unchanged() -> None:
    out = translate({"completely_made_up_stat": 0.42}, "NCAA")
    assert out["completely_made_up_stat"] == 0.42


def test_unknown_league_raises() -> None:
    with pytest.raises(UnknownLeagueError, match="EuroLeague"):
        translate({"ts_pct": 0.60}, "EuroLeague")


def test_custom_coefficients_override_default() -> None:
    custom = {"FakeLeague": {"ts_pct": StatCoefficient(multiplier=2.0, intercept=0.05)}}
    out = translate({"ts_pct": 0.50}, "FakeLeague", coefficients=custom)
    assert out["ts_pct"] == pytest.approx(0.50 * 2.0 + 0.05)


def test_multiple_stats_translate_independently() -> None:
    raw = {"ts_pct": 0.60, "usg_pct": 0.30, "ast_pct": 0.25}
    out = translate(raw, "NCAA")
    coefs = DEFAULT_COEFFICIENTS["NCAA"]
    assert out["ts_pct"] == pytest.approx(0.60 * coefs["ts_pct"].multiplier)
    assert out["usg_pct"] == pytest.approx(0.30 * coefs["usg_pct"].multiplier)
    assert out["ast_pct"] == pytest.approx(0.25 * coefs["ast_pct"].multiplier)


def test_intercept_applied() -> None:
    custom = {"L": {"x": StatCoefficient(multiplier=1.0, intercept=10.0)}}
    out = translate({"x": 5.0}, "L", coefficients=custom)
    assert out["x"] == pytest.approx(15.0)


def test_does_not_mutate_input() -> None:
    raw = {"usg_pct": 0.30}
    raw_copy = dict(raw)
    translate(raw, "NCAA")
    assert raw == raw_copy


def test_empty_stats_returns_empty() -> None:
    assert translate({}, "NCAA") == {}


def test_default_table_includes_nba_and_ncaa() -> None:
    assert "NBA" in DEFAULT_COEFFICIENTS
    assert "NCAA" in DEFAULT_COEFFICIENTS
