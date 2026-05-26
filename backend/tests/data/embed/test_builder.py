import pytest

from app.data.embed import STAT_DIMENSIONS, VECTOR_DIM, build_stat_profile
from app.data.translate import DEFAULT_COEFFICIENTS


def _full_nba_stats() -> dict[str, float]:
    """Synthetic stats with every schema dim populated — used to verify the
    'fully populated' path returns a vector with zero imputed dims."""
    return {dim.name: 1.0 + i * 0.01 for i, dim in enumerate(STAT_DIMENSIONS)}


def test_full_stats_no_imputation() -> None:
    profile = build_stat_profile(_full_nba_stats(), "NBA")
    assert len(profile.vector) == VECTOR_DIM
    assert profile.imputed_dim_indices == []


def test_missing_stat_is_imputed_with_default() -> None:
    stats = _full_nba_stats()
    del stats["per"]
    profile = build_stat_profile(stats, "NBA")
    per_index = next(i for i, d in enumerate(STAT_DIMENSIONS) if d.name == "per")
    assert per_index in profile.imputed_dim_indices
    assert profile.vector[per_index] == 0.0


def test_extra_keys_in_input_are_ignored() -> None:
    stats = _full_nba_stats()
    stats["totally_not_a_stat"] = 42.0
    profile = build_stat_profile(stats, "NBA")
    assert profile.imputed_dim_indices == []
    # extra key not represented in the vector — vector dim is fixed
    assert len(profile.vector) == VECTOR_DIM


def test_ncaa_stats_pass_through_translation() -> None:
    """Stats that have NCAA translation coefficients should arrive scaled."""
    raw = {"usg_pct": 0.30}  # missing every other stat — imputed
    profile = build_stat_profile(raw, "NCAA")
    usg_index = next(i for i, d in enumerate(STAT_DIMENSIONS) if d.name == "usg_pct")
    expected = 0.30 * DEFAULT_COEFFICIENTS["NCAA"]["usg_pct"].multiplier
    assert profile.vector[usg_index] == pytest.approx(expected)
    # usg was supplied so not imputed
    assert usg_index not in profile.imputed_dim_indices


def test_nba_stats_are_identity() -> None:
    profile = build_stat_profile({"ts_pct": 0.60}, "NBA")
    ts_index = next(i for i, d in enumerate(STAT_DIMENSIONS) if d.name == "ts_pct")
    assert profile.vector[ts_index] == pytest.approx(0.60)


def test_empty_stats_full_imputation() -> None:
    profile = build_stat_profile({}, "NBA")
    assert len(profile.imputed_dim_indices) == VECTOR_DIM
    assert all(v == 0.0 for v in profile.vector)


def test_profile_records_source_league() -> None:
    profile = build_stat_profile({}, "NCAA")
    assert profile.source_league == "NCAA"


def test_unknown_league_raises_via_translation() -> None:
    from app.data.translate import UnknownLeagueError

    with pytest.raises(UnknownLeagueError):
        build_stat_profile({"ts_pct": 0.60}, "Atlantis Pro League")


def test_stat_profile_validates_vector_length() -> None:
    from app.data.embed import StatProfile

    with pytest.raises(ValueError, match="50 dims"):
        StatProfile(vector=[0.0] * 10)
