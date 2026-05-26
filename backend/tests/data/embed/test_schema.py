from app.data.embed import (
    METADATA_FIELDS,
    STAT_DIMENSIONS,
    VECTOR_DIM,
    StatCategory,
)


def test_vector_dim_is_50() -> None:
    assert VECTOR_DIM == 50
    assert len(STAT_DIMENSIONS) == 50


def test_dimension_names_are_unique() -> None:
    names = [d.name for d in STAT_DIMENSIONS]
    assert len(set(names)) == len(names), "duplicate dimension names"


def test_every_dim_has_a_documented_source() -> None:
    for dim in STAT_DIMENSIONS:
        assert dim.source, f"{dim.name} has no source documented"


def test_every_dim_has_a_valid_category() -> None:
    for dim in STAT_DIMENSIONS:
        assert isinstance(dim.category, StatCategory)


def test_hustle_stats_are_marked_as_unavailable_pre_2016() -> None:
    hustle_dims = [d for d in STAT_DIMENSIONS if d.category == StatCategory.HUSTLE]
    assert hustle_dims, "expected at least one hustle dim"
    for d in hustle_dims:
        assert d.available_from_season is not None and d.available_from_season >= 2016


def test_drives_marked_unavailable_pre_2014() -> None:
    drives = next(d for d in STAT_DIMENSIONS if d.name == "drives_per_100")
    assert drives.available_from_season == 2014


def test_demographic_fields_are_in_metadata_not_vector() -> None:
    """Sanity check: age / height / position stay out of the vector."""
    vector_names = {d.name for d in STAT_DIMENSIONS}
    for demographic in ("age", "height_inches", "wingspan_inches", "position", "draft_pick"):
        assert demographic not in vector_names
        assert demographic in METADATA_FIELDS
