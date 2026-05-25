"""Per-league translation coefficients for converting raw stats to NBA-equivalent space.

Every non-NBA stat passes through these coefficients before being embedded. The
linear form (`nba = raw * multiplier + intercept`) is deliberately simple — it's
the model Pelton/Vashro popularized and it gets us most of the way for college
and Euroleague translations. Refinement (per-position adjustments, sample-size
weighting, separate per-stat regressions) lands in a later phase as we collect
our own data.

The default values below are illustrative placeholders informed by published
NCAA-to-NBA work; they should not be treated as final. Phase 7 (cost
optimization / coefficient refinement) will fit them against our own NCAA→NBA
matched samples.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StatCoefficient:
    """Linear translation of one stat from a source league to NBA-equivalent.

    `nba_equivalent = raw * multiplier + intercept`
    """

    multiplier: float
    intercept: float = 0.0


# Mapping from stat name → coefficient for a single source league.
LeagueCoefficients = dict[str, StatCoefficient]


DEFAULT_COEFFICIENTS: dict[str, LeagueCoefficients] = {
    # NBA is identity — defining it explicitly so callers always pass through
    # the translator regardless of source, which keeps the call site uniform.
    "NBA": {},
    "NCAA": {
        # Efficiency drops slightly when moving from college-D to NBA-D.
        "ts_pct": StatCoefficient(multiplier=0.96),
        "efg_pct": StatCoefficient(multiplier=0.96),
        # Usage compresses — top NCAA usage rarely translates 1:1 to NBA.
        "usg_pct": StatCoefficient(multiplier=0.83),
        # Playmaking translates better than usage but still degrades.
        "ast_pct": StatCoefficient(multiplier=0.90),
        # Turnovers rise vs NBA defenses.
        "tov_pct": StatCoefficient(multiplier=1.05),
        # Steal/block rates drop substantially — NBA athletes harder to strip / block.
        "stl_pct": StatCoefficient(multiplier=0.75),
        "blk_pct": StatCoefficient(multiplier=0.70),
        # Rebounding scales by position; using a rough average here.
        "orb_pct": StatCoefficient(multiplier=0.85),
        "drb_pct": StatCoefficient(multiplier=0.90),
        # 3-point percentage drops with the longer NBA line.
        "three_pct": StatCoefficient(multiplier=0.93),
        "three_par": StatCoefficient(multiplier=0.95),
        # Free-throw rate drops a bit against more disciplined NBA defenses.
        "ftr": StatCoefficient(multiplier=0.92),
    },
    # Future: "EuroLeague", "G-League", "NCAA-Women" (different project), etc.
}
