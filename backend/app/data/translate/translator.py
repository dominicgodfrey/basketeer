"""Apply per-league translation coefficients to a stat record.

This is the only place where cross-league stat conversion happens. If you find
yourself doing it elsewhere, you've broken the contract documented in CLAUDE.md.
"""

from app.data.translate.coefficients import DEFAULT_COEFFICIENTS, LeagueCoefficients


class UnknownLeagueError(ValueError):
    """Raised when a source league has no coefficient entry — surfacing this
    explicitly is preferable to silently treating it as identity."""


def translate(
    stats: dict[str, float],
    source_league: str,
    coefficients: dict[str, LeagueCoefficients] | None = None,
) -> dict[str, float]:
    """Convert `stats` from `source_league` into NBA-equivalent space.

    - NBA → NBA is identity (no-op).
    - Stats present in the league's coefficient table are transformed linearly.
    - Stats absent from the table pass through unchanged.
    - Unknown source leagues raise `UnknownLeagueError` to fail loud rather than
      silently embed raw cross-league stats.

    Pass a custom `coefficients` mapping for tests or experiments.
    """
    table = coefficients if coefficients is not None else DEFAULT_COEFFICIENTS
    if source_league not in table:
        raise UnknownLeagueError(
            f"No translation coefficients for league {source_league!r}; "
            f"known: {sorted(table.keys())}"
        )

    league_coefs = table[source_league]
    return {
        stat_name: (
            raw_value * league_coefs[stat_name].multiplier + league_coefs[stat_name].intercept
            if stat_name in league_coefs
            else raw_value
        )
        for stat_name, raw_value in stats.items()
    }
