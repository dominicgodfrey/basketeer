"""The 50-dimensional stat profile schema.

Each `StatDim` in `STAT_DIMENSIONS` is one slot in the player-season vector.
The list order is the canonical dimension order — never reorder without
re-embedding everything in Pinecone, since position is identity.

Design notes:
- Stats here are play-style signals. Demographics (age, height, draft pick) and
  identity (player_id, season, league, position) live in *metadata* on the
  Pinecone record, not in the vector, so similarity scoring isn't biased by
  age or build and so filtering (by position, season, current_fa) happens
  Pinecone-side without a re-query.
- All rate stats are per-100 possessions where applicable, percentages
  otherwise. Mixing per-game with per-100 in the same vector creates ugly
  cross-stat magnitudes; sticking with per-100 keeps stats comparable for
  cosine.
- For dims marked `available_from_season=N`, player-seasons before N have the
  dim imputed (currently to 0.0; flagged via `StatProfile.imputed_dim_indices`).
  Phase 7 will switch to position-mean imputation.

Sources column reflects what the ingestion layer (collaborator's track) needs
to fetch:

- `nba_api.LeagueDashPlayerStats` (LDPS) — most rate + advanced stats
- `nba_api.LeagueDashPlayerClutch` — clutch stats
- `nba_api.LeagueHustleStatsPlayer` — hustle (2016+)
- `nba_api.LeaguePlayerTrackingStats` — tracking (2014+)
- `nba_api.LeagueDashTeamStats` — team pace context
- Basketball-Reference scrape — historical (pre-1996) advanced metrics
"""

from dataclasses import dataclass
from enum import Enum


class StatCategory(str, Enum):
    BOX_RATE = "box_rate"
    SHOOTING = "shooting"
    PLAYMAKING = "playmaking"
    REBOUNDING = "rebounding"
    DEFENSE = "defense"
    OFFENSE = "offense"
    COMPOSITE = "composite"
    CONTEXT = "context"
    HUSTLE = "hustle"
    CLUTCH = "clutch"


@dataclass(frozen=True, slots=True)
class StatDim:
    """One dimension of the stat profile vector."""

    name: str
    category: StatCategory
    source: str
    imputation_default: float = 0.0
    available_from_season: int | None = None


# Order is the canonical dimension order. DO NOT reorder.
STAT_DIMENSIONS: list[StatDim] = [
    # A. Box-score rate stats (per-100 possessions)
    StatDim("pts_per_100", StatCategory.BOX_RATE, "LDPS Per100"),
    StatDim("reb_per_100", StatCategory.BOX_RATE, "LDPS Per100"),
    StatDim("ast_per_100", StatCategory.BOX_RATE, "LDPS Per100"),
    StatDim("stl_per_100", StatCategory.BOX_RATE, "LDPS Per100"),
    StatDim("blk_per_100", StatCategory.BOX_RATE, "LDPS Per100"),
    StatDim("tov_per_100", StatCategory.BOX_RATE, "LDPS Per100"),
    StatDim("pf_per_100", StatCategory.BOX_RATE, "LDPS Per100"),
    StatDim("fga_per_100", StatCategory.BOX_RATE, "LDPS Per100"),
    StatDim("fta_per_100", StatCategory.BOX_RATE, "LDPS Per100"),
    # B. Shooting & efficiency
    StatDim("ts_pct", StatCategory.SHOOTING, "LDPS Advanced"),
    StatDim("efg_pct", StatCategory.SHOOTING, "LDPS Advanced"),
    StatDim("fg_pct", StatCategory.SHOOTING, "LDPS Base"),
    StatDim("three_pt_pct", StatCategory.SHOOTING, "LDPS Base"),
    StatDim("ft_pct", StatCategory.SHOOTING, "LDPS Base"),
    StatDim("three_par", StatCategory.SHOOTING, "LDPS Advanced (3PA/FGA)"),
    StatDim("ftr", StatCategory.SHOOTING, "LDPS Advanced (FTA/FGA)"),
    StatDim(
        "pct_fga_at_rim",
        StatCategory.SHOOTING,
        "ShotChartDetail aggregated to zones",
        available_from_season=1996,
    ),
    StatDim(
        "pct_fga_mid_range",
        StatCategory.SHOOTING,
        "ShotChartDetail aggregated to zones",
        available_from_season=1996,
    ),
    StatDim(
        "pct_fga_three",
        StatCategory.SHOOTING,
        "ShotChartDetail aggregated to zones",
        available_from_season=1996,
    ),
    # C. Playmaking & turnovers
    StatDim("usg_pct", StatCategory.PLAYMAKING, "LDPS Advanced"),
    StatDim("ast_pct", StatCategory.PLAYMAKING, "LDPS Advanced"),
    StatDim("tov_pct", StatCategory.PLAYMAKING, "LDPS Advanced"),
    StatDim("ast_to_tov", StatCategory.PLAYMAKING, "Derived from ast_pct/tov_pct"),
    StatDim("ast_to_usg", StatCategory.PLAYMAKING, "Derived from ast_pct/usg_pct"),
    # D. Rebounding
    StatDim("orb_pct", StatCategory.REBOUNDING, "LDPS Advanced"),
    StatDim("drb_pct", StatCategory.REBOUNDING, "LDPS Advanced"),
    StatDim("trb_pct", StatCategory.REBOUNDING, "LDPS Advanced"),
    # E. Defense (rate + composite)
    StatDim("stl_pct", StatCategory.DEFENSE, "LDPS Advanced"),
    StatDim("blk_pct", StatCategory.DEFENSE, "LDPS Advanced"),
    StatDim("drtg", StatCategory.DEFENSE, "LDPS Advanced"),
    StatDim("dbpm", StatCategory.DEFENSE, "BBR / nba_stats Advanced"),
    StatDim("dws", StatCategory.DEFENSE, "BBR / nba_stats Advanced"),
    StatDim(
        "deflections_per_100",
        StatCategory.HUSTLE,
        "LeagueHustleStatsPlayer",
        available_from_season=2016,
    ),
    StatDim(
        "contested_2s_per_100",
        StatCategory.HUSTLE,
        "LeagueHustleStatsPlayer",
        available_from_season=2016,
    ),
    StatDim(
        "contested_3s_per_100",
        StatCategory.HUSTLE,
        "LeagueHustleStatsPlayer",
        available_from_season=2016,
    ),
    # F. Offensive composite
    StatDim("ortg", StatCategory.OFFENSE, "LDPS Advanced"),
    StatDim("obpm", StatCategory.OFFENSE, "BBR / nba_stats Advanced"),
    StatDim("ows", StatCategory.OFFENSE, "BBR / nba_stats Advanced"),
    StatDim("per", StatCategory.OFFENSE, "BBR / nba_stats Advanced"),
    # G. Overall composite
    StatDim("bpm", StatCategory.COMPOSITE, "BBR / nba_stats Advanced"),
    StatDim("vorp", StatCategory.COMPOSITE, "BBR / nba_stats Advanced"),
    StatDim("ws_per_48", StatCategory.COMPOSITE, "BBR / nba_stats Advanced"),
    # H. Context (playing time + team pace)
    StatDim("min_pct", StatCategory.CONTEXT, "LDPS Advanced (MIN_PCT)"),
    StatDim("min_per_game", StatCategory.CONTEXT, "LDPS Base"),
    StatDim("team_pace", StatCategory.CONTEXT, "LeagueDashTeamStats"),
    # I. Clutch & playoff + tracking
    StatDim(
        "playoff_min_pct",
        StatCategory.CLUTCH,
        "LDPS season_type=Playoffs",
    ),
    StatDim(
        "clutch_pts_per_100",
        StatCategory.CLUTCH,
        "LeagueDashPlayerClutch",
        available_from_season=1996,
    ),
    StatDim(
        "screen_assists_per_100",
        StatCategory.HUSTLE,
        "LeagueHustleStatsPlayer",
        available_from_season=2016,
    ),
    StatDim(
        "box_outs_per_100",
        StatCategory.HUSTLE,
        "LeagueHustleStatsPlayer",
        available_from_season=2017,
    ),
    StatDim(
        "drives_per_100",
        StatCategory.CONTEXT,
        "LeaguePlayerTrackingStats",
        available_from_season=2014,
    ),
]

VECTOR_DIM = len(STAT_DIMENSIONS)
assert VECTOR_DIM == 50, f"stat profile must be 50-d, got {VECTOR_DIM}"


# Identity + demographic fields stored as Pinecone *metadata* (not in the vector).
# Filtering happens on these server-side; similarity scoring ignores them.
METADATA_FIELDS: dict[str, str] = {
    "player_id": "Internal player id (str).",
    "season": "Season start year (int). 2024 means 2024-25.",
    "league": "'NBA' | 'NCAA' | 'EuroLeague' | ...",
    "position": "'PG' | 'SG' | 'SF' | 'PF' | 'C' or combined like 'SG-SF'.",
    "team_id": "Internal team id (str), nullable for FA-only seasons.",
    "is_current_fa": "True if the player is currently a free agent.",
    "age": "Age during the season (int).",
    "height_inches": "Listed height (int).",
    "wingspan_inches": "Combine wingspan if available, else null.",
    "experience_years": "Pro seasons completed before this season (int).",
    "draft_pick": "Overall pick number, 0 if undrafted, null if unknown.",
}
