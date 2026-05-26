"""Build the 50-d stat profile vector from a player-season stats dict.

Pipeline:
1. Translate raw stats from `source_league` to NBA-equivalent space.
2. Map translated stats into the canonical dimension order (`STAT_DIMENSIONS`).
3. Impute missing stats with `dim.imputation_default` (currently 0.0) and
   record which slots were imputed.

This is the only place where a stat dict becomes a vector. If you need a
different vector representation (e.g. position-mean imputation in Phase 7),
add it here rather than embedding ad-hoc elsewhere.
"""

from dataclasses import dataclass, field

from app.data.embed.schema import STAT_DIMENSIONS, VECTOR_DIM
from app.data.translate import translate
from app.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class StatProfile:
    """A 50-d vector plus provenance metadata about how it was constructed.

    `imputed_dim_indices` lists indices into STAT_DIMENSIONS where the raw input
    didn't supply a value and the imputation default was used instead. Downstream
    (Pinecone upsert) can choose to drop high-imputation rows from the index, or
    include them with a flag in metadata for transparency.
    """

    vector: list[float]
    imputed_dim_indices: list[int] = field(default_factory=list)
    source_league: str = "NBA"

    def __post_init__(self) -> None:
        if len(self.vector) != VECTOR_DIM:
            raise ValueError(
                f"stat profile vector must have {VECTOR_DIM} dims, got {len(self.vector)}"
            )


def build_stat_profile(stats: dict[str, float], source_league: str) -> StatProfile:
    """Build a stat profile vector from a player-season stats dict.

    `stats` may be a superset of the schema — extra keys are ignored. Missing
    keys are imputed with the dim's default. Raw stats are translated to
    NBA-equivalent space before placement.
    """
    translated = translate(stats, source_league)

    vector: list[float] = []
    imputed: list[int] = []
    for i, dim in enumerate(STAT_DIMENSIONS):
        if dim.name in translated:
            vector.append(float(translated[dim.name]))
        else:
            vector.append(dim.imputation_default)
            imputed.append(i)

    if imputed:
        logger.debug(
            "build_stat_profile league=%s imputed=%d/%d",
            source_league,
            len(imputed),
            VECTOR_DIM,
        )

    return StatProfile(
        vector=vector,
        imputed_dim_indices=imputed,
        source_league=source_league,
    )
