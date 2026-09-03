"""Pydantic schemas for FPL API payloads.

Field aliases match the raw FPL API keys so models can be built directly via
`Model.model_validate(raw_dict)`. Several numeric fields come back from the API
as strings (or empty strings for "no value yet"), so validators normalize those
before the numeric type coercion runs.
"""
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from src import config


def _to_optional_float(value):
    if value in (None, ""):
        return None
    return float(value)


def _to_float(value):
    if value in (None, ""):
        return 0.0
    return float(value)


class Team(BaseModel):
    id: int
    name: str
    short_name: str
    strength_attack_home: int
    strength_attack_away: int
    strength_defence_home: int
    strength_defence_away: int


class Player(BaseModel):
    model_config = {"populate_by_name": True}

    id: int
    web_name: str
    team_id: int = Field(alias="team")
    element_type: int
    now_cost: int
    selected_by_percent: float
    form: float
    total_points: int
    ep_next: Optional[float] = None
    xg: float = Field(alias="expected_goals", default=0.0)
    xa: float = Field(alias="expected_assists", default=0.0)
    xgi: float = Field(alias="expected_goal_involvements", default=0.0)
    xg_per_90: float = Field(alias="expected_goals_per_90", default=0.0)
    xa_per_90: float = Field(alias="expected_assists_per_90", default=0.0)
    saves_per_90: float = 0.0
    defensive_contribution_per_90: float = 0.0
    starts_per_90: float = 0.0
    starts: int = 0  # cumulative starts so far this season -- used for xMins lineup-security baseline
    chance_of_playing_next_round: Optional[int] = None  # 0-100, or null meaning "no doubt" (treated as 100)
    penalties_order: Optional[int] = None  # 1 = primary penalty taker, null = not on the list
    corners_order: Optional[int] = Field(alias="corners_and_indirect_freekicks_order", default=None)
    transfers_in_event: int = 0  # transfers in *today/this event* -- Price Change Sentinel input
    transfers_out_event: int = 0  # transfers out *today/this event* -- Price Change Sentinel input
    expected_goals_conceded_per_90: float = 0.0  # per-player xGC -- see calculate_positional_xp's
    # use of this vs. the team-level FDR proxy (_team_xga_proxy)
    cost_change_event: int = 0  # today's ALREADY-REALIZED price move, not a prediction
    price_change_percent: float = 0.0  # FPL's own "progress toward next price change" signal --
    # see compute_price_change_alerts
    status: str
    news: str = ""

    @field_validator(
        "selected_by_percent", "form", "xg", "xa", "xgi",
        "xg_per_90", "xa_per_90", "saves_per_90", "defensive_contribution_per_90", "starts_per_90",
        "expected_goals_conceded_per_90", "price_change_percent",
        mode="before",
    )
    @classmethod
    def _empty_to_zero(cls, v):
        return _to_float(v)

    @field_validator("cost_change_event", mode="before")
    @classmethod
    def _empty_to_zero_cost_change(cls, v):
        return int(_to_float(v))

    @field_validator("transfers_in_event", "transfers_out_event", mode="before")
    @classmethod
    def _empty_to_zero_transfers(cls, v):
        return int(_to_float(v))

    @field_validator("starts", mode="before")
    @classmethod
    def _empty_to_zero_int(cls, v):
        return int(_to_float(v))

    @field_validator("ep_next", "chance_of_playing_next_round", "penalties_order", "corners_order", mode="before")
    @classmethod
    def _empty_to_none(cls, v):
        return _to_optional_float(v)

    @property
    def cost_millions(self) -> float:
        """Actual squad price in GBP millions, e.g. now_cost=100 -> 10.0."""
        return self.now_cost / config.PRICE_DIVISOR


class Gameweek(BaseModel):
    id: int
    name: str
    deadline_time: Optional[str] = None
    is_current: bool
    is_next: bool
    finished: bool


class Fixture(BaseModel):
    id: int
    event: Optional[int] = None
    team_h: int
    team_a: int
    team_h_difficulty: Optional[int] = None
    team_a_difficulty: Optional[int] = None
    finished: bool


class SquadPick(BaseModel):
    """One row of a manager's squad for a given gameweek.

    manager_id/event are contextual (not present in the raw per-pick payload),
    so this is built explicitly rather than via model_validate(raw_pick) alone.
    """
    model_config = {"populate_by_name": True}

    manager_id: int
    event: int
    player_id: int = Field(alias="element")
    position_in_squad: int = Field(alias="position")
    is_captain: bool
    is_vice: bool = Field(alias="is_vice_captain")
