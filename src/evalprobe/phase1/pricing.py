from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int | None
    cached_input_tokens: int | None
    cache_write_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    total_tokens: int | None


@dataclass(frozen=True, slots=True)
class Pricing:
    input_per_million_usd: float
    cached_input_per_million_usd: float
    cache_write_per_million_usd: float
    output_per_million_usd: float
    long_context_threshold_tokens: int
    long_context_input_multiplier: float
    long_context_output_multiplier: float

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Pricing:
        return cls(
            input_per_million_usd=float(config["input_per_million_usd"]),
            cached_input_per_million_usd=float(config["cached_input_per_million_usd"]),
            cache_write_per_million_usd=float(config["cache_write_per_million_usd"]),
            output_per_million_usd=float(config["output_per_million_usd"]),
            long_context_threshold_tokens=int(config["long_context_threshold_tokens"]),
            long_context_input_multiplier=float(config["long_context_input_multiplier"]),
            long_context_output_multiplier=float(config["long_context_output_multiplier"]),
        )


def calculate_cost_usd(usage: Usage, pricing: Pricing) -> float | None:
    required = (
        usage.input_tokens,
        usage.cached_input_tokens,
        usage.cache_write_tokens,
        usage.output_tokens,
    )
    if any(value is None for value in required):
        return None
    assert usage.input_tokens is not None
    assert usage.cached_input_tokens is not None
    assert usage.cache_write_tokens is not None
    assert usage.output_tokens is not None
    uncached_tokens = usage.input_tokens - usage.cached_input_tokens - usage.cache_write_tokens
    if uncached_tokens < 0:
        return None
    input_multiplier = (
        pricing.long_context_input_multiplier
        if usage.input_tokens > pricing.long_context_threshold_tokens
        else 1.0
    )
    output_multiplier = (
        pricing.long_context_output_multiplier
        if usage.input_tokens > pricing.long_context_threshold_tokens
        else 1.0
    )
    input_cost = (
        uncached_tokens * pricing.input_per_million_usd
        + usage.cached_input_tokens * pricing.cached_input_per_million_usd
        + usage.cache_write_tokens * pricing.cache_write_per_million_usd
    ) * input_multiplier
    output_cost = usage.output_tokens * pricing.output_per_million_usd * output_multiplier
    return (input_cost + output_cost) / 1_000_000


def approximate_tokens(character_count: int, characters_per_token: int) -> int:
    if characters_per_token <= 0:
        raise ValueError("characters_per_token must be positive")
    return math.ceil(character_count / characters_per_token)


def estimate_max_cost_usd(
    approximate_input_tokens: int, max_output_tokens: int, pricing: Pricing
) -> float:
    input_multiplier = (
        pricing.long_context_input_multiplier
        if approximate_input_tokens > pricing.long_context_threshold_tokens
        else 1.0
    )
    output_multiplier = (
        pricing.long_context_output_multiplier
        if approximate_input_tokens > pricing.long_context_threshold_tokens
        else 1.0
    )
    return (
        approximate_input_tokens * pricing.input_per_million_usd * input_multiplier
        + max_output_tokens * pricing.output_per_million_usd * output_multiplier
    ) / 1_000_000
