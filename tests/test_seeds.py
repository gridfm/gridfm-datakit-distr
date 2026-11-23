"""Tests for the seed tree and scenario ids."""

from __future__ import annotations

import hashlib
import random
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
import pytest

from gridfm_datakit_distr.model.ids import (
    MAX_NETWORK_INDEX,
    MAX_SCENARIO_INDEX,
    MAX_VARIANT_INDEX,
    scenario_id,
    split_scenario_id,
)
from gridfm_datakit_distr.scenarios.seeds import (
    ScenarioStream,
    VariantStream,
    network_key,
    scenario_rng,
    scenario_seed,
    scenario_sequence,
    variant_rng,
    variant_seed,
    variant_sequence,
)

SEED = 42


# ------------------------------------------------------------------- scenario ids


def test_scenario_id_packs_the_indices() -> None:
    assert scenario_id(0, 0, 0) == 0
    assert scenario_id(1, 2, 3) == (1 << 47) | (2 << 31) | 3
    top = scenario_id(MAX_NETWORK_INDEX, MAX_VARIANT_INDEX, MAX_SCENARIO_INDEX)
    assert top < 2**63  # fits int64
    assert split_scenario_id(top) == (MAX_NETWORK_INDEX, MAX_VARIANT_INDEX, MAX_SCENARIO_INDEX)


def test_scenario_ids_are_unique_and_invertible() -> None:
    rng = random.Random(0)
    triples = {
        (rng.randrange(MAX_NETWORK_INDEX + 1), rng.randrange(10_000), rng.randrange(2**31))
        for _ in range(2_000)
    }
    ids = {scenario_id(*t) for t in triples}
    assert len(ids) == len(triples)
    assert all(split_scenario_id(scenario_id(*t)) == t for t in triples)


@pytest.mark.parametrize(
    "triple",
    [(-1, 0, 0), (MAX_NETWORK_INDEX + 1, 0, 0), (0, MAX_VARIANT_INDEX + 1, 0), (0, 0, 2**31)],
)
def test_scenario_id_bounds(triple: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError, match="must be in"):
        scenario_id(*triple)


def test_split_rejects_negative_ids() -> None:
    with pytest.raises(ValueError, match="not a scenario id"):
        split_scenario_id(-1)


# ------------------------------------------------------------------- seed tree


def test_sequences_follow_the_spawn_tree_of_the_plan() -> None:
    net = network_key("csiro_enwl/n01_f1")
    network = np.random.SeedSequence(SEED, spawn_key=(net >> 32, net & 0xFFFF_FFFF))
    variants = network.spawn(4)
    scenarios = variants[1].spawn(6)
    assert variant_sequence(SEED, net, 1).generate_state(4).tolist() == (
        variants[1].generate_state(4).tolist()
    )
    assert scenario_sequence(SEED, net, 1, 5).generate_state(4).tolist() == (
        scenarios[5].generate_state(4).tolist()
    )


def test_the_network_key_is_the_sha256_of_its_id() -> None:
    digest = hashlib.sha256(b"ieee13").digest()
    assert network_key("ieee13") == int.from_bytes(digest[:8], "big")
    assert network_key("ieee13") != network_key("ieee34")
    assert 2**32 <= network_key("ieee13") < 2**64  # two words in the spawn key


def _draws(keys: list[tuple[int, int, int]]) -> dict[tuple[int, int, int], list[float]]:
    out = {}
    for i, k, s in keys:
        rng = scenario_rng(SEED, i, k, s, ScenarioStream.LOAD_NOISE)
        out[(i, k, s)] = rng.standard_normal(3).tolist()
    return out


KEYS = [(i, k, s) for i in range(2) for k in range(3) for s in range(5)]


def test_draws_do_not_depend_on_the_order() -> None:
    forward = _draws(KEYS)
    shuffled = list(KEYS)
    random.Random(1).shuffle(shuffled)
    assert _draws(shuffled) == forward
    assert _draws(list(reversed(KEYS))) == forward


def test_draws_do_not_depend_on_the_process() -> None:
    chunks = [KEYS[:10], KEYS[10:20], KEYS[20:]]
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as pool:
        parts = list(pool.map(_draws, chunks))
    merged = {key: value for part in parts for key, value in part.items()}
    assert merged == _draws(KEYS)


def test_streams_are_distinct() -> None:
    first = [scenario_rng(SEED, 0, 0, 0, s).random() for s in ScenarioStream]
    assert len(set(first)) == len(ScenarioStream)
    first = [variant_rng(SEED, 0, 0, s).random() for s in VariantStream]
    assert len(set(first)) == len(VariantStream)
    # neighbouring nodes and seeds differ
    a = scenario_rng(SEED, 0, 0, 0, ScenarioStream.LOAD_NOISE).random()
    assert a != scenario_rng(SEED, 0, 0, 1, ScenarioStream.LOAD_NOISE).random()
    assert a != scenario_rng(SEED, 0, 1, 0, ScenarioStream.LOAD_NOISE).random()
    assert a != scenario_rng(SEED + 1, 0, 0, 0, ScenarioStream.LOAD_NOISE).random()


def test_variant_streams_never_share_a_key_with_scenarios() -> None:
    # a variant's purpose streams live above the scenario indices (< 2**31)
    stream = variant_rng(SEED, 0, 0, VariantStream.LINE_LENGTH).bit_generator.seed_seq
    assert isinstance(stream, np.random.SeedSequence)
    assert stream.spawn_key[3] >= 2**31
    assert stream.spawn_key[:3] == (0, 0, 0)


def test_resampling_attempts_get_their_own_draws() -> None:
    a0 = variant_rng(SEED, 0, 0, VariantStream.LINE_LENGTH, attempt=0).random()
    a1 = variant_rng(SEED, 0, 0, VariantStream.LINE_LENGTH, attempt=1).random()
    assert a0 != a1
    assert a0 == variant_rng(SEED, 0, 0, VariantStream.LINE_LENGTH).random()


def test_stream_codes_are_stable() -> None:
    # codes are part of the data's provenance: append new streams, never renumber
    assert {s.name: int(s) for s in VariantStream} == {
        "LINE_LENGTH": 0,
        "NEUTRAL_IMPEDANCE": 1,
        "GROUNDING_RESISTANCE": 2,
        "PHASE_REALLOCATION": 3,
        "RECONFIGURATION": 4,
        "DER_PLACEMENT": 5,
        "DER_SIZE": 6,
        "DER_POWER_FACTOR": 7,
        "DER_REACTIVE_FRACTION": 8,
        "DER_REACTIVE_THRESHOLD": 9,
        "STORAGE_PLACEMENT": 10,
        "STORAGE_SIZE": 11,
    }
    assert {s.name: int(s) for s in ScenarioStream} == {
        "PROFILE_TIME": 0,
        "LOAD_SCALE": 1,
        "LOAD_NOISE": 2,
        "POWER_FACTOR": 3,
        "PV_OUTPUT": 4,
        "SOURCE_VOLTAGE": 5,
        "TAP_JITTER": 6,
        "CUSTOMER_TIMES": 7,
        "OPF_COSTS": 8,
        "PV_AVAILABILITY": 9,
        "STORAGE_SETPOINT": 10,
        "DG_SETPOINT": 11,
    }


def test_seed_columns_are_stable_fingerprints() -> None:
    v = variant_seed(SEED, 1, 2)
    s = scenario_seed(SEED, 1, 2, 3)
    assert isinstance(v, int)
    assert 0 <= v < 2**64
    assert 0 <= s < 2**64
    assert v != s
    assert v == variant_seed(SEED, 1, 2)
    expected = int(
        np.random.SeedSequence(SEED, spawn_key=(0, 1, 2, 3)).generate_state(1, np.uint64)[0]
    )
    assert s == expected


@pytest.mark.parametrize("bad", [(-1, 0), (0, -1), (2**64, 0), (0, MAX_VARIANT_INDEX + 1)])
def test_tree_bounds(bad: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="must be in"):
        variant_sequence(SEED, *bad)
    with pytest.raises(ValueError, match="seed must be"):
        variant_sequence(-1, 0, 0)
