"""
Tests for the local reference generator's three response strategies. These
exercise the REAL generator (no mocking) since it is fully local and
network-free -- see local_reference_model.py's module docstring.
"""

from local_reference_model import ReferenceSystem


def test_confident_fabricator_varies_across_calls():
    sys_ = ReferenceSystem(config="confident-fabricator")
    outputs = {sys_.respond("hangi yıl?") for _ in range(20)}
    # With 20 draws from a pool of ~10+7+6 fabricated facts spread across
    # 4 sentence templates, seeing at least a few distinct outputs is
    # overwhelmingly likely; this is the property the whole config exists
    # to demonstrate.
    assert len(outputs) > 3


def test_hedging_never_contains_digits():
    sys_ = ReferenceSystem(config="hedging")
    for _ in range(15):
        resp = sys_.respond("herhangi bir soru")
        assert not any(ch.isdigit() for ch in resp)


def test_hedging_responses_share_most_of_their_wording():
    sys_ = ReferenceSystem(config="hedging")
    outputs = [sys_.respond("soru") for _ in range(10)]
    # every hedge response should contain the shared stock disclaimer
    assert all("emin değilim" in o for o in outputs)


def test_grounded_returns_same_fact_every_time():
    sys_ = ReferenceSystem(config="grounded")
    outputs = [sys_.respond("Türkiye'nin başkenti neresidir?") for _ in range(10)]
    assert all("Ankara" in o for o in outputs)


def test_grounded_falls_back_gracefully_without_kb_match():
    sys_ = ReferenceSystem(config="grounded")
    resp = sys_.respond("Bu, bilgi tabanında kesinlikle karşılığı olmayan bir soru mu acaba?")
    assert isinstance(resp, str) and len(resp) > 0


def test_invalid_config_raises():
    import pytest
    with pytest.raises(ValueError):
        ReferenceSystem(config="not-a-real-config")
