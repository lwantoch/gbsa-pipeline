"""Unit tests for _spatial.py geometric primitives."""

from __future__ import annotations

import numpy as np

from gbsa_pipeline._spatial import contact_pairs


def test_contact_pairs_returns_empty_when_no_atoms() -> None:
    """Empty coordinate arrays produce no pairs."""
    assert contact_pairs(np.empty((0, 3)), np.array([[1.0, 0.0, 0.0]]), 2.0) == []
    assert contact_pairs(np.array([[1.0, 0.0, 0.0]]), np.empty((0, 3)), 2.0) == []


def test_contact_pairs_detects_close_pair() -> None:
    """Atoms within cutoff are returned with their distance."""
    a = np.array([[0.0, 0.0, 0.0]])
    b = np.array([[1.0, 0.0, 0.0]])
    result = contact_pairs(a, b, 2.0)
    assert len(result) == 1
    i, j, dist = result[0]
    assert i == 0
    assert j == 0
    assert abs(dist - 1.0) < 1e-6


def test_contact_pairs_boundary_inclusive() -> None:
    """A pair exactly at the cutoff distance is included."""
    a = np.array([[0.0, 0.0, 0.0]])
    b = np.array([[2.0, 0.0, 0.0]])
    result = contact_pairs(a, b, 2.0)
    assert len(result) == 1


def test_contact_pairs_excludes_distant_pair() -> None:
    """Atoms beyond cutoff are not returned."""
    a = np.array([[0.0, 0.0, 0.0]])
    b = np.array([[10.0, 0.0, 0.0]])
    assert contact_pairs(a, b, 2.0) == []


def test_contact_pairs_multiple_pairs() -> None:
    """All pairs within cutoff are returned; those outside are not."""
    a = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    b = np.array([[1.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    result = contact_pairs(a, b, 2.0)
    assert len(result) == 1
    assert result[0][:2] == (0, 0)
