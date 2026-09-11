"""Integration test for fetching a real system from MemProtMD.

Needs network access to https://memprotmd.bioch.ox.ac.uk/; skipped when that
host is unreachable. estimate_membrane_geometry() is exercised against a
committed fixture instead (tests/unit_tests/membrane_test.py) since it needs
no network.
"""

from __future__ import annotations

import socket
from typing import TYPE_CHECKING

import pytest

from gbsa_pipeline.membrane import fetch_memprotmd_system

if TYPE_CHECKING:
    from pathlib import Path


def _memprotmd_reachable() -> bool:
    try:
        socket.create_connection(("memprotmd.bioch.ox.ac.uk", 443), timeout=5).close()
    except OSError:
        return False
    return True


@pytest.mark.integration
def test_fetch_memprotmd_system_downloads_1py6(tmp_path: Path) -> None:
    """fetch_memprotmd_system("1py6", ...) returns a loadable structure+topology pair.

    1py6 (bacteriorhodopsin) is a stable, long-standing MemProtMD entry used
    as this pipeline's reference example (tests/testdata/membrane/1py6/ is a
    trimmed, committed copy of the same download for the offline unit tests).
    """
    if not _memprotmd_reachable():
        pytest.skip("memprotmd.bioch.ox.ac.uk is not reachable")

    system = fetch_memprotmd_system("1py6", tmp_path)

    assert system.structure.exists()
    assert system.topology.exists()
    assert system.index is not None
    assert system.index.exists()
    assert "DPP" in system.structure.read_text()


@pytest.mark.integration
def test_fetch_memprotmd_system_unknown_pdb_code_raises(tmp_path: Path) -> None:
    """A PDB code with no MemProtMD entry raises FileNotFoundError, not an HTTP error."""
    if not _memprotmd_reachable():
        pytest.skip("memprotmd.bioch.ox.ac.uk is not reachable")

    with pytest.raises(FileNotFoundError):
        fetch_memprotmd_system("zzzz", tmp_path)
