"""Integration tests needing network access or a real GROMACS install.

fetch_memprotmd_system() needs network access to
https://memprotmd.bioch.ox.ac.uk/; skipped when that host is unreachable.
canonicalize_gromacs_system() needs `gmx` on PATH; skipped otherwise.
estimate_membrane_geometry() is exercised against a committed fixture
instead (tests/unit_tests/membrane_test.py) since it needs neither.
"""

from __future__ import annotations

import shutil
import socket
from pathlib import Path

import pytest

from gbsa_pipeline.membrane import canonicalize_gromacs_system, fetch_memprotmd_system

TESTDATA_1PY6 = Path(__file__).resolve().parents[1] / "testdata" / "membrane" / "1py6"


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


@pytest.mark.integration
def test_canonicalize_gromacs_system_fixes_legacy_pdb_issues(tmp_path: Path) -> None:
    """canonicalize_gromacs_system() produces a .gro that BioSimSpace can load.

    tests/testdata/membrane/1py6/atomistic-system.pdb has both legacy-GROMACS
    issues this function exists for (old-style hydrogen names, a >9999-residue
    PDB numbering wrap) -- this is the real system the fix was diagnosed
    against, not a synthetic case. Loading via BSS.IO.readMolecules confirms
    the fix, not just that gmx grompp/editconf ran without error.
    """
    if shutil.which("gmx") is None:
        pytest.skip("gmx not available on PATH")

    canonical_gro = canonicalize_gromacs_system(
        TESTDATA_1PY6 / "atomistic-system.pdb",
        TESTDATA_1PY6 / "topol.top",
        tmp_path / "canonicalize",
    )

    assert canonical_gro.exists()

    import BioSimSpace as BSS  # noqa: PLC0415

    system = BSS.IO.readMolecules([str(canonical_gro), str(TESTDATA_1PY6 / "topol.top")], make_whole=True)
    assert system.nAtoms() == 53316
