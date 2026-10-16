"""Queued procedures must default to the conventional amplifier.

ExperimentQueueRunner now applies each procedure's camera parameters, so a
procedure built without explicit settings would switch the camera to its
amplifier_type default. Match the UI default (Conventional, no EM gain).
"""

from __future__ import annotations

import os

os.environ["ANDOR_MOCK"] = "1"

import pytest


@pytest.mark.parametrize("name", ["SpectrumProcedure", "ImageProcedure"])
def test_default_amplifier_is_conventional(name):
    import andor_qt.procedures as procedures

    proc = getattr(procedures, name)()
    assert proc.amplifier_type == 1
