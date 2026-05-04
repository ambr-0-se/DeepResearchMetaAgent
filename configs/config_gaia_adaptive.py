"""
Deprecated path: use ``configs/config_gaia_c1.py``.

``adaptive`` in filenames referred to the reactive diagnose + ``modify_subagent``
track, which is **paper condition C1** after the C0–C3 contiguous renumbering.
This stub keeps old commands alive:

    python examples/run_gaia.py --config configs/config_gaia_adaptive.py

Prefer ``config_gaia_c1.py`` in new scripts and docs.
"""

_base_ = './config_gaia_c1.py'
