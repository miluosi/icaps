"""Production NYC recourse-architecture training/testing entrypoint.

This runner covers the canonical paper recourse methods and causal controls.
State, learner-family, and solver matrices have separate audit entrypoints.
``test_all_nyc_models.py`` remains import-compatible with older commands.
"""
from test_all_nyc_models import *  # noqa: F401,F403
from test_all_nyc_models import main


if __name__ == '__main__':
    main()
