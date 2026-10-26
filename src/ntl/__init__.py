"""ntl — Name That LLM (Kaggle competition) toolkit.

Modular helpers for data loading, EDA, linguistic features, the attribution
pipeline and submissions.  Designed to be imported into a Kaggle/notebook
environment (and inlined into a notebook for a no-install, no-internet run).
"""

from .config import DATA_DIR, LABELS, PROVIDERS, RANDOM_STATE, competition_dirs
from .data import (
    as_text_array,
    describe_in_style_pool,
    inventory,
    kaggle_input_dirs,
    load_in_style_pool,
    load_sample_submission,
    load_test,
    load_train,
)
from .decision import argmax_decision, assign_within_groups, balanced_accuracy, sinkhorn
from .groups import group_report, group_sizes, infer_groups
from .pipeline import StackConfig, StackModel, group_cv_score
from .submit import make_submission, validate_submission
from .views import (
    DEFAULT_VIEWS,
    VIEWS,
    function_transform,
    punct_transform,
    shape_transform,
)

__version__ = "0.4.0"

__all__ = [
    "DATA_DIR",
    "DEFAULT_VIEWS",
    "LABELS",
    "PROVIDERS",
    "RANDOM_STATE",
    "VIEWS",
    "StackConfig",
    "StackModel",
    "argmax_decision",
    "as_text_array",
    "assign_within_groups",
    "balanced_accuracy",
    "competition_dirs",
    "describe_in_style_pool",
    "function_transform",
    "group_cv_score",
    "group_report",
    "group_sizes",
    "infer_groups",
    "inventory",
    "kaggle_input_dirs",
    "load_in_style_pool",
    "load_sample_submission",
    "load_test",
    "load_train",
    "make_submission",
    "punct_transform",
    "shape_transform",
    "sinkhorn",
    "validate_submission",
]
