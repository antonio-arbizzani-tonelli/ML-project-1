"""Select the frozen Phase 14 submission or the corrected Phase 34 search."""


def get_booster_class(implementation):
    """Keep the submitted numerical implementation independent of later fixes."""
    if implementation == "phase14":
        from src.numpy_boosting_phase14 import HistogramGradientBoostingClassifier
    elif implementation == "phase34":
        from src.numpy_boosting import HistogramGradientBoostingClassifier
    else:
        raise ValueError("implementation must be 'phase14' or 'phase34'.")
    return HistogramGradientBoostingClassifier
