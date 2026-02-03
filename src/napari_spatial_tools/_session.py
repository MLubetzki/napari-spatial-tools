"""
Module to store session data accessible from the console.
This works around the blocking behavior of Interactive().
"""

# Global session storage
_current_sdata = None
_current_interactive = None
_sample_order = []  # List of sample names in the order they were passed as command line args


def set_session_data(sdata, interactive, sample_order=None):
    """Store the current session's spatial data and interactive viewer."""
    global _current_sdata, _current_interactive, _sample_order
    _current_sdata = sdata
    _current_interactive = interactive
    if sample_order is not None:
        _sample_order = sample_order


def get_session_data():
    """Get the current session's spatial data."""
    return _current_sdata


def get_interactive():
    """Get the current session's interactive viewer."""
    return _current_interactive


def get_sample_order():
    """Get the list of sample names in order."""
    return _sample_order
