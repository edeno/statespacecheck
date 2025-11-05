"""Tests for package version handling."""

from __future__ import annotations

import sys
from unittest.mock import patch


def test_version_attribute_exists() -> None:
    """Test that __version__ attribute is accessible."""
    import statespacecheck

    assert hasattr(statespacecheck, "__version__")
    assert isinstance(statespacecheck.__version__, str)
    # Version should follow semantic versioning pattern
    assert len(statespacecheck.__version__) > 0


def test_version_fallback_to_metadata() -> None:
    """Test that version fallback works when _version module is unavailable."""
    # Remove statespacecheck from sys.modules to force reimport
    modules_to_remove = [key for key in sys.modules.keys() if key.startswith("statespacecheck")]
    original_modules = {key: sys.modules[key] for key in modules_to_remove}

    try:
        # Remove all statespacecheck modules
        for key in modules_to_remove:
            del sys.modules[key]

        # Mock the _version import to raise ImportError
        with patch.dict(sys.modules, {"statespacecheck._version": None}):
            # This should trigger the ImportError and use the fallback
            import statespacecheck

            # Version should still be accessible via fallback
            assert hasattr(statespacecheck, "__version__")
            assert isinstance(statespacecheck.__version__, str)
            assert len(statespacecheck.__version__) > 0
    finally:
        # Restore original modules
        for key in modules_to_remove:
            if key in original_modules:
                sys.modules[key] = original_modules[key]
