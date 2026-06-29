"""i18n helpers — installs Qt translators for English (source) and German.

Every user-facing string in the UI goes through ``QObject.tr(...)``. The
language selector in SERVICE calls :func:`install_translator` with the new
locale; ``QApplication.instance().installTranslator(...)`` swaps the active
catalog and the UI re-translates on the fly (each top-level view listens for
``QEvent.LanguageChange`` and re-applies its labels).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from PySide6.QtCore import QCoreApplication, QTranslator

Language = Literal["en", "de"]

_current_translator: QTranslator | None = None


def translations_dir() -> Path:
    """Default location of the compiled ``.qm`` files."""
    return Path(__file__).resolve().parents[2] / "translations"


def install_translator(language: Language, *, search_dir: Path | None = None) -> bool:
    """Install (and replace any previous) translator for ``language``.

    Returns True if the catalog loaded successfully, False otherwise. English
    is treated as the source language — passing ``"en"`` removes any active
    translator and returns True.
    """
    global _current_translator
    app = QCoreApplication.instance()
    if app is None:
        raise RuntimeError("install_translator requires a QCoreApplication / QApplication")

    if _current_translator is not None:
        app.removeTranslator(_current_translator)
        _current_translator = None

    if language == "en":
        # No catalog needed — strings flow through tr() unchanged.
        return True

    qm = (search_dir or translations_dir()) / f"viscontrol_{language}.qm"
    translator = QTranslator()
    if not qm.exists() or not translator.load(str(qm)):
        return False
    app.installTranslator(translator)
    _current_translator = translator
    return True
