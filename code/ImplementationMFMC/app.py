"""Lancement graphique : python app.py. CLI : python app.py --help."""
import sys
import os
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    for variable, folder in (("TORCH_HOME", "torch"), ("TORCH_EXTENSIONS_DIR", "torch_extensions"),
                             ("DNNLIB_CACHE_DIR", "dnnlib"), ("MPLCONFIGDIR", "matplotlib")):
        os.environ.setdefault(variable, str(root / ".cache" / folder))
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    if len(sys.argv) > 1:
        from src.cli import main as cli_main
        return cli_main()
    try:
        from PySide6.QtWidgets import QApplication
        from src.gui.main_window import MainWindow
    except ImportError as exc:
        print(f"Dépendance manquante : {exc}. Installer requirements.txt avec Python 3.10 x64.", file=sys.stderr)
        return 2
    app = QApplication(sys.argv)
    app.setApplicationName("FaceAnon")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
