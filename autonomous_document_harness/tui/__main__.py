"""Allow ``python -m autonomous_document_harness.tui`` to launch the TUI."""
from .app import main
raise SystemExit(main())
