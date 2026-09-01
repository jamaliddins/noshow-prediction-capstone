"""One-command local demo: start the no-show service and open it in a browser.

    python run_demo.py

Serves on http://localhost:8000 :

    /            interactive demo page (form -> real model -> risk tier)
    /docs        Swagger UI for the full API
    /health      liveness + whether the model artifact loaded
    /model       model name, held-out test metrics, thresholds
    /predict     score one appointment
    /predict/batch  score up to 1000, highest risk first

Options:
    --port N        serve on a different port (default 8000)
    --no-browser    do not open a browser window
    --host H        bind address (default 127.0.0.1; use 0.0.0.0 to expose)

Everything is served by the real pipeline in models/model.joblib. If that
artifact is missing this script says so and tells you to run the training
step first, rather than starting a service that cannot predict.
"""
from __future__ import annotations

import argparse
import socket
import sys
import threading
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def _fail(message: str, hint: str = "") -> None:
    """Print a readable failure and exit non-zero."""
    print(f"\n  ERROR: {message}")
    if hint:
        print(f"  {hint}")
    print()
    sys.exit(1)


def check_python() -> None:
    if sys.version_info < (3, 10):
        _fail(
            f"Python 3.10+ is required, this is {sys.version.split()[0]}.",
            "Activate the project venv, or install a newer Python.",
        )


def check_dependencies() -> None:
    """Import what the demo needs, naming the missing package if any."""
    required = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "pandas": "pandas",
        "numpy": "numpy",
        "sklearn": "scikit-learn",
        "xgboost": "xgboost",
        "joblib": "joblib",
    }
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    if missing:
        _fail(
            f"Missing package(s): {', '.join(missing)}",
            "Install them with:  pip install -r requirements.txt",
        )


def check_model() -> dict:
    """Confirm the trained artifacts exist and load; return the metadata."""
    from src.predict import load_model

    try:
        _, metadata = load_model()
    except FileNotFoundError:
        _fail(
            "No trained model found at models/model.joblib.",
            "Train one first:  python -m src.train",
        )
    except Exception as exc:  # corrupt or version-mismatched artifact
        _fail(
            f"models/model.joblib exists but failed to load: {exc}",
            "Retrain with:  python -m src.train",
        )
    return metadata


def free_port(host: str, port: int) -> int:
    """Return `port` if it is free, otherwise the next free port after it."""
    for candidate in range(port, port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, candidate))
                return candidate
            except OSError:
                continue
    _fail(f"No free port in range {port}-{port + 19}.", "Pass --port N.")


def build_app():
    """The API service, with the interactive demo page mounted at /.

    src.api owns the endpoints (validation, batch, model info); scripts
    /demo_web.py owns the HTML page. Mounting both here keeps the demo and
    the documented API on one URL, so there is a single thing to run.
    """
    from fastapi.responses import HTMLResponse

    from scripts.demo_web import api_predict, index
    from src.api import app, root

    # src.api already claims GET / for its JSON index, and FastAPI serves the
    # first matching route — so drop it and give the demo page that spot,
    # keeping the JSON index reachable at /api.
    app.router.routes = [
        r for r in app.router.routes if getattr(r, "path", None) != "/"
    ]
    app.add_api_route("/", index, methods=["GET"],
                      response_class=HTMLResponse, include_in_schema=False)
    app.add_api_route("/api", root, methods=["GET"], include_in_schema=False)

    # The page posts to /api/predict; src.api's own /predict is unchanged.
    app.add_api_route("/api/predict", api_predict, methods=["POST"],
                      include_in_schema=False)
    return app


def open_browser_when_up(url: str, delay: float = 1.5) -> None:
    """Open the browser shortly after uvicorn starts serving."""
    threading.Timer(delay, lambda: webbrowser.open(url)).start()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the no-show prediction demo on localhost."
    )
    parser.add_argument("--port", type=int, default=8000, help="default 8000")
    parser.add_argument("--host", default="127.0.0.1",
                        help="default 127.0.0.1; 0.0.0.0 exposes on the network")
    parser.add_argument("--no-browser", action="store_true",
                        help="do not open a browser window")
    args = parser.parse_args()

    print("\n" + "=" * 66)
    print("  NO-SHOW PREDICTION - LOCAL DEMO")
    print("=" * 66)

    check_python()
    print(f"  [ok] Python {sys.version.split()[0]}")

    check_dependencies()
    print("  [ok] dependencies importable")

    metadata = check_model()
    test_f1 = metadata["test_metrics"]["f1"]
    print(f"  [ok] model loaded: {metadata['model_name']} "
          f"({metadata.get('calibration', 'uncalibrated')}), "
          f"test F1 {test_f1:.3f}, threshold {metadata['threshold']:.3f}")

    port = free_port(args.host, args.port)
    if port != args.port:
        print(f"  [!]  port {args.port} is in use — using {port} instead")

    app = build_app()

    display_host = "localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host
    url = f"http://{display_host}:{port}"
    print("-" * 66)
    print(f"  Demo page   {url}")
    print(f"  API docs    {url}/docs")
    print(f"  Health      {url}/health")
    print("-" * 66)
    print("  Press Ctrl+C to stop.\n")

    if not args.no_browser:
        open_browser_when_up(url)

    import uvicorn

    try:
        uvicorn.run(app, host=args.host, port=port, log_level="warning")
    except KeyboardInterrupt:
        pass
    print("\n  Demo stopped.\n")


if __name__ == "__main__":
    main()
