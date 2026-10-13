"""Start the web app.

    python run.py                    # http://0.0.0.0:8000, waitress (production WSGI server) if installed
    python run.py --port 5000 --dev  # Flask dev server with auto-reload

Env: E0_DATA_DIR (default ./data), E0_MODEL_DIR (default ./model), E0_THREADS (TFLite threads, default 4).
"""
import argparse

from app import create_app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--dev", action="store_true", help="Flask debug server (never on a shared network)")
    args = ap.parse_args()
    app = create_app()
    if args.dev:
        app.run(host=args.host, port=args.port, debug=True)
        return
    try:
        from waitress import serve
    except ImportError:
        print("waitress not installed; using the Flask server")
        app.run(host=args.host, port=args.port)
        return
    print(f"Ovarian Twin on http://{args.host}:{args.port}  (model {app.extensions['e0']['model'].version})", flush=True)
    serve(app, host=args.host, port=args.port, threads=4)


if __name__ == "__main__":
    main()
