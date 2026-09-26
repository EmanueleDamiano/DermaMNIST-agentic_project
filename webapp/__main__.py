"""python -m webapp [--host 127.0.0.1] [--port 8000]"""

import argparse
import warnings

warnings.filterwarnings("ignore", message="enable_nested_tensor")

from webapp.app import main  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="DermaAgent platform")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    main(args.host, args.port)
