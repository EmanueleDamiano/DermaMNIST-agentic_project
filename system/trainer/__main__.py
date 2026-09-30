"""python run.py trainer ..."""
import sys
import warnings

warnings.filterwarnings("ignore", message="enable_nested_tensor")

from trainer.agent import main  # noqa: E402

sys.exit(main())
