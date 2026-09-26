"""python -m train_agent ..."""
import sys
import warnings

warnings.filterwarnings("ignore", message="enable_nested_tensor")

from train_agent.agent import main  # noqa: E402

sys.exit(main())
