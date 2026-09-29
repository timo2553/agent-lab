"""让 pytest 无论从哪个目录启动都能 import agent_lab。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))