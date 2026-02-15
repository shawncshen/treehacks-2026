import os
import sys

# Add project root to path so emg_app and emg_core can be found
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Force auto-init for Vercel
os.environ["AUTO_INIT"] = "1"

from emg_app import app
