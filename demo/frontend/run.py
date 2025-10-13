import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add the frontend directory to the Python path
frontend_dir = str(Path(__file__).parent.absolute())
if frontend_dir not in sys.path:
    sys.path.insert(0, frontend_dir)

# Now import the app
from app.interface import launch_ui

# Load environment variables
env_path = Path(frontend_dir) / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

if __name__ == "__main__":
    launch_ui(
        server_name=os.getenv("FRONTEND_HOST", "0.0.0.0"),
        server_port=int(os.getenv("FRONTEND_PORT", "7861")),
        share=os.getenv("FRONTEND_SHARE", "false").lower() == "true"
    )