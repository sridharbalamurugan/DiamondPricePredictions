import logging
import os
from datetime import datetime

# Generate unique log filename
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(os.getcwd(), "logs", LOG_FILE)

# Ensure "logs" directory exists
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s: %(message)s",
    level=logging.INFO
)

logging.info("Logging has started successfully.")


