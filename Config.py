# config.py
import os


DATASET_PATH = './Dataset_format/BBH/geometric_shapes.csv'

# put your API
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL = "deepseek-chat" # anyone you want
TEMPERATURE = 0.6


current_time = None
question_type = None

# Optional runtime controls used by reproduce_mars.py. The legacy run.sh entry
# still works with these defaults.
MAX_ITERATIONS = 10
EARLY_STOP_DELTA = 0.01
MAX_CRITIC_REVISIONS = 1
CONCURRENCY = 1
DRY_RUN = False
OUTPUT_DIR = "./Output"
TASK_OUTPUT_DIR = None
PROMPT_HISTORY_PATH = None
PREDICTIONS_PATH = None
LAST_PROMPT_HISTORY = []
LAST_PREDICTIONS = []
LAST_STOPPED_REASON = None
