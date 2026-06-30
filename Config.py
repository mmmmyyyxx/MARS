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
ANSWER_FORMAT = "auto"

# Optional runtime controls used by reproduce_mars.py. The legacy run.sh entry
# still works with these defaults.
MAX_ITERATIONS = 10
EARLY_STOP_DELTA = 0.01
MAX_CRITIC_REVISIONS = 1
CONCURRENCY = 1
MAX_SAMPLES = None
MAX_ANSWER_RETRIES = 3
REQUEST_TIMEOUT = 60
DRY_RUN = False
OUTPUT_DIR = "./Output"
TASK_OUTPUT_DIR = None
PROMPT_HISTORY_PATH = None
PREDICTIONS_PATH = None
LAST_PROMPT_HISTORY = []
LAST_PREDICTIONS = []
LAST_STOPPED_REASON = None
BEST_PROMPT = ""
BEST_ACCURACY = -1.0
BEST_ITERATION = None
