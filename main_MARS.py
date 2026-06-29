import re
from typing import AsyncGenerator, List, Sequence,Tuple
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import AgentMessage, ChatMessage, TextMessage
from autogen_core import CancellationToken
from openai import OpenAI
import pandas as pd
import sys
from datetime import datetime
import os
from tqdm import tqdm 
from Agents import TargetAgent,TeacherAgent,StudentAgent,PlannerAgent,CriticAgent,analyze_prompt_history
from Agents import ChatManagerAgent,UserProxyAgent
import Config
import asyncio
import time
from contextlib import redirect_stdout

def get_question_type():
    """ Read command line arguments and return the question type """
    if len(sys.argv) > 1:
        return sys.argv[1]
    return "choice"  # Default Choice Questions

async def run_agents():
    """ Run Agents, passing the question type """
    chat_manager_agent = ChatManagerAgent("chat_manager")
    target_agent = TargetAgent("target_agent") 

    task_message = TextMessage(
        content="Here is a topic for geometric graph generation, I want to input a prompt and this topic into the big language model so that the big language model outputs the highest correctness rate.",
        source="user"
    )

    chat_manager_response = await chat_manager_agent.on_messages([task_message], CancellationToken())
    print(chat_manager_response.chat_message.content)
    await asyncio.sleep(0.1)


def run_mars_task(
    task_id: str,
    dataset_path: str,
    question_type: str,
    user_prompt: str,
    planner_prompt: str,
    config,
    output_dir: str,
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("./Output", exist_ok=True)

    Config.DATASET_PATH = dataset_path
    Config.question_type = question_type
    Config.ANSWER_FORMAT = getattr(config, "answer_format", "auto")
    Config.API_KEY = config.api_key
    Config.BASE_URL = config.base_url
    Config.MODEL = config.model
    Config.TEMPERATURE = config.temperature
    Config.MAX_ITERATIONS = config.max_iterations
    Config.EARLY_STOP_DELTA = config.early_stop_delta
    Config.MAX_CRITIC_REVISIONS = config.max_critic_revisions
    Config.CONCURRENCY = config.concurrency
    Config.DRY_RUN = config.dry_run
    Config.TASK_OUTPUT_DIR = output_dir
    Config.PREDICTIONS_PATH = os.path.join(output_dir, "predictions.csv")
    Config.PROMPT_HISTORY_PATH = os.path.join(output_dir, "prompt_accuracy_history.csv")
    Config.LAST_PROMPT_HISTORY = []
    Config.LAST_PREDICTIONS = []
    Config.LAST_STOPPED_REASON = None
    Config.current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    user_prompt_path = "./Prompt/EDIT_1_userproxy_task_input.txt"
    planner_prompt_path = "./Prompt/EDIT_2_prompt_planner_template.txt"
    with open(user_prompt_path, "r", encoding="utf-8") as file:
        original_user_prompt = file.read()
    with open(planner_prompt_path, "r", encoding="utf-8") as file:
        original_planner_prompt = file.read()

    raw_log_path = os.path.join(output_dir, "raw_logs.txt")
    start_time = time.time()
    try:
        with open(user_prompt_path, "w", encoding="utf-8") as file:
            file.write(user_prompt)
        with open(planner_prompt_path, "w", encoding="utf-8") as file:
            file.write(planner_prompt)

        with open(raw_log_path, "w", encoding="utf-8") as log_file:
            with redirect_stdout(log_file):
                print(f"task_id: {task_id}")
                print(f"start time: {start_time:.4f} s")
                asyncio.run(run_agents())
                end_time = time.time()
                print(f"end time: {end_time:.4f} s")
                print(f"program runtime: {end_time - start_time:.4f} s")
    finally:
        with open(user_prompt_path, "w", encoding="utf-8") as file:
            file.write(original_user_prompt)
        with open(planner_prompt_path, "w", encoding="utf-8") as file:
            file.write(original_planner_prompt)

    history = list(getattr(Config, "LAST_PROMPT_HISTORY", []))
    end_time = time.time()
    return {
        "task_id": task_id,
        "prompt_history": history,
        "predictions": list(getattr(Config, "LAST_PREDICTIONS", [])),
        "stopped_reason": getattr(Config, "LAST_STOPPED_REASON", None) or "completed",
        "runtime_seconds": end_time - start_time,
        "raw_log_path": raw_log_path,
    }

if __name__ == "__main__":

    Config.current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    os.makedirs('./Output', exist_ok=True)
    file_name = os.path.join('./Output', f'{Config.current_time}_output.txt')
    sys.stdout = open(file_name, 'w', encoding='utf-8')
    
    Config.question_type = get_question_type()
    start_time = time.time()
    print(f"start time: {start_time:.4f} s")

    asyncio.run(run_agents())

    end_time = time.time()
    print(f"end time: {end_time:.4f} s")
    print(f"program runtime: {end_time - start_time:.4f} s")

    # Restore standard output
    sys.stdout.close()
    sys.stdout = sys.__stdout__
