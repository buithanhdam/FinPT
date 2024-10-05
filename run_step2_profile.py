import os
import sys
import time
import json
import logging
import argparse
import random
import dotenv
from google.api_core.exceptions import ResourceExhausted
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor

dotenv.load_dotenv()

def setup_logging() -> logging.Logger:
    logging.basicConfig(
        format="[%(asctime)s - %(levelname)s - %(name)s] -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
    )
    return logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step2 Get_Profile Args")
    parser.add_argument("--ds_name", type=str, default="products_finpt_data", help="Specify which dataset to use")
    parser.add_argument("--model_name", type=str, default="gemini-1.5-flash", help="Specify which model name to use")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Ratio of data for training set")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Ratio of data for validation set")
    return parser.parse_args()

def setup_environment() -> str:

    profile_root_dir = os.path.join("./output")
    os.makedirs(profile_root_dir, exist_ok=True)
    return profile_root_dir

def setup_gemini(model_name:str = 'gemini-1.5-flash') -> genai.GenerativeModel:

    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    return genai.GenerativeModel(model_name)

def get_file_paths(profile_dir: str, ds_name: str):

    instruction_path = os.path.join(profile_dir, f"instruction_for_profile_{ds_name}.jsonl")
    train_path = os.path.join(profile_dir, f"profile_{ds_name}_train.jsonl")
    val_path = os.path.join(profile_dir, f"profile_{ds_name}_validation.jsonl")
    test_path = os.path.join(profile_dir, f"profile_{ds_name}_test.jsonl")
    return instruction_path, train_path, val_path, test_path

def process_instruction(model: genai.GenerativeModel, instruction: str,logger: logging.Logger,retries:int=3, initial_wait:int=1) -> str:

    wait_time = initial_wait
    for attempt in range(retries):
        try:
            response = model.generate_content([
                "You are a helpful financial assistant.",
                instruction
            ], generation_config=genai.types.GenerationConfig(
                temperature=0.1,
            ))
            return response.text.strip()
        except ResourceExhausted as e:
            if attempt < retries - 1:
                logger.warning(f"!!Resource exhausted. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                wait_time *= 2 + random.uniform(0, 0.5)  # Exponential backoff với jitter
            else:
                logger.error(f"Failed after {retries} retries: {e}")
                return None

def write_response(fp_out, response: str):
    res_json = json.dumps(response)
    fp_out.write(res_json + "\n")

def process_line(model, line, logger, split_name, index):
    instruction = str(json.loads(line.strip()))
    response = process_instruction(model, instruction)
    if response is None:
        logger.warning(f"Failed process instruction for index: {index}")
    if (index + 1) % 50 == 0:
        logger.info(f">>> Processed {index + 1} items for {split_name} set")
    return response

def split_and_process_file(instruction_path: str, train_path: str, val_path: str, test_path: str, 
                           model: genai.GenerativeModel, train_ratio: float, val_ratio: float, logger: logging.Logger):
    with open(instruction_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    random.shuffle(lines)
    total_lines = len(lines)
    train_end = int(total_lines * train_ratio)
    val_end = train_end + int(total_lines * val_ratio)

    datasets = [
        (lines[:train_end], train_path, "train"),
        (lines[train_end:val_end], val_path, "validation"),
        (lines[val_end:], test_path, "test")
    ]

    for data, path, split_name in datasets:
        with open(path, "w", encoding="utf-8") as fp_out:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(process_line, model, line, logger, split_name, i) for i, line in enumerate(data)]
                
                for i, future in enumerate(futures):
                    response = future.result()
                    if response is None:
                        continue
                    write_response(fp_out, response)
                    time.sleep(0.1)

        logger.info(f">>> DONE: [{split_name}] processed {len(data)} items")

def run_gemini(ds_name: str, train_ratio: float, val_ratio: float, profile_root_dir: str = "./output",model_name:str = 'gemini-1.5-flash') -> int:
    logger = logging.getLogger(__name__)

    logger.info(f">>> ds_name: {ds_name}")
    instruction_path, train_path, val_path, test_path = get_file_paths(profile_root_dir, ds_name)

    model = setup_gemini(model_name=model_name)
    logger.info(f">>> Initialize model successfully with model_name: {model_name}")
    split_and_process_file(instruction_path, train_path, val_path, test_path, model, train_ratio, val_ratio, logger)

    logger.info(f"\n>>> DONE: [{ds_name}] Processing completed for all sets\n\n")
    return 0

def main():
    logger = setup_logging()
    args = parse_arguments()
    logger.info(args)

    profile_root_dir = setup_environment()

    run_gemini(
        ds_name=args.ds_name,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        model_name=args.model_name,
        profile_root_dir=profile_root_dir
    )
if __name__ == "__main__":
    """
    python run_step2_profile.py --ds_name products_finpt_data --model_name gemini-1.5-flash --train_ratio 0.7 --val_ratio 0.15
    """
    main()
    sys.exit(0)