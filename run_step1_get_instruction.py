import os
import sys
import json
import logging
import argparse
import pandas as pd

def setup_logging():
    logging.basicConfig(
        format="[%(asctime)s - %(levelname)s - %(name)s] -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
    )
    return logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process and profile dataset")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory for input data")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory for output files")
    parser.add_argument("--dataset_names", nargs='+', default=[], help="List of dataset names to process (without file extension)")
    return parser.parse_args()

def setup_directories(args):
    os.makedirs(args.output_dir, exist_ok=True)
    return args.data_dir

def load_and_process_dataset(dataset_name, data_dir, output_dir, logger):
    try:
        # Thử đọc file Excel trước
        file_path = os.path.join(data_dir, f"{dataset_name}.xlsx")
        if os.path.exists(file_path):
            data = pd.read_excel(file_path)
        else:
            # Nếu không có file Excel, thử đọc file CSV
            file_path = os.path.join(data_dir, f"{dataset_name}.csv")
            data = pd.read_csv(file_path)
        #Proccessing columns here
        data.columns = [
            pd.to_datetime(col, errors='coerce').strftime('%Y-%m-%d') 
            if pd.api.types.is_datetime64_dtype(pd.to_datetime(col, errors='coerce')) 
            else col 
            for col in data.columns
        ]  
        process_data(data, dataset_name, output_dir, logger)
    except Exception as e:
        logger.error(f"Error processing {dataset_name}: {e}")

def process_data(data, dataset_name, output_dir, logger):
    logger.info(f"Processing dataset: {dataset_name}")
    output_file = os.path.join(output_dir, f"instruction_for_profile_{dataset_name}.jsonl")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for _, row in data.iterrows():
            profile = create_profile(row)
            f.write(json.dumps(profile) + "\n")
    
    logger.info(f"Processed {len(data)} instances for {dataset_name}")

def create_profile(row):
    profile_instruction = "Construct a concise inventory time-series profile description " \
                          "including all the following information:\n"
    for column, value in row.items():
        profile_instruction += f"{column}: {value};\n"
    return profile_instruction.strip()

def main():
    logger = setup_logging()
    args = parse_arguments()
    data_dir = setup_directories(args)

    # Nếu không có dataset_names được chỉ định, xử lý tất cả các file Excel và CSV trong thư mục data
    if not args.dataset_names:
        args.dataset_names = [f.split('.')[0] for f in os.listdir(data_dir) if f.endswith(('.xlsx', '.csv'))]

    for dataset_name in args.dataset_names:
        logger.info(f"Processing dataset: {dataset_name}")
        load_and_process_dataset(dataset_name, data_dir, args.output_dir, logger)

    logger.info("All datasets processed successfully")

if __name__ == "__main__":
    main()
    sys.exit(0)
    #python run_step1_get_instruction.py --data_dir ./data --output_dir ./output --dataset_names products_finpt_data