import time
import os
import sys

current_directory = os.getcwd()
root_directory = os.path.normpath(os.path.join(current_directory, '..', '..'))
sys.path.append(root_directory)

from src.config.inference_config import InferenceConfig

class Logger:
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.counter = 0
        self.start_time = time.time()
        self.enable_logging = config.enable_logging

    def log_progress(self):
        if not self.enable_logging:
            return
        self.counter += 1
        if self.counter % self.config.logging_interval == 0:
            elapsed = time.time() - self.start_time
            print(f"Processed {self.counter} requests in {elapsed:.2f} seconds.")
    
    def log_total_time(self, total_items: int):
        if not self.enable_logging:
            return
        total_time = time.time() - self.start_time
        print(f"Total processing time: {total_time:.2f} seconds for {total_items} items.")
        print(f"Average time per item: {total_time/total_items:.4f} seconds.")