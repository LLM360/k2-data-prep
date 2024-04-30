import os
import fire
import tqdm
import json
import random

CHUNK_DIR = '/mount/data/data_chunks'
OUTPUT_DIR = '/mount/data/shuffled_data_chunks'
N_CHUNKS = 32
SEED_BASE = 11111


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for chunk_idx in range(N_CHUNKS):
        input_filename = f'{CHUNK_DIR}/chunk_{chunk_idx}.jsonl'
        output_filename = f'{OUTPUT_DIR}/chunk_{chunk_idx}.jsonl'

        if os.path.exists(output_filename):
            continue

        lines = open(input_filename).readlines()
        random.seed(SEED_BASE + chunk_idx)
        random.shuffle(lines)

        with open(output_filename, 'w') as output_file:
            for line in tqdm.tqdm(lines, desc=output_filename):
                output_file.write(line.strip() + '\n')


if __name__ == '__main__':
    fire.Fire(main)
