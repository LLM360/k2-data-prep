import os
import fire
import json
import tqdm
from datasets import load_dataset


os.environ["RED_PAJAMA_DATA_DIR"] = "/lustre/home/bowen.tan/dataset_cache/redpajama"
REDPAJAME_SUBSETS = ['arxiv', 'book', 'c4', 'stackexchange', 'wikipedia']
OUTPUT_DIR = '/lustre/scratch/shared-folders/llm_project/bowen.tan/chunked_datasets'
CHUNK_WORD_SIZE = 10 ** 9


def main():
    for subset_name in REDPAJAME_SUBSETS:
        subset_dir = f'{OUTPUT_DIR}/redpajama.{subset_name}'
        if os.path.exists(subset_dir):
            print(f'{subset_dir} exists. skiped.')
            continue
        else:
            os.makedirs(subset_dir)

        dataset = load_dataset(
            'togethercomputer/RedPajama-Data-1T',
            name=subset_name,
            split='train',
            streaming=True)

        file_idx = 0
        output_file = open(f'{subset_dir}/{file_idx}.jsonl', 'w')
        n_chunk_words = 0
        for example in tqdm.tqdm(dataset, desc=subset_dir):
            output_file.write(json.dumps(example) + '\n')

            n_chunk_words += len(example['text'].split())
            if n_chunk_words >= CHUNK_WORD_SIZE:
                file_idx += 1
                output_file = open(f'{subset_dir}/{file_idx}.jsonl', 'w')
                n_chunk_words = 0


if __name__ == '__main__':
    fire.Fire(main)
