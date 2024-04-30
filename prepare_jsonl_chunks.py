import os
import fire
import json
import glob
import tqdm
from datasets import load_dataset


DATA_DIRS = {
    'refinedweb': '/lustre/scratch/shared-folders/llm_project/nikhil.ranjan/falcon-refinedweb-json',
    's2orc': '/lustre/scratch/shared-folders/llm_project/all_original_json_datasets/s2orc/s2orc_proc_text',
    # 'pile-of-law': '/lustre/scratch/shared-folders/llm_project/nikhil.ranjan/freelaw/pile-of-law/data',
    'pubmed-central': '/lustre/scratch/shared-folders/llm_project/pile/train/dedup/PubMedCentral',
    'pubmed-abstracts': '/lustre/scratch/shared-folders/llm_project/pile/train/dedup/PubMedAbstracts',
    'uspto': '/lustre/scratch/shared-folders/llm_project/pile/train/dedup/USPTOBackgrounds',
    'dm-math': '/lustre/scratch/shared-folders/llm_project/pile/train/dedup/DMMathematics'
}
OUTPUT_DIR = '/lustre/scratch/shared-folders/llm_project/bowen.tan/chunked_datasets'
CHUNK_WORD_SIZE = 10 ** 9


def main():
    for dataset_name, dataset_dir in DATA_DIRS.items():
        subset_dir = f'{OUTPUT_DIR}/{dataset_name}'
        if os.path.exists(subset_dir):
            print(f'{dataset_name}:{subset_dir} exists, skipped.')
            continue

        os.makedirs(subset_dir)

        file_idx = 0
        output_file = open(f'{subset_dir}/{file_idx}.jsonl', 'w')
        n_chunk_words = 0
        for path in sorted(glob.glob(f'{dataset_dir}/*.jsonl')):
            dataset = load_dataset(
                'json', data_files=path, split='train', streaming=True)

            for example in tqdm.tqdm(dataset, desc=path):
                output_file.write(json.dumps(example) + '\n')

                text_key = 'content' if dataset_name == 'refinedweb' else 'text'

                n_chunk_words += len(example[text_key].split())
                if n_chunk_words >= CHUNK_WORD_SIZE:
                    file_idx += 1
                    output_file = open(f'{subset_dir}/{file_idx}.jsonl', 'w')
                    n_chunk_words = 0


if __name__ == '__main__':
    fire.Fire(main)
