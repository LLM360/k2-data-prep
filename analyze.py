import json
import fire
import tqdm


OUTPUT_DIR = '/lustre/scratch/shared-folders/llm_project/bowen.tan/final_data_chunks'
N_CHUNKS = 360


def main():
    counter = {}
    for line in tqdm.tqdm(open(f'{OUTPUT_DIR}/chunk_0.jsonl')):
        example = json.loads(line)
        subset_name = example['subset_name']

        if subset_name.startswith('starcoder.'):
            subset_name = 'starcoder'
            if 'spm0.0' in example['src_filename']:
                subset_name = subset_name + '.FIM'
            elif 'spm1.0' in example['src_filename']:
                subset_name = subset_name + '.SPM'

        counter[subset_name] = counter.get(subset_name, 0) + 1

    n_total = sum(list(counter.values()))
    print(f'{n_total} samples.')
    for key, value in counter.items():
        print(f'{key}\t{value}\t{value / n_total}')


if __name__ == '__main__':
    fire.Fire(main)