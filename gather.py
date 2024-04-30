import json
import fire
import os
import tqdm
import random
import time
import multiprocessing


TOKENIZED_DATASETS_DIR = '/lustre/scratch/shared-folders/llm_project/bowen.tan/tokenized_datasets'
TOKENIZED_FIM_DATASETS_DIR = '/lustre/scratch/shared-folders/llm_project/bowen.tan/fim_datasets'
OUTPUT_DIR = '/lustre/scratch/shared-folders/llm_project/bowen.tan/final_data_chunks'

# TOKENIZED_DATASETS_DIR = '/mount/data/s3'
# TOKENIZED_FIM_DATASETS_DIR = '/mount/data/fim'
# OUTPUT_DIR = '/mount/data/data_chunks'

N_CHUNKS = 360
CHUNK_BATCH_SIZE = 16
FIM_RATE = 0.5
SPM_RATE = 0.5


def distribute_samples_starcoder(subset_name,
                                 filename,
                                 output_file,
                                 file_idx_to_write,
                                 tgt_file_idx,
                                 n_repeat):
    examples_path = f'{TOKENIZED_DATASETS_DIR}/{subset_name}/{filename}'
    fim_path = f'{TOKENIZED_FIM_DATASETS_DIR}/{subset_name}.spm0.0/{filename}'
    spm_path = f'{TOKENIZED_FIM_DATASETS_DIR}/{subset_name}.spm1.0/{filename}'

    pseudo_n_chunks = int(N_CHUNKS / n_repeat)
    file_idx_to_write = file_idx_to_write % pseudo_n_chunks

    random.seed(time.time() + tgt_file_idx)

    for line_idx, (line, line_fim, line_spm) in tqdm.tqdm(
            enumerate(zip(open(examples_path), open(fim_path), open(spm_path))),
            desc=f'{subset_name}/{filename} (pseudo_chunks={pseudo_n_chunks})'):
        if file_idx_to_write == tgt_file_idx % pseudo_n_chunks:
            src_filename = examples_path
            if random.random() < FIM_RATE:
                line = line_fim
                src_filename = fim_path
                if random.random() < SPM_RATE:
                    line = line_spm
                    src_filename = spm_path

            example = json.loads(line)
            example['subset_name'] = subset_name
            example['src_filename'] = src_filename
            example['line_idx_in_src'] = line_idx

            output_file.write(json.dumps(example) + '\n')
            output_file.flush()

        file_idx_to_write = (file_idx_to_write + 1) % pseudo_n_chunks

    return file_idx_to_write


def distribute_samples(subset_name,
                       filename,
                       output_file,
                       file_idx_to_write,
                       tgt_file_idx,
                       n_repeat):
    examples_path = f'{TOKENIZED_DATASETS_DIR}/{subset_name}/{filename}'

    pseudo_n_chunks = int(N_CHUNKS / n_repeat)
    file_idx_to_write = file_idx_to_write % pseudo_n_chunks

    for line_idx, line in tqdm.tqdm(
            enumerate(open(examples_path)),
            desc=f'{subset_name}/{filename} (pseudo_chunks={pseudo_n_chunks})'):
        if file_idx_to_write == tgt_file_idx % pseudo_n_chunks:
            example = json.loads(line)
            example['subset_name'] = subset_name
            example['src_filename'] = examples_path
            example['line_idx_in_src'] = line_idx

            output_file.write(json.dumps(example) + '\n')
            output_file.flush()

        file_idx_to_write = (file_idx_to_write + 1) % pseudo_n_chunks

    return file_idx_to_write


def gather_chunk(tgt_file_idx):
    output_file = open(f'{OUTPUT_DIR}/chunk_{tgt_file_idx}.jsonl', 'w')

    file_idx_to_write = 0
    for subset_name in sorted(os.listdir(TOKENIZED_DATASETS_DIR)):
        if subset_name == 'redpajama.c4':
            continue
        elif subset_name in ['dm-math', 'pubmed-abstracts', 'uspto']:
            n_repeat = 3
        elif subset_name == 'redpajama.wikipedia':
            n_repeat = 6
        elif subset_name == 'redpajama.arxiv':
            n_repeat = 1
        elif subset_name.startswith('redpajama.'):
            n_repeat = 3
        elif subset_name.startswith('starcoder.'):
            n_repeat = 0.5
        else:
            n_repeat = 1

        for filename in sorted(os.listdir(
                f'{TOKENIZED_DATASETS_DIR}/{subset_name}')):
            assert filename.endswith(f'.jsonl')

            if subset_name.startswith('starcoder.'):
                file_idx_to_write = distribute_samples_starcoder(
                    subset_name=subset_name,
                    filename=filename,
                    output_file=output_file,
                    file_idx_to_write=file_idx_to_write,
                    tgt_file_idx=tgt_file_idx,
                    n_repeat=n_repeat)
            else:
                file_idx_to_write = distribute_samples(
                    subset_name=subset_name,
                    filename=filename,
                    output_file=output_file,
                    file_idx_to_write=file_idx_to_write,
                    tgt_file_idx=tgt_file_idx,
                    n_repeat=n_repeat)

    return tgt_file_idx


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for l in tqdm.trange(
            0, N_CHUNKS, CHUNK_BATCH_SIZE, desc='Processing chunks'):
        with multiprocessing.Pool(processes=CHUNK_BATCH_SIZE) as pool:
            results = pool.map(
                gather_chunk, list(range(l, l + CHUNK_BATCH_SIZE)))

        print(results)


if __name__ == '__main__':
    fire.Fire(main)
