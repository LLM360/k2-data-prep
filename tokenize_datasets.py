import os
import fire
import glob
import tqdm
import json
from functools import partial
from transformers import AutoTokenizer
import multiprocessing


CHUNKED_DATASETS_DIR = '/lustre/scratch/shared-folders/llm_project/bowen.tan/chunked_datasets'
OUTPUT_DIR = '/lustre/scratch/shared-folders/llm_project/bowen.tan/tokenized_datasets'
TOKENIZER_NAME = 'huggyllama/llama-7b'
WORD_BUFFER_SIZE = 2048 * 2
CONTEXT_LENGTH = 2048
N_BATCHES_PER_FILE = 4

ADDITIONAL_SPECIAL_TOKENS = [
    "<fim_prefix>",
    "<fim_middle>",
    "<fim_suffix>",
    "<fim_pad>",
    "<filename>",
    "<gh_stars>",
    "<issue_start>",
    "<issue_comment>",
    "<issue_closed>",
    "<jupyter_start>",
    "<jupyter_text>",
    "<jupyter_code>",
    "<jupyter_output>",
    "<empty_output>",
    "<commit_before>",
    "<commit_msg>",
    "<commit_after>",
    "<reponame>"
]


def tokenize_text(text, tokenizer):
    text = [text]
    for special_token in tokenizer.all_special_tokens:
        text_upd = []
        for cut in text:
            for i, t in enumerate(cut.split(special_token)):
                if i != 0:
                    text_upd.append(special_token)
                text_upd.append(t)
        text = text_upd

    token_ids = []
    for cut in text:
        token_ids.extend(tokenizer(cut, add_special_tokens=False)['input_ids'])

    return token_ids


def example_iterator(filename, subset_name, eos_token):
    if subset_name == 'refinedweb' or subset_name.startswith('starcoder'):
        text_key = 'content'
    else:
        text_key = 'text'

    word_buffer = []
    for line in tqdm.tqdm(open(filename), desc=filename):
        example = json.loads(line)
        if isinstance(example[text_key], dict):
            assert subset_name == 'pile-of-law'
            text = example[text_key][text_key]
        else:
            text = example[text_key]

        text = text + eos_token

        word_buffer.extend(text.split(' '))
        while len(word_buffer) >= WORD_BUFFER_SIZE:
            yield ' '.join(word_buffer[:WORD_BUFFER_SIZE])
            word_buffer = word_buffer[WORD_BUFFER_SIZE:]

    yield ' '.join(word_buffer)


def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.add_special_tokens(
        {'additional_special_tokens': ADDITIONAL_SPECIAL_TOKENS})
    tokenize_fn = partial(tokenize_text, tokenizer=tokenizer)

    for subset_name in os.listdir(CHUNKED_DATASETS_DIR):
        for chunk_filename in glob.glob(
                f'{CHUNKED_DATASETS_DIR}/{subset_name}/*.jsonl'):
            output_dir = f'{OUTPUT_DIR}/{subset_name}'
            output_filename = f'{output_dir}/' + chunk_filename.split('/')[-1]

            if os.path.exists(output_filename):
                print(f'{output_filename} exists. skipped.')
                continue

            os.makedirs(output_dir, exist_ok=True)
            with open(output_filename, 'w') as output_file:
                examples = []
                for example in example_iterator(
                        filename=chunk_filename,
                        subset_name=subset_name,
                        eos_token=tokenizer.eos_token):
                    examples.append(example)

                pool = multiprocessing.Pool(processes=os.cpu_count())

                batch_size = len(examples) // N_BATCHES_PER_FILE + 1
                token_buffer = []
                for i in range(0, len(examples), batch_size):
                    for token_ids in tqdm.tqdm(
                            pool.map(tokenize_fn, examples[i:i+batch_size]),
                            desc=f'Writing batch {i // batch_size}'):
                        token_buffer.extend(token_ids)
                        while len(token_buffer) >= CONTEXT_LENGTH:
                            output_file.write(json.dumps({
                                'token_ids': token_buffer[:CONTEXT_LENGTH],
                                'source': subset_name
                            }) + '\n')
                            token_buffer = token_buffer[CONTEXT_LENGTH:]


if __name__ == '__main__':
    fire.Fire(main)
