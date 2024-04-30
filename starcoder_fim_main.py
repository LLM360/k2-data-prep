import os
import fire
import glob
import tqdm
import json
import numpy as np
from functools import partial
from transformers import AutoTokenizer
import multiprocessing

TOKENIZED_DATASETS_DIR = '/lustre/scratch/shared-folders/llm_project/bowen.tan/tokenized_datasets'

FIM_RATE = 1.
OUTPUT_DIR = f'/lustre/scratch/shared-folders/llm_project/bowen.tan/fim_datasets'

TOKENIZER_NAME = 'huggyllama/llama-7b'
WORD_BUFFER_SIZE = 2048 * 2
CONTEXT_LENGTH = 2048
MULTIPROCESSING_BUFFERSIZE = 12800
MULTIPROCESSING_CHUNKSIZE = 100

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
    return tokenizer(text, add_special_tokens=False)['input_ids']


# From https://github.com/EleutherAI/gpt-neox/blob/FIM-clean/megatron/data/gpt2_dataset.py#L339
def permute(sample, tokenizer, fim_rate, spm_rate, truncate_or_pad):
    """
    Take in a sample (np array w/ size (0,chunklength)) and perform a FIM transformation on it.
    Maintain the same sample length (if transform creates a few extra tokens, drop them).
    """
    suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id = (
        tokenizer.vocab[tok]
        for tok in ['<fim_suffix>', '<fim_prefix>', '<fim_middle>', '<fim_pad>']
    )

    if np.random.binomial(1, fim_rate):  # sample bernoulli dist

        contents = tokenizer.decode(sample, skip_special_tokens=False)

        try:
            # A boundary can be =0 (prefix will be empty)
            # a boundary can be =len(contents) (suffix will be empty)
            # The two boundaries can be equal (middle will be empty)
            boundaries = list(
                np.random.randint(low=0, high=len(contents) + 1, size=2))
            boundaries.sort()
        except ValueError as e:
            print(len(contents), contents)
            print(e)
            raise e

        prefix = contents[:boundaries[0]]
        middle = contents[boundaries[0]:boundaries[1]]
        suffix = contents[boundaries[1]:]

        prefix = np.array(
            tokenize_text(prefix, tokenizer=tokenizer), dtype=np.int64)
        middle = np.array(
            tokenize_text(middle, tokenizer=tokenizer), dtype=np.int64)
        suffix = np.array(
            tokenize_text(suffix, tokenizer=tokenizer), dtype=np.int64)

        # here we truncate each given segment to fit the same length as it was before
        # A consequence is that we never reach the end of a file?
        # we should rather truncate at the context-level
        if truncate_or_pad:
            # need to make same length as the input. Take the 3 sentinel tokens into account
            new_length = suffix.shape[0] + prefix.shape[0] + middle.shape[0] + 3
            diff = new_length - sample.shape[0]
            if diff > 0:  # too long
                if suffix.shape[
                    0] <= diff:  # if there's no space to truncate the suffix: stop and report it. atm i should have stopped this from happening
                    return sample
                suffix = suffix[:suffix.shape[0] - diff]
            elif diff < 0:  # too short
                suffix = np.concatenate(
                    [suffix, np.full((-1 * diff), pad_tok_id)])

        if np.random.binomial(1, spm_rate):
            # SPM (variant 2 from FIM paper)
            new_sample = np.concatenate([
                [prefix_tok_id, suffix_tok_id], suffix,
                [middle_tok_id], prefix, middle
            ])
        else:
            # PSM
            new_sample = np.concatenate([
                [prefix_tok_id], prefix,
                [suffix_tok_id], suffix,
                [middle_tok_id], middle
            ])

    else:
        # don't do FIM preproc
        new_sample = sample

    return new_sample


def fim(token_ids, tokenizer, fim_rate, spm_rate):
    sample = np.array(token_ids, dtype=np.int64)
    sample_len = sample.shape[0]

    permute_fn = partial(
        permute,
        tokenizer=tokenizer,
        fim_rate=fim_rate,
        spm_rate=spm_rate,
        truncate_or_pad=False)

    if fim_rate != 0:
        assert (fim_rate <= 1 and fim_rate >= 0), \
            "FIM rate must be a probability 0 <= rate <= 1"

        eod = tokenizer.eos_token_id
        pad = tokenizer.vocab['<fim_pad>']

        segment_breaks = np.argwhere(sample == eod)  # split sample by document

        if segment_breaks.shape != (0, 1):
            # then there is an EOD token in this example
            curr_start_position = 0
            new_samples = []
            for loc in np.nditer(segment_breaks):
                # Only permute non-empty segments.
                if loc - curr_start_position > 0:
                    # permute {prefix, suffix, middle} or {suffix, prefix, middle}
                    permuted = permute_fn(
                        sample=sample[curr_start_position:loc])
                    new_samples += [permuted, [eod]]

                curr_start_position = loc + 1  # jump over the EOD token
            # Permute the segment after the last EOD
            permuted = permute_fn(sample=sample[curr_start_position:])
            new_samples.append(permuted)

            sample = np.concatenate(new_samples)
        else:
            sample = permute_fn(sample=sample)

    # Truncate or pad sequence to max-length
    diff = sample.shape[0] - sample_len
    if diff > 0:  # too long
        sample = sample[:sample_len]
    elif diff < 0:  # too short
        sample = np.concatenate([sample, np.full((-1 * diff), pad)])

    assert sample.shape[0] == sample_len
    # end FIM-specific code
    return sample.tolist()
    # return {'text': np.array(sample, dtype=np.int64)}


def process_example(example, tokenizer, fim_rate, spm_rate):
    example['token_ids'] = fim(
        token_ids=example['token_ids'],
        tokenizer=tokenizer,
        fim_rate=fim_rate,
        spm_rate=spm_rate)
    return example


def main(spm_rate=0.):
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.add_special_tokens(
        {'additional_special_tokens': ADDITIONAL_SPECIAL_TOKENS})
    process_fn = partial(
        process_example,
        tokenizer=tokenizer,
        fim_rate=FIM_RATE,
        spm_rate=spm_rate)

    for subset_name in os.listdir(f'{TOKENIZED_DATASETS_DIR}/'):
        if not subset_name.startswith('starcoder.'):
            continue

        for tokenized_chunk_filename in glob.glob(
                f'{TOKENIZED_DATASETS_DIR}/{subset_name}/*.jsonl'):
            subset_output_dir = f'{OUTPUT_DIR}/{subset_name}.spm{spm_rate}'
            output_filename = (
                    f'{subset_output_dir}/' +
                    tokenized_chunk_filename.split('/')[-1])

            if os.path.exists(output_filename):
                print(f'{output_filename} exists. skipped.')
                continue
            else:
                os.makedirs(subset_output_dir, exist_ok=True)

            with open(output_filename, 'w') as output_file:
                pool = multiprocessing.Pool(processes=os.cpu_count())

                buffer = []
                for line in tqdm.tqdm(
                        open(tokenized_chunk_filename), desc=output_filename):
                    buffer.append(json.loads(line))

                    if len(buffer) == MULTIPROCESSING_BUFFERSIZE:
                        for example in pool.map(
                                process_fn,
                                buffer,
                                chunksize=MULTIPROCESSING_CHUNKSIZE):
                            output_file.write(json.dumps(example) + '\n')
                        buffer = []

                for example in pool.map(process_fn, buffer):
                    output_file.write(json.dumps(example) + '\n')


if __name__ == '__main__':
    fire.Fire(main)
