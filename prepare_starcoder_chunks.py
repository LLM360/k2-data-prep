import fire
import os
import tqdm
import json
from datasets import load_dataset


DATA_DIRS = ['ada', 'agda', 'alloy', 'antlr', 'applescript', 'assembly', 'augeas', 'awk', 'batchfile', 'bluespec', 'c-sharp', 'c', 'clojure', 'cmake', 'coffeescript', 'common-lisp', 'cpp', 'css', 'cuda', 'dart', 'dockerfile', 'elixir', 'elm', 'emacs-lisp', 'erlang', 'f-sharp', 'fortran', 'git-commits-cleaned', 'github-issues-filtered-structured', 'glsl', 'go', 'groovy', 'haskell', 'html', 'idris', 'isabelle', 'java-server-pages', 'java', 'javascript', 'json', 'julia', 'jupyter-scripts-dedup-filtered', 'jupyter-structured-clean-dedup', 'kotlin', 'lean', 'literate-agda', 'literate-coffeescript', 'literate-haskell', 'lua', 'makefile', 'maple', 'markdown', 'mathematica', 'matlab', 'ocaml', 'pascal', 'perl', 'php', 'powershell', 'prolog', 'protocol-buffer', 'python', 'r', 'racket', 'restructuredtext', 'rmarkdown', 'ruby', 'rust', 'sas', 'scala', 'scheme', 'shell', 'smalltalk', 'solidity', 'sparql', 'sql', 'stan', 'standard-ml', 'stata', 'systemverilog', 'tcl', 'tcsh', 'tex', 'thrift', 'typescript', 'verilog', 'vhdl', 'visual-basic', 'xslt', 'yacc', 'yaml', 'zig']
CACHE_DIR = '/lustre/home/bowen.tan/dataset_cache/starcoderdata'
OUTPUT_DIR = '/lustre/scratch/shared-folders/llm_project/bowen.tan/chunked_datasets'
CHUNK_WORD_SIZE = 10 ** 9


def main():
    for data_dir in DATA_DIRS:
        subset_dir = f'{OUTPUT_DIR}/starcoder.{data_dir}'
        if os.path.exists(subset_dir):
            print(f'{subset_dir} exists. skiped.')
            continue
        else:
            os.makedirs(subset_dir)

        dataset = load_dataset(
            "bigcode/starcoderdata",
            data_dir=data_dir,
            split="train",
            cache_dir=CACHE_DIR)

        file_idx = 0
        output_file = open(f'{subset_dir}/{file_idx}.jsonl', 'w')
        n_chunk_words = 0
        for example in tqdm.tqdm(dataset, desc=data_dir):
            output_file.write(json.dumps(example) + '\n')

            n_chunk_words += len(example['content'].split())
            if n_chunk_words >= CHUNK_WORD_SIZE:
                file_idx += 1
                output_file = open(f'{subset_dir}/{file_idx}.jsonl', 'w')
                n_chunk_words = 0


if __name__ == '__main__':
    fire.Fire(main)
