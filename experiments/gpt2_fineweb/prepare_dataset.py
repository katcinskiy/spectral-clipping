"""Stream a HuggingFace dataset, GPT-2 tokenize, pack into fixed-size blocks, save locally.

Default: HuggingFaceFW/fineweb (sample-10BT), 2B tokens, block size 1024.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Sequence, Value, load_dataset
from transformers import AutoTokenizer


def iter_token_blocks(
    *,
    dataset_name, dataset_config, split, tokenizer_name, text_column,
    block_size, target_tokens, add_eos_token, trust_remote_code,
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.eos_token_id is None and add_eos_token:
        raise ValueError("Tokenizer must have an eos_token_id when add_eos_token=true.")

    stream = load_dataset(
        dataset_name, name=dataset_config, split=split,
        streaming=True, trust_remote_code=trust_remote_code,
    )
    target_blocks = target_tokens // block_size
    if target_blocks * block_size != target_tokens:
        raise ValueError("target_tokens must be divisible by block_size.")

    buffer: list[int] = []
    emitted = 0
    for example in stream:
        text = example.get(text_column)
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if add_eos_token:
            ids.append(int(tokenizer.eos_token_id))
        buffer.extend(ids)

        while len(buffer) >= block_size and emitted < target_blocks:
            yield {"input_ids": buffer[:block_size]}
            del buffer[:block_size]
            emitted += 1
            if emitted % 10_000 == 0:
                print(f"[prepare] emitted {emitted:,} blocks ({emitted * block_size:,} tokens)")
        if emitted >= target_blocks:
            break

    if emitted < target_blocks:
        raise RuntimeError(
            f"Stream ended before target. Emitted {emitted * block_size:,} / {target_tokens:,} tokens."
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-name", default="HuggingFaceFW/fineweb")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--split", default="train")
    p.add_argument("--tokenizer-name", default="gpt2")
    p.add_argument("--text-column", default="text")
    p.add_argument("--block-size", type=int, default=1024)
    p.add_argument("--target-tokens", type=int, default=2_000_000_000)
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Where to save the packed dataset (passed to load_from_disk later).")
    p.add_argument("--save-max-shard-size", default="16GB")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--no-add-eos-token", action="store_true")
    args = p.parse_args()
    add_eos = not args.no_add_eos_token

    features = Features({"input_ids": Sequence(Value("int32"), length=args.block_size)})
    dataset = Dataset.from_generator(
        iter_token_blocks,
        gen_kwargs=dict(
            dataset_name=args.dataset_name, dataset_config=args.dataset_config,
            split=args.split, tokenizer_name=args.tokenizer_name,
            text_column=args.text_column, block_size=args.block_size,
            target_tokens=args.target_tokens, add_eos_token=add_eos,
            trust_remote_code=bool(args.trust_remote_code),
        ),
        features=features,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    DatasetDict({"train": dataset}).save_to_disk(
        str(args.output_dir), max_shard_size=args.save_max_shard_size,
    )

    with (args.output_dir / "prep_manifest.json").open("w", encoding="utf-8") as f:
        json.dump({
            "dataset_name": args.dataset_name, "dataset_config": args.dataset_config,
            "split": args.split, "tokenizer_name": args.tokenizer_name,
            "text_column": args.text_column, "block_size": args.block_size,
            "target_tokens": args.target_tokens,
            "num_blocks": args.target_tokens // args.block_size,
            "add_eos_token": add_eos,
        }, f, indent=2)
        f.write("\n")
    print(f"[prepare] saved {args.target_tokens:,} tokens to {args.output_dir}")


if __name__ == "__main__":
    main()
