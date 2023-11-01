import copy
import datasets


def get_preprocessed_cmg(tokenizer, split):
    dataset = datasets.load_dataset("samsum", split=split)

    prompt = (
        f"Give a short commit message for code from git diff:\n{{diff}}\n---\Short commit message:\n"
    )

    def apply_prompt_template(sample):
        return {
            "prompt": prompt.format(diff=sample["diff"]),
            "msg": sample["msg"],
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False, max_length = 991, truncation=True)
        msg = tokenizer.encode(sample["msg"] +  tokenizer.eos_token, add_special_tokens=False, max_length = 32)

        pad = tokenizer.encode(tokenizer.eos_token, max_length = 1024 - len(prompt) - len(msg), padding = 'max_length')

        sample = {
            "input_ids": prompt + msg + pad,
            "attention_mask" : [1] * (len(prompt) + len(msg) + len(pad)),
            "labels": [-100] * len(prompt) + msg + [-100] * len(pad),
            }

        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset