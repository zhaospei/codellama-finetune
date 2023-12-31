import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    default_data_collator, 
    Trainer, 
    TrainingArguments,
    TrainerCallback,
)
from utils.tokenizer import get_preprocessed_cmg, get_preprocessed_cmg_history
from contextlib import nullcontext
from tqdm import tqdm
import json
import argparse

def run(batch_size, load_in_8bit):
    dataset_id = "zhaospei/cmg-data-v2"
    model_id = "codellama/CodeLlama-7b-hf"

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=load_in_8bit, device_map='auto', torch_dtype=torch.float16)


    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training
    train_dataset = get_preprocessed_cmg_history(dataset_id, tokenizer, 'train')

    # train_dataset = get_preprocessed_dataset(tokenizer, samsum_dataset, 'train')
    # train_dataset = get_preprocessed_samsum(cmg_dataset, tokenizer, 'train')

    model.train()

    def create_peft_config(model):
        from peft import (
            get_peft_model,
            LoraConfig,
            TaskType,
            prepare_model_for_int8_training,
        )

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules = ["q_proj", "v_proj"]
        )

        # prepare int-8 model for training
        if load_in_8bit:
            model = prepare_model_for_int8_training(model)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        return model, peft_config

    # create peft config
    model, lora_config = create_peft_config(model)

    enable_profiler = False
    output_dir = "tmp/code-llama-output"

    config = {
        'lora_config': lora_config,
        'learning_rate': 1e-4,
        'num_train_epochs': 1,
        'gradient_accumulation_steps': 2,
        'per_device_train_batch_size': batch_size,
        'gradient_checkpointing': False,
    }

    # Set up profiler
    if enable_profiler:
        wait, warmup, active, repeat = 1, 1, 2, 1
        total_steps = (wait + warmup + active) * (1 + repeat)
        schedule =  torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)
        profiler = torch.profiler.profile(
            schedule=schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{output_dir}/logs/tensorboard"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True)
        
        class ProfilerCallback(TrainerCallback):
            def __init__(self, profiler):
                self.profiler = profiler
                
            def on_step_end(self, *args, **kwargs):
                self.profiler.step()

        profiler_callback = ProfilerCallback(profiler)
    else:
        profiler = nullcontext()

    # Define training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        bf16=True,  # Use BF16 if available
        # logging strategies
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="no",
        optim="adamw_torch_fused",
        max_steps=total_steps if enable_profiler else -1,
        **{k:v for k,v in config.items() if k != 'lora_config'}
    )

    with profiler:
        # Create Trainer instance
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=default_data_collator,
            callbacks=[profiler_callback] if enable_profiler else [],
        )

        # Start training
        trainer.train()

    model.save_pretrained(output_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--load_in_8bit", action='store_true',
                        help="Load model 8 bit.")

    args = parser.parse_args()
    run(args.batch_size, args.load_in_8bit)

if __name__ == '__main__':
    main()

# model.eval()

# def read_contextual_medit_examples(filename):
#     """Read examples from filename."""
#     examples = []
#     with open(filename, encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             js = json.loads(line)
#             examples.append(js['prompt'])
#     return examples

# def write_string_to_file(absolute_filename, string):
#         with open(absolute_filename, 'a') as fout:
#             fout.write(string)

# examples = read_contextual_medit_examples('test.input.jsonl')

# for eval_prompt in tqdm(examples):
#     model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

#     output = ''

#     with torch.no_grad():
#         output = tokenizer.decode(model.generate(**model_input, max_new_tokens=32, pad_token_id=tokenizer.eos_token_id)[0], skip_special_tokens=True)
#     write_string_to_file('test.codellama.reload.output', '' + output + '<nl>')