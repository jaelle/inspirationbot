from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import torch.nn.functional as F
import numpy as np

def generate(prompt_text,length=64,temperature=1.0,top_k=0,top_p=0.9,num_return_sequences=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_dir = '/content/inspirationbot/output/'
    tokenizer = GPT2Tokenizer.from_pretrained(local_dir)
    model = GPT2LMHeadModel.from_pretrained(local_dir)
    model = model.to(device)

    input_ids  = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
    input_ids = input_ids.to(device)

    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=length + len(input_ids[0]),
        temperature=temperature, 
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
        pad_token_id=50256,
        num_return_sequences=num_return_sequences)
    
    if len(output_sequences.shape) > 2:
        output_sequences.squeeze_()

    generated_sequences = []

    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        generated_sequence = generated_sequence.tolist()

        # Decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

        # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
        total_sequence = (
            prompt_text + text[len(tokenizer.decode(input_ids[0], clean_up_tokenization_spaces=True)) :]
        )

        generated_sequences.append(total_sequence)

    return generated_sequences