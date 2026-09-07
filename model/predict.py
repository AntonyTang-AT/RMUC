# model/predict.py
import json, re
from model import prompt

def relaxed_extract(text):
    m = re.search(r'\{.*?\}', text, re.DOTALL)
    return m.group(0) if m else None

def load_model(path_or_name, quantize=True):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch
    tok = AutoTokenizer.from_pretrained(path_or_name)
    if quantize:
        cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(path_or_name, quantization_config=cfg, device_map='auto')
    else:
        model = AutoModelForCausalLM.from_pretrained(path_or_name, device_map='auto')
    model.eval()
    return model, tok

def llm_decide(model, tok, worldstate):
    p = prompt.build_prompt(worldstate)
    inputs = tok(p, return_tensors='pt').to(model.device)
    out = model.generate(**inputs, max_new_tokens=64, do_sample=False, pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    sub = relaxed_extract(text)
    parsed = prompt.parse_decision(sub) if sub else prompt.parse_decision('')
    return {'state': parsed['state'], 'raw': text}
