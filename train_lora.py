"""LoRA fine-tuning of a small causal LM for 3-class sentiment (Unit II).
Run from the project root (a free Colab GPU is enough; CPU works but is slow):
    python -m modules.train_lora --epochs 5
It writes models/lora_adapter/ including training_info.json with REAL measured numbers."""
import argparse
import json
import os
import time
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from modules.lora_model import BASE_MODEL, format_prompt, parse_label
from utils.dataset_loader import load_lora_data, split_data

OUT = "models/lora_adapter"


def encode(tok, text, label):
    p = tok(format_prompt(text), add_special_tokens=False)["input_ids"]
    a = tok(" " + label, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    return {"input_ids": p + a, "labels": [-100] * len(p) + a}  # loss only on the answer tokens


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.tensor([b["input_ids"] + [pad_id] * (n - len(b["input_ids"])) for b in batch])
    lab = torch.tensor([b["labels"] + [-100] * (n - len(b["labels"])) for b in batch])
    return {"input_ids": ids, "labels": lab, "attention_mask": (ids != pad_id).long()}


@torch.no_grad()
def accuracy(model, tok, test, device):
    model.eval()
    correct = 0
    for _, r in test.iterrows():
        x = tok(format_prompt(r["text"]), return_tensors="pt").to(device)
        out = model.generate(**x, max_new_tokens=4, do_sample=False, pad_token_id=tok.eos_token_id)
        correct += parse_label(tok.decode(out[0][x["input_ids"].shape[1]:], skip_special_tokens=True)) == r["label"]
    return round(100 * correct / len(test), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, test = split_data(load_lora_data())
    tok = AutoTokenizer.from_pretrained(a.base)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = get_peft_model(AutoModelForCausalLM.from_pretrained(a.base).to(device),
                           LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05,
                                      target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    trainable, total = model.get_nb_trainable_parameters()
    base_acc = None
    with model.disable_adapter():  # baseline: same base model, adapter switched off
        base_acc = accuracy(model, tok, test, device)
    data = [encode(tok, r["text"], r["label"]) for _, r in train.iterrows()]
    t0 = time.time()
    tr = Trainer(model=model, train_dataset=data, data_collator=lambda b: collate(b, tok.pad_token_id),
                 args=TrainingArguments(output_dir="lora_runs", num_train_epochs=a.epochs, learning_rate=a.lr,
                                        per_device_train_batch_size=a.batch, logging_steps=2, save_strategy="no",
                                        report_to=[], remove_unused_columns=False))
    tr.train()
    secs = round(time.time() - t0, 1)
    lora_acc = accuracy(model, tok, test, device)
    os.makedirs(OUT, exist_ok=True)
    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    test.to_csv(os.path.join(OUT, "test_set.csv"), index=False)
    info = {"base_model": a.base, "task": "3-class sentiment classification", "device": device, "epochs": a.epochs,
            "learning_rate": a.lr, "lora_rank": a.rank, "lora_alpha": 2 * a.rank, "target_modules": ["q_proj", "v_proj"],
            "batch_size": a.batch, "train_examples": len(train), "test_examples": len(test),
            "trainable_params": trainable, "total_params": total, "training_seconds": secs,
            "loss_history": [l["loss"] for l in tr.state.log_history if "loss" in l],
            "base_model_test_accuracy_pct": base_acc, "lora_test_accuracy_pct": lora_acc}
    json.dump(info, open(os.path.join(OUT, "training_info.json"), "w"), indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
