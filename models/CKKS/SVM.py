"""
ckks_svm.py
===========
CKKS-encrypted inference for FL-SVM using TenSEAL.
All model code identical to SVM.py.
TF-IDF fitted on training data only — test set transformed using
training IDF weights (fixes data leakage in original SVM.py).

Runs on both datasets:
  - Promoter : data_padded.csv        / test_data_padded.csv
  - CVI      : train_cvi_padded.csv   / test_cvi_padded.csv

CKKS PARAMETERS:
  poly_modulus_degree : 16384
  coeff_mod_bit_sizes : [60, 40, 40, 60]
  scale               : 2^40
  circuit depth       : 1 (single mm + bias, no activation)
  security level      : 128-bit
"""

import numpy as np
import time
import copy
from multiprocessing import cpu_count
import torch.multiprocessing as mp
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import product
import tenseal as ts
import warnings
warnings.filterwarnings("ignore")

def make_ckks_context():
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=16384,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    ctx.global_scale = 2 ** 40
    ctx.generate_galois_keys()
    return ctx

def get_kmer_vocabulary(k=4):
    bases = ['A', 'C', 'G', 'T']
    return [''.join(p) for p in product(bases, repeat=k)]

def build_model_architecture(input_size):
    return nn.Linear(input_size, 1)

def hinge_loss(outputs, targets):
    return torch.mean(torch.clamp(1 - outputs * targets, min=0))

def train_client_process(model_state, X_local, y_local, client_id,
                         input_size, epochs, threads_per_worker):
    try:
        torch.set_num_threads(threads_per_worker)
        model = build_model_architecture(input_size)
        if model_state:
            model.load_state_dict(model_state)
        optimizer = optim.SGD(model.parameters(), lr=0.005)
        y_local = np.where(y_local == 0, -1, 1)
        X_tensor = torch.FloatTensor(X_local)
        y_tensor = torch.FloatTensor(y_local).view(-1, 1)
        train_percent = int(len(X_tensor) * 0.8)
        loader = DataLoader(
            TensorDataset(X_tensor[:train_percent], y_tensor[:train_percent]),
            batch_size=256, shuffle=True
        )
        model.train()
        for _ in range(epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                hinge_loss(model(batch_X), batch_y).backward()
                optimizer.step()
        cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
        model.eval()
        with torch.no_grad():
            logits    = model(X_tensor[train_percent:])
            preds     = (logits > 0).int()
            y_eval    = (y_tensor[train_percent:] > 0).int()
            local_acc = (preds == y_eval).float().mean().item()
        return {'client_id': client_id, 'state_dict': cpu_state,
                'local_accuracy': local_acc}
    except Exception as e:
        return {'client_id': client_id, 'state_dict': None, 'error': str(e)}

class PyTorchModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = build_model_architecture(input_size)

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            return (self.model(torch.FloatTensor(X)) > 0).int().numpy().ravel()

    def evaluate(self, X_test, y_test):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.FloatTensor(X_test)).numpy().ravel()
            y_pred = (logits > 0).astype(int)
        return {
            "accuracy":         np.mean(y_pred == y_test),
            "f1":               f1_score(y_test, y_pred),
            "precision":        precision_score(y_test, y_pred, zero_division=0),
            "recall":           recall_score(y_test, y_pred, zero_division=0),
            "roc_auc":          roc_auc_score(y_test, logits),
            "pr_auc":           average_precision_score(y_test, logits),
            "confusion_matrix": confusion_matrix(y_test, y_pred)
        }

    def get_state_dict(self): return copy.deepcopy(self.model.state_dict())
    def set_state_dict(self, sd): self.model.load_state_dict(sd)
class FederatedLearning:
    def __init__(self, num_clients, input_size,
                 global_epochs=5, local_epochs=5, max_workers=None):
        self.num_clients   = num_clients
        self.global_epochs = global_epochs
        self.local_epochs  = local_epochs
        self.max_workers   = max_workers or cpu_count()
        self.input_size    = input_size
        self.global_model  = PyTorchModel(input_size)
        self.local_models  = [PyTorchModel(input_size) for _ in range(num_clients)]
        self.client_data   = []

    def aggregate_weights(self):
        sds = [m.get_state_dict() for m in self.local_models]
        avg = {k: torch.stack([sd[k] for sd in sds]).mean(dim=0) for k in sds[0]}
        self.global_model.set_state_dict(avg)

    def train_clients_parallel(self):
        gs  = self.global_model.get_state_dict()
        tpw = max(1, self.max_workers // self.num_clients)
        args = [(gs, *self.client_data[i], i, self.input_size,
                 self.local_epochs, tpw) for i in range(self.num_clients)]
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=self.num_clients) as pool:
            results = pool.starmap(train_client_process, args)
        for r in results:
            if r.get("state_dict") is not None:
                self.local_models[r["client_id"]].set_state_dict(r["state_dict"])
                print(f"    Client {r['client_id']} local_acc={r['local_accuracy']:.4f}")
            else:
                print(f"    Client {r['client_id']} FAILED: {r.get('error')}")

    def train(self, X, y, X_test, y_test):
        self.X_test_shortened = X_test[:500]
        self.y_test_shortened = y_test[:500].astype(int)
        skf = StratifiedKFold(n_splits=self.num_clients, shuffle=True, random_state=42)
        for _, idx in skf.split(X, y):
            self.client_data.append((X[idx], y[idx]))
        for epoch in range(self.global_epochs):
            print(f"\nGlobal Epoch {epoch+1}/{self.global_epochs}")
            self.train_clients_parallel()
            self.aggregate_weights()
            m = self.global_model.evaluate(X_test, y_test.astype(int))
            print(f"  Accuracy : {m['accuracy']:.4f}  F1: {m['f1']:.4f}  "
                  f"ROC-AUC: {m['roc_auc']:.4f}  PR-AUC: {m['pr_auc']:.4f}")
            print(f"  Confusion Matrix:\n{m['confusion_matrix']}")
        return self.global_model

    def predict_ckks(self, ctx, n_samples=500):
        self.global_model.model.eval()
        W = self.global_model.model.weight.data.numpy().astype(np.float64)
        b = self.global_model.model.bias.data.numpy().astype(np.float64)

        X_eval = self.X_test_shortened[:n_samples]
        preds, lats = [], []

        print(f"\nCKKS encrypted inference on {len(X_eval)} samples...")
        print(f"  Circuit depth  : 1 (linear only, no activation)")
        print(f"  poly_mod_degree: 16384  |  scale: 2^40  |  security: 128-bit")

        for i, x in enumerate(X_eval):
            t0 = time.time()

            t_enc = time.time()
            enc   = ts.ckks_vector(ctx, x.astype(np.float64).tolist())
            t_enc = time.time() - t_enc

            t_eval = time.time()
            enc    = enc.mm(W.T) + b.tolist()
            t_eval = time.time() - t_eval

            t_dec = time.time()
            logit = float(np.array(enc.decrypt())[0])
            pred  = int(logit > 0)
            t_dec = time.time() - t_dec

            preds.append(pred)
            lats.append({"encoding_s": t_enc, "ckks_eval_s": t_eval,
                         "decoding_s": t_dec, "total_s": time.time() - t0})

            if (i + 1) % 100 == 0:
                avg = np.mean([s["total_s"] for s in lats])
                eta = avg * (len(X_eval) - i - 1)
                print(f"  [{i+1}/{len(X_eval)}] avg {avg:.4f}s/sample  ETA: {eta:.1f}s")

        enc_t  = [s["encoding_s"]  for s in lats]
        eval_t = [s["ckks_eval_s"] for s in lats]
        dec_t  = [s["decoding_s"]  for s in lats]
        tot_t  = [s["total_s"]     for s in lats]
        wall   = sum(tot_t)

        print(f"\n  Total wall-clock       : {wall:.2f}s")
        print(f"  Throughput             : {len(X_eval)/wall:.4f} samples/sec")
        print(f"\n  {'Phase':<28} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
        print(f"  {'-'*60}")
        for label, vals in [
            ("Encoding (pre-CKKS)",   enc_t),
            ("CKKS Evaluation",       eval_t),
            ("Decoding (post-CKKS)",  dec_t),
            ("Total per sample",      tot_t),
        ]:
            print(f"  {label:<28} {np.mean(vals):>8.4f} {np.std(vals):>8.4f} "
                  f"{np.min(vals):>8.4f} {np.max(vals):>8.4f}")

        return np.array(preds), lats

def load_sequences(csv_path):
    data = pd.read_csv(csv_path)
    data.columns = data.columns.str.strip().str.replace('"','').str.replace("'",'')
    seqs = data["sequence"].str.upper().tolist()
    y    = data["label"].values.astype(np.float32)
    return seqs, y

def build_features(train_seqs, test_seqs, k=4):
    vocab      = get_kmer_vocabulary(k)
    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(k, k),
        vocabulary=vocab, lowercase=False, norm="l2"
    )
    X_train = vectorizer.fit_transform(train_seqs).toarray().astype(np.float32)
    X_test  = vectorizer.transform(test_seqs).toarray().astype(np.float32)
    print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Test : {X_test.shape[0]} samples,  {X_test.shape[1]} features")
    return X_train, X_test

# --- MAIN ---

if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    DATASETS = {
        "Promoter": ("data_padded.csv",       "test_data_padded.csv"),
        "CVI":      ("train_cvi_padded.csv",   "test_cvi_padded.csv"),
    }

    ctx = make_ckks_context()
    print("  poly_modulus_degree : 16384")
    print("  coeff_mod_bit_sizes : [60, 40, 40, 60]")
    print("  scale               : 2^40")
    print("  security level      : 128-bit")

    for dataset_name, (train_csv, test_csv) in DATASETS.items():
        print("\n" + "#"*60)
        print(f"# DATASET: {dataset_name.upper()}")
        print("#"*60)
        train_seqs, y_train = load_sequences(train_csv)
        test_seqs,  y_test  = load_sequences(test_csv)
        X_train, X_test = build_features(train_seqs, test_seqs, k=4)

        fl = FederatedLearning(
            num_clients=5,
            input_size=X_train.shape[1],
            global_epochs=20,
            local_epochs=20,
        )

        fl.train(X_train, y_train, X_test, y_test)

        print("\n--- Plaintext evaluation ---")
        m_plain = fl.global_model.evaluate(X_test, y_test.astype(int))
        for k in ["accuracy", "f1", "precision", "recall", "roc_auc", "pr_auc"]:
            print(f"  {k:<12}: {m_plain[k]:.4f}")
        print(f"  confusion_matrix:\n{m_plain['confusion_matrix']}")

        preds, lats = fl.predict_ckks(ctx, n_samples=500)
        y_true = fl.y_test_shortened

        print(f"\n{'='*50}")
        print(f"FINAL CKKS INFERENCE RESULTS — {dataset_name}")
        print(f"{'='*50}")
        print(f"  Samples evaluated : {len(y_true)}")
        print(f"  Accuracy          : {np.mean(preds==y_true)*100:.2f}%")
        print(f"  F1 Score          : {f1_score(y_true,preds,average='binary'):.4f}")
        print(f"  Precision         : {precision_score(y_true,preds,zero_division=0):.4f}")
        print(f"  Recall            : {recall_score(y_true,preds,zero_division=0):.4f}")
        print(f"  Confusion Matrix  :\n{confusion_matrix(y_true,preds)}")
        print(f"{'='*50}")
