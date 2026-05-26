import numpy as np
import time
import copy
from multiprocessing import cpu_count
import torch.multiprocessing as mp
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from itertools import product
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import tenseal as ts
import warnings
warnings.filterwarnings("ignore")

def make_ckks_context():
    """CKKS context for Child MLP: 3 linear + 2 quad activations = 5 levels."""
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=16384,
        coeff_mod_bit_sizes=[60, 26, 26, 26, 26, 26, 60]
    )
    ctx.global_scale = 2 ** 26
    ctx.generate_galois_keys()
    return ctx

def get_kmer_vocabulary(k=4):
    return [''.join(p) for p in product(['A','C','G','T'], repeat=k)]

def preprocess_data(csv_path, k=4, verbose=True):
    data = pd.read_csv(csv_path)
    data.columns = data.columns.str.strip().str.replace('"','').str.replace("'",'')
    sequences = data["sequence"].str.upper().tolist()
    y = data["label"].values.astype(np.float32)
    vocab = get_kmer_vocabulary(k)
    vec = CountVectorizer(analyzer='char', ngram_range=(k,k),
                          vocabulary=vocab, lowercase=False)
    X = vec.transform(sequences).toarray().astype(np.float32)
    max_val = X.max()
    if max_val > 0: X = X / max_val
    if verbose: print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y

def compute_metrics(y_test, y_pred, probs=None):
    m = {
        "accuracy":         np.mean(y_pred == y_test),
        "f1":               f1_score(y_test, y_pred, average="binary"),
        "precision":        precision_score(y_test, y_pred, zero_division=0),
        "recall":           recall_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }
    if probs is not None:
        m["roc_auc"] = roc_auc_score(y_test, probs)
        m["pr_auc"]  = average_precision_score(y_test, probs)
    return m

def print_metrics(m, label=""):
    if label: print(f"\n--- {label} ---")
    for k in ["accuracy","f1","precision","recall","roc_auc","pr_auc"]:
        if k in m: print(f"  {k:<12}: {m[k]:.4f}")
    print(f"  confusion_matrix:\n{m['confusion_matrix']}")

def print_latency(lats):
    enc_t  = [s["encoding_s"]  for s in lats]
    eval_t = [s["ckks_eval_s"] for s in lats]
    dec_t  = [s["decoding_s"]  for s in lats]
    tot_t  = [s["total_s"]     for s in lats]
    wall   = sum(tot_t); n = len(lats)
    print(f"\n  Total wall-clock       : {wall:.2f}s")
    print(f"  Throughput             : {n/wall:.4f} samples/sec")
    print(f"  {'Phase':<28} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*60}")
    for lbl, vals in [
        ("Encoding (pre-CKKS)",   enc_t),
        ("CKKS Evaluation",       eval_t),
        ("Decoding (post-CKKS)",  dec_t),
        ("Total per sample",      tot_t),
    ]:
        print(f"  {lbl:<28} {np.mean(vals):>8.4f} {np.std(vals):>8.4f} "
              f"{np.min(vals):>8.4f} {np.max(vals):>8.4f}")

class LSTMTeacher(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, X):
        if X.dim() == 2: X = X.unsqueeze(1)
        h0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size)
        out, _ = self.lstm(X, (h0, c0))
        return self.fc(out[:, -1, :])

    def evaluate(self, X_test, y_test):
        self.eval()
        with torch.no_grad():
            probs = torch.sigmoid(self(torch.FloatTensor(X_test))).numpy().ravel()
        return compute_metrics(y_test, (probs > 0.5).astype(int), probs)

    def get_state_dict(self): return copy.deepcopy(self.state_dict())
    def set_state_dict(self, sd): self.load_state_dict(sd)

class ChildMLP(nn.Module):
    def __init__(self, input_size, hidden_size=32, output_size=1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, X):
        x = self.bn1(self.fc1(X)) ** 2 #Quadratic Approximation 
        x = self.bn2(self.fc2(x)) ** 2
        return self.fc3(x)

    def get_fused_weights(self):
        def fold(linear, bn):
            scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            shift = bn.bias - bn.running_mean * scale
            W_f = (scale.unsqueeze(1) * linear.weight).data.numpy().astype(np.float64)
            b_f = (scale * linear.bias + shift).data.numpy().astype(np.float64)
            return W_f, b_f
        W1f, b1f = fold(self.fc1, self.bn1)
        W2f, b2f = fold(self.fc2, self.bn2)
        W3 = self.fc3.weight.data.numpy().astype(np.float64)
        b3 = self.fc3.bias.data.numpy().astype(np.float64)
        return W1f, b1f, W2f, b2f, W3, b3

    def evaluate(self, X_test, y_test, threshold=0.0):
        self.eval()
        with torch.no_grad():
            logits = self(torch.FloatTensor(X_test)).numpy().ravel()
            probs  = 1 / (1 + np.exp(-logits))
        return compute_metrics(y_test, (logits > threshold).astype(int), probs)

    def get_state_dict(self): return copy.deepcopy(self.state_dict())
    def set_state_dict(self, sd): self.load_state_dict(sd)

class DistillationLoss(nn.Module):
    def __init__(self, temperature=3.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, child_logits, teacher_preds, target):
        hard = F.binary_cross_entropy_with_logits(child_logits, target)
        soft = F.binary_cross_entropy(
            torch.sigmoid(child_logits / self.temperature), teacher_preds
        )
        return self.alpha * soft + (1 - self.alpha) * hard

def distill(teacher, X_train, y_train, input_size,
            hidden_size=32, epochs=20, temperature=3.0, alpha=0.5):
    """Identical to distillation() in LSTM.py."""
    print("Performing Distillation...")
    teacher.eval()
    Xt = torch.FloatTensor(X_train)
    yt = torch.FloatTensor(y_train).view(-1, 1)
    with torch.no_grad():
        teacher_preds = torch.sigmoid(teacher(Xt) / temperature).detach()

    child     = ChildMLP(input_size, hidden_size, 1)
    loader    = DataLoader(TensorDataset(Xt, teacher_preds, yt),
                           batch_size=256, shuffle=True)
    optimizer = optim.Adam(child.parameters())
    criterion = DistillationLoss(temperature, alpha)

    child.train()
    for ep in range(epochs):
        total = 0
        for bX, bsoft, bhard in loader:
            optimizer.zero_grad()
            loss = criterion(child(bX), bsoft, bhard)
            loss.backward(); optimizer.step()
            total += loss.item()
        print(f"  Epoch {ep+1}/{epochs} loss: {total/len(loader):.4f}")

    child.eval()
    with torch.no_grad():
        threshold = float(np.median(child(Xt).numpy().ravel()))
    print(f"Distillation complete. Threshold: {threshold:.4f}")
    return child, threshold

def train_client_process(model_state, X_local, y_local, client_id,
                          hidden_size, input_size, output_size,
                          epochs, threads_per_worker, n_layers=3):
    try:
        torch.set_num_threads(threads_per_worker)
        model = LSTMTeacher(input_size, hidden_size, n_layers)
        if model_state: model.load_state_dict(model_state, strict=True)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        Xt = torch.FloatTensor(X_local)
        yt = torch.FloatTensor(y_local).view(-1, 1)
        cut = int(len(Xt) * 0.8)
        loader = DataLoader(TensorDataset(Xt[:cut], yt[:cut]),
                            batch_size=256, shuffle=True)
        model.train()
        for _ in range(epochs):
            for bX, by in loader:
                optimizer.zero_grad()
                criterion(model(bX), by).backward()
                optimizer.step()
        cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
        model.eval()
        with torch.no_grad():
            local_acc = ((torch.sigmoid(model(Xt[cut:])) > 0.5).int() ==
                          yt[cut:].int()).float().mean().item()
        return {'client_id': client_id, 'state_dict': cpu_state,
                'local_accuracy': local_acc}
    except Exception as e:
        return {'client_id': client_id, 'state_dict': None, 'error': str(e)}

class FederatedLearning:
    def __init__(self, num_clients, input_size, hidden_size=32,
                 output_size=1, verbose=1, global_epochs=40,
                 max_workers=None, local_epochs=40, n_layers=3):
        self.num_clients   = num_clients
        self.global_epochs = global_epochs
        self.local_epochs  = local_epochs
        self.verbose       = verbose
        self.max_workers   = max_workers or cpu_count()
        self.hidden_size   = hidden_size
        self.input_size    = input_size
        self.n_layers      = n_layers
        self.global_model  = LSTMTeacher(input_size, hidden_size, n_layers)
        self.local_models  = [
            LSTMTeacher(input_size, hidden_size, n_layers)
            for _ in range(num_clients)
        ]
        self.client_data   = []

    def aggregate_weights(self):
        sds = [m.get_state_dict() for m in self.local_models]
        avg = {k: torch.stack([sd[k] for sd in sds]).mean(dim=0) for k in sds[0]}
        self.global_model.set_state_dict(avg)

    def train_clients_parallel(self):
        gs  = self.global_model.get_state_dict()
        tpw = max(1, self.max_workers // self.num_clients)
        args = [
            (gs, *self.client_data[i], i, self.hidden_size, self.input_size,
             1, self.local_epochs, tpw, self.n_layers)
            for i in range(self.num_clients)
        ]
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=self.num_clients) as pool:
            results = pool.starmap(train_client_process, args)
        for r in results:
            if r.get("state_dict"):
                self.local_models[r["client_id"]].set_state_dict(r["state_dict"])
                if self.verbose:
                    print(f"    Client {r['client_id']}: "
                          f"local_acc={r['local_accuracy']:.4f}")
            else:
                print(f"    Client {r['client_id']} FAILED: {r.get('error')}")

    def train(self, X, y, X_test, y_test):
        self.X_train          = X
        self.y_train          = y
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

def predict_ckks_child(child_model, threshold, ctx, X_eval):
    child_model.eval()
    W1, b1, W2, b2, W3, b3 = child_model.get_fused_weights()

    preds, lats = [], []

    for i, x in enumerate(X_eval):
        t0 = time.time()

        t_enc = time.time()
        enc   = ts.ckks_vector(ctx, x.astype(np.float64).tolist())
        t_enc = time.time() - t_enc

        t_eval = time.time()
        enc = enc.mm(W1.T) + b1.tolist()  
        enc = enc.square()                   
        enc = enc.mm(W2.T) + b2.tolist()  
        enc = enc.square()                   
        enc = enc.mm(W3.T) + b3.tolist()  
        t_eval = time.time() - t_eval

        t_dec = time.time()
        logit = np.array(enc.decrypt())[0]
        pred  = int(logit > threshold)
        t_dec = time.time() - t_dec

        preds.append(pred)
        lats.append({
            "encoding_s":  t_enc,
            "ckks_eval_s": t_eval,
            "decoding_s":  t_dec,
            "total_s":     time.time() - t0,
        })

        if (i + 1) % 50 == 0:
            avg = np.mean([s["total_s"] for s in lats])
            eta = avg * (len(X_eval) - i - 1)
            print(f"  [{i+1}/{len(X_eval)}] avg {avg:.2f}s/sample  "
                  f"ETA: {eta/60:.1f} min")

    return np.array(preds), lats


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    DATASETS = {
        "CVI": ("train_cvi_padded.csv",    "test_cvi_padded.csv"),
    }

    print("="*60)
    print("CKKS INFERENCE — FL-LSTM → Child MLP (TenSEAL/SEAL)")
    print("="*60)
    print("\nCKKS parameters:")
    print("  poly_modulus_degree : 16384")
    print("  coeff_mod_bit_sizes : [60, 26, 26, 26, 26, 26, 60]")
    print("  scale               : 2^26")
    print("  activation          : x^2 (quadratic, depth 1)")
    print("  security level      : 128-bit")

    ctx = make_ckks_context()

    for dataset_name, (train_path, test_path) in DATASETS.items():
        print("\n" + "="*60)
        print(f"DATASET: {dataset_name.upper()}")
        print("="*60)

        X_train, y_train = preprocess_data(train_path, k=4)
        X_test,  y_test  = preprocess_data(test_path,  k=4)
        fl = FederatedLearning(
            num_clients=5,
            input_size=X_train.shape[1],
            global_epochs=40,
            local_epochs=40,
        )
        fl.train(X_train, y_train, X_test, y_test)

        print("\n--- LSTM teacher plaintext evaluation ---")
        m_lstm = fl.global_model.evaluate(X_test, y_test.astype(int))
        print_metrics(m_lstm)

        child, threshold = distill(
            fl.global_model, fl.X_train, fl.y_train,
            input_size=X_train.shape[1],
            hidden_size=32, epochs=20,
            temperature=3.0, alpha=0.5
        )

        print("\n--- Child MLP plaintext evaluation ---")
        m_child_plain = child.evaluate(X_test, y_test.astype(int), threshold)
        print_metrics(m_child_plain)
        print(f"\n--- CKKS encrypted inference (500 samples) ---")
        preds, lats = predict_ckks_child(
            child, threshold, ctx, fl.X_test_shortened
        )
        y_true = fl.y_test_shortened

        m_enc = compute_metrics(y_true, preds)
        print_metrics(m_enc, f"CKKS Child MLP Encrypted — {dataset_name}")
        print_latency(lats)

        # Final summary
        print(f"\n{'='*60}")
        print(f"SUMMARY — {dataset_name}")
        print(f"{'='*60}")
        print(f"  {'Model':<35} {'Acc':>8} {'F1':>8}")
        print(f"  {'-'*53}")
        print(f"  {'LSTM teacher (plaintext)':<35} "
              f"{m_lstm['accuracy']:>8.4f} {m_lstm['f1']:>8.4f}")
        print(f"  {'Child MLP (plaintext)':<35} "
              f"{m_child_plain['accuracy']:>8.4f} {m_child_plain['f1']:>8.4f}")
        print(f"  {'Child MLP (CKKS encrypted)':<35} "
              f"{m_enc['accuracy']:>8.4f} {m_enc['f1']:>8.4f}")
        print(f"  Avg CKKS latency : "
              f"{np.mean([s['total_s'] for s in lats]):.2f}s/sample")
        print(f"  Throughput       : "
              f"{500/sum(s['total_s'] for s in lats):.4f} samples/sec")
        print(f"{'='*60}")
