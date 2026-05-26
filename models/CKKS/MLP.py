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
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import tenseal as ts
import warnings
warnings.filterwarnings("ignore")

def make_ckks_context():
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=16384,
        coeff_mod_bit_sizes=[60, 26, 26, 26, 26, 26, 60]
    )
    ctx.global_scale = 2 ** 26
    ctx.generate_galois_keys()
    return ctx

def get_kmer_vocabulary(k=4):
    bases = ['A', 'C', 'G', 'T']
    return [''.join(p) for p in product(bases, repeat=k)]

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
    if max_val > 0:
        X = X / max_val
    if verbose:
        print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y


class QuadActivation(nn.Module):
    def forward(self, x):
        return x * x

def build_ckks_mlp(input_size, hidden_size=32, output_size=1):
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        QuadActivation(),
        nn.Dropout(0.4),
        nn.Linear(hidden_size, hidden_size),
        nn.Dropout(0.4),
        QuadActivation(),
        nn.Linear(hidden_size, 1)
    )

class CKKSMLPModel(nn.Module):
    def __init__(self, input_size=256, hidden_size=32, output_size=1):
        super().__init__()
        self.model = build_ckks_mlp(input_size, hidden_size, output_size)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def evaluate(self, X_test, y_test):
        self.model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(
                self.model(torch.FloatTensor(X_test))
            ).numpy().ravel()
            y_pred = (probs > 0.5).astype(int)
        return {
            "accuracy":         np.mean(y_pred == y_test),
            "f1":               f1_score(y_test, y_pred, average="binary"),
            "precision":        precision_score(y_test, y_pred, zero_division=0),
            "recall":           recall_score(y_test, y_pred, zero_division=0),
            "roc_auc":          roc_auc_score(y_test, probs),
            "pr_auc":           average_precision_score(y_test, probs),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }

    def get_state_dict(self):
        return copy.deepcopy(self.model.state_dict())

    def set_state_dict(self, sd):
        self.model.load_state_dict(sd)

def train_client_process(model_state, X_local, y_local, client_id,
                         hidden_size, input_size, output_size,
                         epochs, threads_per_worker):
    try:
        torch.set_num_threads(threads_per_worker)
        model = build_ckks_mlp(input_size, hidden_size, output_size)
        if model_state:
            model.load_state_dict(model_state, strict=True)
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
            probs = torch.sigmoid(model(Xt[cut:]))
            local_acc = ((probs > 0.5).int() == yt[cut:].int()).float().mean().item()
        return {'client_id': client_id, 'state_dict': cpu_state,
                'local_accuracy': local_acc}
    except Exception as e:
        return {'client_id': client_id, 'state_dict': None, 'error': str(e)}

class FederatedLearning:
    def __init__(self, num_clients, input_size=256, hidden_size=32,
                 output_size=1, verbose=1, global_epochs=5,
                 max_workers=None, local_epochs=1):
        self.num_clients   = num_clients
        self.global_epochs = global_epochs
        self.local_epochs  = local_epochs
        self.verbose       = verbose
        self.max_workers   = max_workers or cpu_count()
        self.hidden_size   = hidden_size
        self.output_size   = output_size
        self.input_size    = input_size
        self.global_model  = CKKSMLPModel(input_size, hidden_size, output_size)
        self.local_models  = [
            CKKSMLPModel(input_size, hidden_size, output_size)
            for _ in range(num_clients)
        ]
        self.client_data   = []

    def aggregate_weights(self):
        sds = [m.get_state_dict() for m in self.local_models]
        avg = {}
        for k in sds[0]:
            avg[k] = torch.stack([sd[k] for sd in sds]).mean(dim=0)
        self.global_model.set_state_dict(avg)

    def train_clients_parallel(self):
        gs  = self.global_model.get_state_dict()
        tpw = max(1, self.max_workers // self.num_clients)
        args = [
            (gs, *self.client_data[i], i, self.hidden_size,
             self.input_size, self.output_size, self.local_epochs, tpw)
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

    def train(self, X, y, X_test=None, y_test=None):
        self.X_test_transformed = X_test
        self.y_test             = y_test.astype(int)
        self.X_test_shortened   = X_test[:500]
        self.y_test_shortened   = y_test[:500].astype(int)

        skf = StratifiedKFold(n_splits=self.num_clients,
                               shuffle=True, random_state=42)
        for _, idx in skf.split(X, y):
            self.client_data.append((X[idx], y[idx]))

        for epoch in range(self.global_epochs):
            print(f"\nGlobal Epoch {epoch+1}/{self.global_epochs}")
            self.train_clients_parallel()
            self.aggregate_weights()
            if self.X_test_transformed is not None:
                m = self.global_model.evaluate(
                    self.X_test_transformed, self.y_test
                )
                print(f"  Accuracy : {m['accuracy']:.4f}  "
                      f"F1: {m['f1']:.4f}  "
                      f"ROC-AUC: {m['roc_auc']:.4f}  "
                      f"PR-AUC: {m['pr_auc']:.4f}")
                print(f"  Confusion Matrix:\n{m['confusion_matrix']}")
        return self.global_model

    def predict_ckks(self, X, ctx, n_samples=500):
        self.global_model.model.eval()
        layers = [m for m in self.global_model.model.modules()
                  if isinstance(m, nn.Linear)]
        W1 = layers[0].weight.data.numpy().astype(np.float64)
        b1 = layers[0].bias.data.numpy().astype(np.float64)
        W2 = layers[1].weight.data.numpy().astype(np.float64)
        b2 = layers[1].bias.data.numpy().astype(np.float64)
        W3 = layers[2].weight.data.numpy().astype(np.float64)
        b3 = layers[2].bias.data.numpy().astype(np.float64)

        X_eval = X[:n_samples]
        preds, sample_latencies = [], []

        print(f"\nCKKS encrypted inference on {len(X_eval)} samples...")
        print("  Scheme    : CKKS (TenSEAL / Microsoft SEAL)")
        print("  Activation: x^2 (quadratic)")
        print(f"  poly_mod_degree: 16384 | scale: 2^26 | security: 128-bit")

        for i, x in enumerate(X_eval):
            t0 = time.time()

            # Encrypt
            t_enc_start = time.time()
            enc = ts.ckks_vector(ctx, x.astype(np.float64).tolist())
            t_enc = time.time() - t_enc_start

            # FHE evaluation
            t_eval_start = time.time()
            enc = enc.mm(W1.T) + b1.tolist()   
            enc = enc.square()                  
            enc = enc.mm(W2.T) + b2.tolist()   
            enc = enc.square()                   
            enc = enc.mm(W3.T) + b3.tolist()  
            t_eval = time.time() - t_eval_start

            t_dec_start = time.time()
            logit = np.array(enc.decrypt())[0]
            pred  = int(logit > 0)
            t_dec = time.time() - t_dec_start

            total = time.time() - t0
            preds.append(pred)
            sample_latencies.append({
                "encoding_s":    t_enc,
                "ckks_eval_s":   t_eval,
                "decoding_s":    t_dec,
                "total_s":       total,
            })

            if (i+1) % 50 == 0:
                print(f"  [{i+1}/{len(X_eval)}] "
                      f"avg latency: {np.mean([s['total_s'] for s in sample_latencies]):.2f}s")

        wall = sum(s["total_s"] for s in sample_latencies)
        enc_t  = [s["encoding_s"]  for s in sample_latencies]
        eval_t = [s["ckks_eval_s"] for s in sample_latencies]
        dec_t  = [s["decoding_s"]  for s in sample_latencies]
        tot_t  = [s["total_s"]     for s in sample_latencies]

        print(f"\nTotal Wall-Clock Runtime       : {wall:.2f}s")
        print(f"Throughput                     : {len(X_eval)/wall:.4f} samples/sec")
        print(f"\n--- Per-Sample Latency Breakdown (seconds) ---")
        print(f"{'Phase':<30} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
        print(f"{'-'*62}")
        for label, vals in [
            ("Encoding (pre-CKKS)",   enc_t),
            ("CKKS Evaluation",       eval_t),
            ("Decoding (post-CKKS)",  dec_t),
            ("Total per sample",      tot_t),
        ]:
            print(f"{label:<30} {np.mean(vals):>8.4f} {np.std(vals):>8.4f} "
                  f"{np.min(vals):>8.4f} {np.max(vals):>8.4f}")

        return np.array(preds), sample_latencies


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    DATASETS = {
        "Promoter": ("data_padded.csv",       "test_data_padded.csv"),
        "CVI": ("train_cvi_padded.csv",    "test_cvi_padded.csv"),
    }

    print("  poly_modulus_degree : 16384")
    print("  coeff_mod_bit_sizes : [60,26,26,26,26,26,60]")
    print("  scale               : 2^26")
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
            local_epochs=35,
        )

        fl.train(X_train, y_train, X_test, y_test)

        print("\n--- Plaintext evaluation (CKKS-trained model) ---")
        m_plain = fl.global_model.evaluate(X_test, y_test.astype(int))
        print(f"  Accuracy  : {m_plain['accuracy']:.4f}")
        print(f"  F1        : {m_plain['f1']:.4f}")
        print(f"  Precision : {m_plain['precision']:.4f}")
        print(f"  Recall    : {m_plain['recall']:.4f}")
        print(f"  ROC-AUC   : {m_plain['roc_auc']:.4f}")
        print(f"  PR-AUC    : {m_plain['pr_auc']:.4f}")
        print(f"  Confusion :\n{m_plain['confusion_matrix']}")

        preds, latencies = fl.predict_ckks(
            fl.X_test_shortened, ctx, n_samples=500
        )
        y_true = fl.y_test_shortened

        print("\n" + "="*50)
        print(f"FINAL CKKS INFERENCE RESULTS — {dataset_name}")
        print("="*50)
        print(f"  Samples evaluated : {len(y_true)}")
        print(f"  Accuracy          : {np.mean(preds==y_true)*100:.2f}%")
        print(f"  F1 Score          : {f1_score(y_true,preds,average='binary'):.4f}")
        print(f"  Precision         : {precision_score(y_true,preds,zero_division=0):.4f}")
        print(f"  Recall            : {recall_score(y_true,preds,zero_division=0):.4f}")
        print(f"  Confusion Matrix  :\n{confusion_matrix(y_true,preds)}")
        print("="*50)
