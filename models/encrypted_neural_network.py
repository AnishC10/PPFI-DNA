import numpy as np
import time
import copy
from concurrent.futures import ThreadPoolExecutor
import torch.multiprocessing as mp
from multiprocessing import cpu_count
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from concrete.ml.torch.compile import compile_torch_model
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from itertools import product

# --- K-MER HELPER ---
def get_kmer_vocabulary(k=4):
    bases = ['A', 'C', 'G', 'T']
    return [''.join(p) for p in product(bases, repeat=k)]

# --- HELPER FUNCTIONS ---

def build_model_architecture(input_size, hidden_size, output_size):
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(hidden_size, hidden_size),
        nn.Dropout(0.4),
        nn.ReLU(),
        nn.Linear(hidden_size, 1)
    )

def train_client_process(model_state, X_local, y_local, client_id,
                         hidden_size, input_size, output_size,
                         epochs, threads_per_worker):
    try:
        torch.set_num_threads(threads_per_worker)

        model = build_model_architecture(input_size, hidden_size, output_size)
        if model_state:
            model.load_state_dict(model_state, strict=True)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        X_tensor = torch.FloatTensor(X_local)
        y_tensor = torch.FloatTensor(y_local).view(-1, 1)
        train_percent = int(len(X_tensor)*0.8)
        loader = DataLoader(
            TensorDataset(X_tensor[:train_percent], y_tensor[:train_percent]),
            batch_size=256,
            shuffle=True
        )

        model.train()
        for _ in range(epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

        cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}

        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(X_tensor[train_percent:]))
            preds = (probs > 0.5).int()
            local_acc = (preds == y_tensor[train_percent:].int()).float().mean().item()

        return {
            'client_id': client_id,
            'state_dict': cpu_state,
            'local_accuracy': local_acc
        }

    except Exception as e:
        return {
            'client_id': client_id,
            'state_dict': None,
            'error': str(e)
        }

# --- CLASSES ---

class PyTorchModel(nn.Module):
    def __init__(self, input_size=2292, hidden_size=32, output_size=1):
        super(PyTorchModel, self).__init__()
        self.model = build_model_architecture(input_size, hidden_size, output_size)
        self.loss = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

    def fit(self, X, y, epochs):
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).view(-1, 1)

        loader = DataLoader(
            TensorDataset(X_tensor, y_tensor),
            batch_size=128,
            shuffle=True
        )

        self.model.train()
        for _ in range(epochs):
            for batch_X, batch_y in loader:
                self.optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = self.loss(logits, batch_y)
                loss.backward()
                self.optimizer.step()
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(self.model(torch.FloatTensor(X)))
            return (probs > 0.5).int().numpy().ravel()

    def evaluate(self, X_test, y_test):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.FloatTensor(X_test))
            probs = torch.sigmoid(logits).numpy().ravel()
            y_pred = (probs > 0.5).astype(int)

        return {
            "accuracy": np.mean(y_pred == y_test),
            "f1": f1_score(y_test, y_pred, average="binary"),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probs),
            "pr_auc": average_precision_score(y_test, probs),
            "confusion_matrix": confusion_matrix(y_test, y_pred)
        }

    def get_state_dict(self):
        return copy.deepcopy(self.model.state_dict())

    def set_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

class FederatedLearning:
    def __init__(self, num_clients, input_size=2292, hidden_size=32,
                 output_size=1, verbose=1, global_epochs=5,
                 max_workers=None, local_epochs=1):

        self.num_clients = num_clients
        self.global_epochs = global_epochs
        self.local_epochs = local_epochs
        self.verbose = verbose

        self.max_workers = max_workers or cpu_count()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = input_size

        self.global_model = PyTorchModel(self.input_size, hidden_size, output_size)
        self.local_models = [
            PyTorchModel(self.input_size, hidden_size, output_size)
            for _ in range(num_clients)
        ]
        self.client_data = []

    def aggregate_weights(self):
        state_dicts = [m.get_state_dict() for m in self.local_models]
        avg_state = {}
        for k in state_dicts[0]:
            avg_state[k] = torch.stack([sd[k] for sd in state_dicts]).mean(dim=0)
        self.global_model.set_state_dict(avg_state)

    def train_clients_parallel(self):
        global_state = self.global_model.get_state_dict()
        threads_per_worker = max(1, self.max_workers // self.num_clients)

        args = []
        for i in range(self.num_clients):
            X_local, y_local = self.client_data[i]
            args.append((
                global_state,
                X_local,
                y_local,
                i,
                self.hidden_size,
                self.input_size,
                self.output_size,
                self.local_epochs,
                threads_per_worker
            ))

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=self.num_clients) as pool:
            results = pool.starmap(train_client_process, args)

        for r in results:
            if r.get("state_dict") is not None:
                self.local_models[r["client_id"]].set_state_dict(r["state_dict"])
                if self.verbose:
                    print(f"    Client {r['client_id']}: local_acc={r['local_accuracy']:.4f}")

    def train(self, X, y, X_test=None, y_test=None):
        self.X_test_transformed = X_test
        self.y_test             = y_test.astype(int)
        self.X_test_shortened   = X_test[:500]
        self.y_test_shortened   = y_test[:500].astype(int)

        skf = StratifiedKFold(n_splits=self.num_clients, shuffle=True, random_state=42)
    
        for i, (_, client_idx) in enumerate(skf.split(X, y)):
            client_X = X[client_idx]
            client_y = y[client_idx]
            self.client_data.append((client_X, client_y))

        for epoch in range(self.global_epochs):
            print(f"\nGlobal Epoch {epoch+1}/{self.global_epochs}")
            self.train_clients_parallel()
            self.aggregate_weights()

            if self.X_test_transformed is not None:
                m = self.global_model.evaluate(
                    self.X_test_transformed, self.y_test
                )
                print(f"  Accuracy   : {m['accuracy']:.4f}")
                print(f"  F1 Score   : {m['f1']:.4f}")
                print(f"  Precision  : {m['precision']:.4f}")
                print(f"  Recall     : {m['recall']:.4f}")
                print(f"  ROC-AUC    : {m['roc_auc']:.4f}")
                print(f"  PR-AUC     : {m['pr_auc']:.4f}")
                print(f"  Confusion Matrix:\n{m['confusion_matrix']}")

        return self.global_model

    def compile_to_fhe(self, X_calibration):
        print("\n--- Compiling to TFHE ---")
        self.global_model.model.eval()

        compile_start = time.time()
        self.quantized_module = compile_torch_model(
            self.global_model.model,
            torch.FloatTensor(X_calibration),
            n_bits=6,
            p_error=0.01
        )
        compile_elapsed = time.time() - compile_start
        print(f"Successfully compiled to TFHE! (compile time: {compile_elapsed:.2f}s)")

        param_bytes = sum(
            p.nelement() * p.element_size()
            for p in self.global_model.model.parameters()
        )
        print(f"Plaintext Model Size           : {param_bytes / 1024:.1f} KB")
        print(f"Quantization Bits              : 6")
        print(f"p_error                        : 0.01")

        return self.quantized_module

    def predict_encrypted(self, X):
        print(f"\nMaking encrypted predictions on {len(X)} samples using {self.max_workers} threads...")

        sample_latencies = []

        def predict_single(x):
            t0 = time.time()
            x_reshaped = x.reshape(1, -1)
            t1 = time.time()
            result = self.quantized_module.forward(x_reshaped, fhe="execute")
            t2 = time.time()
            pred = int(result > 0)
            t3 = time.time()
            sample_latencies.append({
                "encoding_s":    t1 - t0,
                "fhe_execute_s": t2 - t1,
                "decoding_s":    t3 - t2,
                "total_s":       t3 - t0,
            })
            return pred

        wall_start = time.time()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            preds = list(
                tqdm(
                    executor.map(predict_single, X),
                    total=len(X),
                    desc="Encrypted inference",
                    unit="sample"
                )
            )
        wall_elapsed = time.time() - wall_start

        enc_times = [s["encoding_s"]    for s in sample_latencies]
        fhe_times = [s["fhe_execute_s"] for s in sample_latencies]
        dec_times = [s["decoding_s"]    for s in sample_latencies]
        tot_times = [s["total_s"]       for s in sample_latencies]

        print(f"\nTotal Wall-Clock Runtime       : {wall_elapsed:.2f}s")
        print(f"Throughput                     : {len(X)/wall_elapsed:.2f} samples/sec")
        print(f"\n--- Per-Sample Latency Breakdown (seconds) ---")
        print(f"{'Phase':<30} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
        print(f"{'-'*62}")
        for label, vals in [
            ("Encoding (pre-FHE)",   enc_times),
            ("FHE Evaluation",       fhe_times),
            ("Decoding (post-FHE)",  dec_times),
            ("Total per sample",     tot_times),
        ]:
            print(f"{label:<30} {np.mean(vals):>8.4f} {np.std(vals):>8.4f} "
                  f"{np.min(vals):>8.4f} {np.max(vals):>8.4f}")

        self._last_latencies = sample_latencies
        return np.array(preds)


# --- DATA PREPROCESSING (K-MER VERSION) ---

def preprocess_data(csv_path, k=4, verbose=True):
    data = pd.read_csv(csv_path)
    data.columns = data.columns.str.strip().str.replace('"', '').str.replace("'", '')

    sequences = data["sequence"].str.upper().tolist()
    y = data["label"].values.astype(np.float32)

    vocab = get_kmer_vocabulary(k)
    vectorizer = CountVectorizer(
        analyzer='char',
        ngram_range=(k, k),
        vocabulary=vocab,
        lowercase=False
    )

    X = vectorizer.transform(sequences).toarray().astype(np.float32)

    max_val = X.max()
    if max_val > 0:
        X = X / max_val

    if verbose:
        print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")

    return X, y

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    X_train, y_train = preprocess_data("train_cvi_padded.csv", k=4)
    X_test,  y_test  = preprocess_data("test_cvi_padded.csv",  k=4)

    fed_learning = FederatedLearning(
        num_clients=5,
        input_size=X_train.shape[1],
        global_epochs=25,
        local_epochs=20
    )

    fed_learning.train(X_train, y_train, X_test, y_test)
    fed_learning.compile_to_fhe(fed_learning.X_test_transformed[:70])

    preds  = fed_learning.predict_encrypted(fed_learning.X_test_shortened)
    y_true = fed_learning.y_test_shortened  # already int from train()

    print("\n" + "="*50)
    print("FINAL ENCRYPTED INFERENCE RESULTS")
    print("="*50)
    print(f"  Samples evaluated : {len(y_true)}")
    print(f"  Accuracy          : {np.mean(preds == y_true) * 100:.2f}%")
    print(f"  F1 Score          : {f1_score(y_true, preds, average='binary'):.4f}")
    print(f"  Precision         : {precision_score(y_true, preds, zero_division=0):.4f}")
    print(f"  Recall            : {recall_score(y_true, preds, zero_division=0):.4f}")
    print(f"  Confusion Matrix  :\n{confusion_matrix(y_true, preds)}")
    print("="*50)
