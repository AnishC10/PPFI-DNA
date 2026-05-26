import numpy as np
import time
import copy
from concurrent.futures import ThreadPoolExecutor
import torch.multiprocessing as mp
from multiprocessing import cpu_count
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from concrete.ml.torch.compile import compile_torch_model
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from itertools import product

# --- KMER HELPER ---

def get_kmer_vocabulary(k=4):
    bases = ['A','C','G','T']
    return [''.join(p) for p in product(bases, repeat=k)]

# --- SVM MODEL ARCHITECTURE ---

def build_model_architecture(input_size):
    return nn.Linear(input_size, 1)  # Linear SVM

# --- HINGE LOSS ---

def hinge_loss(outputs, targets):
    return torch.mean(torch.clamp(1 - outputs * targets, min=0))

# --- CLIENT TRAINING PROCESS ---

def train_client_process(model_state, X_local, y_local, client_id,
                         input_size, epochs, threads_per_worker):

    try:

        torch.set_num_threads(threads_per_worker)

        model = build_model_architecture(input_size)

        if model_state:
            model.load_state_dict(model_state)

        optimizer = optim.SGD(model.parameters(), lr=0.005)

        # convert labels {0,1} -> {-1,1}
        y_local = np.where(y_local == 0, -1, 1)

        X_tensor = torch.FloatTensor(X_local)
        y_tensor = torch.FloatTensor(y_local).view(-1,1)

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

                outputs = model(batch_X)

                loss = hinge_loss(outputs, batch_y)

                loss.backward()

                optimizer.step()

        cpu_state = {k:v.cpu() for k,v in model.state_dict().items()}

        model.eval()

        with torch.no_grad():

            logits = model(X_tensor[train_percent:])

            preds = (logits > 0).int()

            y_eval = (y_tensor[train_percent:] > 0).int()

            local_acc = (preds == y_eval).float().mean().item()

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

# --- MODEL CLASS ---

class PyTorchModel(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.model = build_model_architecture(input_size)

    def predict(self, X):

        self.model.eval()

        with torch.no_grad():

            logits = self.model(torch.FloatTensor(X))

            return (logits > 0).int().numpy().ravel()

    def evaluate(self, X_test, y_test):

        self.model.eval()

        with torch.no_grad():

            logits = self.model(torch.FloatTensor(X_test)).numpy().ravel()

            y_pred = (logits > 0).astype(int)

        return {
            "accuracy": np.mean(y_pred == y_test),
            "f1": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, logits),
            "pr_auc": average_precision_score(y_test, logits),
            "confusion_matrix": confusion_matrix(y_test, y_pred)
        }

    def get_state_dict(self):

        return copy.deepcopy(self.model.state_dict())

    def set_state_dict(self, state_dict):

        self.model.load_state_dict(state_dict)

# --- FEDERATED LEARNING CLASS ---

class FederatedLearning:

    def __init__(self, num_clients, input_size,
                 global_epochs=5,
                 local_epochs=5,
                 max_workers=None):

        self.num_clients = num_clients
        self.global_epochs = global_epochs
        self.local_epochs = local_epochs

        self.max_workers = max_workers or cpu_count()

        self.input_size = input_size

        self.global_model = PyTorchModel(input_size)

        self.local_models = [
            PyTorchModel(input_size)
            for _ in range(num_clients)
        ]

        self.client_data = []

    def aggregate_weights(self):

        state_dicts = [m.get_state_dict() for m in self.local_models]

        avg_state = {}

        for k in state_dicts[0]:

            avg_state[k] = torch.stack(
                [sd[k] for sd in state_dicts]
            ).mean(dim=0)

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
                self.input_size,
                self.local_epochs,
                threads_per_worker
            ))

        ctx = mp.get_context("spawn")

        with ctx.Pool(processes=self.num_clients) as pool:

            results = pool.starmap(train_client_process, args)

        for r in results:

            if r.get("state_dict") is not None:

                self.local_models[r["client_id"]].set_state_dict(r["state_dict"])

                print(f"Client {r['client_id']} local_acc={r['local_accuracy']:.4f}")

    def train(self, X, y, X_test, y_test):

        skf = StratifiedKFold(n_splits=self.num_clients, shuffle=True, random_state=42)

        for _, client_idx in skf.split(X,y):

            self.client_data.append((X[client_idx],y[client_idx]))

        for epoch in range(self.global_epochs):

            print(f"\nGlobal Epoch {epoch+1}/{self.global_epochs}")

            self.train_clients_parallel()

            self.aggregate_weights()

            metrics = self.global_model.evaluate(X_test,y_test)

            print(metrics)

        return self.global_model

    def compile_to_fhe(self, X_calibration):

        print("\nCompiling to TFHE")

        self.global_model.model.eval()

        self.quantized_module = compile_torch_model(
            self.global_model.model,
            torch.FloatTensor(X_calibration),
            n_bits=6,
            p_error=0.01
        )

        print("Compilation finished")
    def predict_encrypted(self, X):
        print(f"\nMaking encrypted predictions on {len(X)} samples using {self.max_workers} threads...")
    
        def predict_single(x):
            y = self.quantized_module.forward(
                x.reshape(1, -1), fhe="execute"
            )
            return int(y > 0)
    
        start = time.time()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            preds = list(
                tqdm(
                    executor.map(predict_single, X),
                    total=len(X),
                    desc="Encrypted inference",
                    unit="sample"
                )
            )
    
        print(f"Total Runtime: {time.time() - start}")
        return np.array(preds)
# --- DATA PREPROCESSING ---

from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_data(csv_path, k=5):

    data = pd.read_csv(csv_path)

    sequences = data["sequence"].str.upper().tolist()
    y = data["label"].values.astype(np.float32)

    vocab = get_kmer_vocabulary(k)

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(k, k),
        vocabulary=vocab,
        lowercase=False,
        norm=None
    )

    X = vectorizer.fit_transform(sequences).toarray().astype(np.float32)

    return X, y
# --- MAIN ---

if __name__ == "__main__":

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    X_train,y_train = preprocess_data("train_cvi_padded.csv",k=4)

    X_test,y_test = preprocess_data("test_cvi_padded.csv",k=4)

    fed_learning = FederatedLearning(
        num_clients=5,
        input_size=X_train.shape[1],
        global_epochs=20,
        local_epochs=20
    )

    fed_learning.train(X_train,y_train,X_test,y_test)

    fed_learning.compile_to_fhe(X_test[:70])
    index = 500
    preds = fed_learning.predict_encrypted(X_test[:index])
    print("\nFinal Results:")
    print(f"Accuracy: {np.mean(preds == y_test[:index]) * 100:.2f}%")
    print(f"F1 Score: {f1_score(y_test[:index], preds, average='binary'):.4f}")
    print(f"Precision: {precision_score(y_test[:index], preds, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_test[:index], preds, zero_division=0):.4f}")
