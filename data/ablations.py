"""
Use for gathering data, such as varying IID, bit-widths, etc
"""
import numpy as np
import time
import copy
import warnings
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
warnings.filterwarnings("ignore")
from concrete.ml.torch.compile import compile_torch_model

DATASETS = {
    "Promoter": {
        "train": "data_padded.csv",
        "test":  "test_data_padded.csv",
    },
    "Enhancer": {
        "train": "train_cvi_padded.csv",
        "test":  "test_cvi_padded.csv",
    },
}

SEEDS         = [42, 7, 123]
GLOBAL_EPOCHS = 15
LOCAL_EPOCHS  = 10
HIDDEN_SIZE   = 32
BIT_WIDTHS    = [4, 6, 8]

def get_kmer_vocabulary(k=4):
    bases = ['A', 'C', 'G', 'T']
    return [''.join(p) for p in product(bases, repeat=k)]

def preprocess_data(csv_path, k=4, verbose=False):
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
        print(f"  Loaded {X.shape[0]} samples, {X.shape[1]} features")
    return X, y, data

def _metrics(y_test, y_pred, probs=None):
    m = {
        "accuracy":         np.mean(y_pred == y_test),
        "f1":               f1_score(y_test, y_pred, average="binary"),
        "precision":        precision_score(y_test, y_pred, zero_division=0),
        "recall":           recall_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }
    if probs is not None:
        try:
            m["roc_auc"] = roc_auc_score(y_test, probs)
            m["pr_auc"]  = average_precision_score(y_test, probs)
        except Exception:
            m["roc_auc"] = 0.0
            m["pr_auc"]  = 0.0
    return m

def print_summary(label, s):
    print(f"\n{'='*60}\nCONFIG: {label}\n{'='*60}")
    for m in ["accuracy","f1","precision","recall","roc_auc","pr_auc"]:
        if f"{m}_mean" in s:
            print(f"  {m:<12}: {s[m+'_mean']:.4f} ± {s[m+'_std']:.4f}")
    print(f"  Confusion Matrix (mean):\n{s['confusion_matrix_mean'].round(1)}")

def fedavg(local_models):
    sds = [m.get_state_dict() for m in local_models]
    avg = {}
    for k in sds[0]:
        avg[k] = torch.stack([sd[k] for sd in sds]).mean(dim=0)
    return avg

def _split(X, y, num_clients, distribution, dirichlet_alpha, seed):
    if distribution == "iid":
        skf = StratifiedKFold(n_splits=num_clients, shuffle=True, random_state=seed)
        return [(X[idx], y[idx]) for _, idx in skf.split(X, y)]
    rng = np.random.default_rng(seed)
    client_indices = [[] for _ in range(num_clients)]
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]; rng.shuffle(cls_idx)
        props = rng.dirichlet([dirichlet_alpha]*num_clients)
        props = (props*len(cls_idx)).astype(int)
        props[-1] = len(cls_idx) - props[:-1].sum()
        start = 0
        for c,n in enumerate(props):
            client_indices[c].extend(cls_idx[start:start+n].tolist()); start+=n
    return [(X[np.array(idx, dtype=np.int64)], y[np.array(idx, dtype=np.int64)]) for idx in client_indices]

def multi_seed(fn, seeds, **kwargs):
    per_seed = [fn(seed=s, **kwargs) for s in seeds]
    keys = ["accuracy","f1","precision","recall","roc_auc","pr_auc"]
    s = {f"{k}_mean": np.mean([p[k] for p in per_seed if k in p]) for k in keys}
    s.update({f"{k}_std": np.std([p[k] for p in per_seed if k in p]) for k in keys})
    s["confusion_matrix_mean"] = np.array([p["confusion_matrix"] for p in per_seed]).mean(axis=0)
    sim_accs = [p["sim_accuracy"] for p in per_seed if p.get("sim_accuracy") is not None]
    sim_f1s  = [p["sim_f1"]       for p in per_seed if p.get("sim_f1")       is not None]
    s["sim_accuracy_mean"] = float(np.mean(sim_accs)) if sim_accs else None
    s["sim_accuracy_std"]  = float(np.std(sim_accs))  if sim_accs else None
    s["sim_f1_mean"]       = float(np.mean(sim_f1s))  if sim_f1s  else None
    return s

def build_mlp(input_size, hidden_size=32, output_size=1):
    return nn.Sequential(
        nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Dropout(0.4),
        nn.Linear(hidden_size, hidden_size), nn.Dropout(0.4), nn.ReLU(),
        nn.Linear(hidden_size, 1)
    )

class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size=32, output_size=1):
        super().__init__()
        self.model = build_mlp(input_size, hidden_size, output_size)
    def evaluate(self, X_test, y_test):
        self.model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(self.model(torch.FloatTensor(X_test))).numpy().ravel()
        return _metrics(y_test, (probs>0.5).astype(int), probs)
    def get_state_dict(self): return copy.deepcopy(self.model.state_dict())
    def set_state_dict(self, sd): self.model.load_state_dict(sd)

def mlp_client_process(model_state, X_local, y_local, client_id,
                        hidden_size, input_size, output_size, epochs, tpw):
    try:
        torch.set_num_threads(tpw)
        model = build_mlp(input_size, hidden_size, output_size)
        if model_state: model.load_state_dict(model_state, strict=True)
        crit = nn.BCEWithLogitsLoss()
        opt  = optim.Adam(model.parameters(), lr=0.001)
        Xt = torch.FloatTensor(X_local); yt = torch.FloatTensor(y_local).view(-1,1)
        cut = int(len(Xt)*0.8)
        loader = DataLoader(TensorDataset(Xt[:cut],yt[:cut]), batch_size=256, shuffle=True)
        model.train()
        for _ in range(epochs):
            for bX,by in loader:
                opt.zero_grad(); crit(model(bX),by).backward(); opt.step()
        cpu_state = {k:v.cpu() for k,v in model.state_dict().items()}
        model.eval()
        with torch.no_grad():
            local_acc = ((torch.sigmoid(model(Xt[cut:]))>0.5).int()==yt[cut:].int()).float().mean().item()
        return {'client_id':client_id,'state_dict':cpu_state,'local_accuracy':local_acc}
    except Exception as e:
        return {'client_id':client_id,'state_dict':None,'error':str(e)}

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=32, output_size=1, num_layers=3):
        super().__init__()
        self.hidden_size=hidden_size; self.num_layers=num_layers
        self.lstm=nn.LSTM(input_size,hidden_size,num_layers,batch_first=True)
        self.fc=nn.Linear(hidden_size,output_size)
    def forward(self,X):
        if X.dim()==2: X=X.unsqueeze(1)
        h0=torch.zeros(self.num_layers,X.size(0),self.hidden_size)
        c0=torch.zeros(self.num_layers,X.size(0),self.hidden_size)
        out,_=self.lstm(X,(h0,c0)); return self.fc(out[:,-1,:])
    def evaluate(self,X_test,y_test):
        self.eval()
        with torch.no_grad():
            probs=torch.sigmoid(self(torch.FloatTensor(X_test))).numpy().ravel()
        return _metrics(y_test,(probs>0.5).astype(int),probs)
    def get_state_dict(self): return copy.deepcopy(self.state_dict())
    def set_state_dict(self,sd): self.load_state_dict(sd)

class ChildMLP(nn.Module):
    def __init__(self,input_size,hidden_size=32,output_size=1):
        super().__init__()
        self.fc1=nn.Linear(input_size,hidden_size); self.fc2=nn.Linear(hidden_size,hidden_size)
        self.fc3=nn.Linear(hidden_size,output_size); self.relu=nn.ReLU()
    def forward(self,X): return self.fc3(self.relu(self.fc2(self.relu(self.fc1(X)))))
    def evaluate(self,X_test,y_test,threshold=0.0):
        self.eval()
        with torch.no_grad():
            logits=self(torch.FloatTensor(X_test)).numpy().ravel()
            probs=1/(1+np.exp(-logits))
        return _metrics(y_test,(logits>threshold).astype(int),probs)
    def get_state_dict(self): return copy.deepcopy(self.state_dict())
    def set_state_dict(self,sd): self.load_state_dict(sd)

class DistillationLoss(nn.Module):
    def __init__(self,temperature=3.0,alpha=0.5):
        super().__init__(); self.temperature=temperature; self.alpha=alpha
    def forward(self,child_logits,teacher_preds,target):
        hard=F.binary_cross_entropy_with_logits(child_logits,target)
        soft=F.binary_cross_entropy(torch.sigmoid(child_logits/self.temperature),teacher_preds)
        return self.alpha*soft+(1-self.alpha)*hard

def lstm_client_process(model_state, X_local, y_local, client_id,
                         hidden_size, input_size, output_size, epochs, tpw, n_layers=3):
    try:
        torch.set_num_threads(tpw)
        model=LSTMModel(input_size,hidden_size,output_size,n_layers)
        if model_state: model.load_state_dict(model_state,strict=True)
        crit=nn.BCEWithLogitsLoss(); opt=optim.Adam(model.parameters(),lr=0.001)
        Xt=torch.FloatTensor(X_local); yt=torch.FloatTensor(y_local).view(-1,1)
        cut=int(len(Xt)*0.8)
        loader=DataLoader(TensorDataset(Xt[:cut],yt[:cut]),batch_size=256,shuffle=True)
        model.train()
        for _ in range(epochs):
            for bX,by in loader: opt.zero_grad(); crit(model(bX),by).backward(); opt.step()
        cpu_state={k:v.cpu() for k,v in model.state_dict().items()}
        model.eval()
        with torch.no_grad():
            local_acc=((torch.sigmoid(model(Xt[cut:]))>0.5).int()==yt[cut:].int()).float().mean().item()
        return {'client_id':client_id,'state_dict':cpu_state,'local_accuracy':local_acc}
    except Exception as e:
        return {'client_id':client_id,'state_dict':None,'error':str(e)}

def build_svm(input_size): return nn.Linear(input_size,1)
def hinge_loss(outputs,targets): return torch.mean(torch.clamp(1-outputs*targets,min=0))

class SVMModel(nn.Module):
    def __init__(self,input_size):
        super().__init__(); self.model=build_svm(input_size)
    def evaluate(self,X_test,y_test):
        self.model.eval()
        with torch.no_grad():
            logits=self.model(torch.FloatTensor(X_test)).numpy().ravel()
        return _metrics(y_test,(logits>0).astype(int),logits)
    def get_state_dict(self): return copy.deepcopy(self.model.state_dict())
    def set_state_dict(self,sd): self.model.load_state_dict(sd)

def svm_client_process(model_state,X_local,y_local,client_id,input_size,epochs,tpw):
    try:
        torch.set_num_threads(tpw)
        model=build_svm(input_size)
        if model_state: model.load_state_dict(model_state)
        opt=optim.SGD(model.parameters(),lr=0.005)
        y_svm=np.where(y_local==0,-1,1)
        Xt=torch.FloatTensor(X_local); yt=torch.FloatTensor(y_svm).view(-1,1)
        cut=int(len(Xt)*0.8)
        loader=DataLoader(TensorDataset(Xt[:cut],yt[:cut]),batch_size=256,shuffle=True)
        model.train()
        for _ in range(epochs):
            for bX,by in loader: opt.zero_grad(); hinge_loss(model(bX),by).backward(); opt.step()
        cpu_state={k:v.cpu() for k,v in model.state_dict().items()}
        model.eval()
        with torch.no_grad():
            local_acc=((model(Xt[cut:])>0).int()==(yt[cut:]>0).int()).float().mean().item()
        return {'client_id':client_id,'state_dict':cpu_state,'local_accuracy':local_acc}
    except Exception as e:
        return {'client_id':client_id,'state_dict':None,'error':str(e)}

def run_fl_mlp(X_train, y_train, X_test, y_test, num_clients,
               distribution, dirichlet_alpha, seed,
               global_epochs, local_epochs, input_size):
    torch.manual_seed(seed); np.random.seed(seed)
    global_model = MLPModel(input_size, HIDDEN_SIZE)
    local_models = [MLPModel(input_size, HIDDEN_SIZE) for _ in range(num_clients)]
    client_data  = _split(X_train, y_train, num_clients, distribution, dirichlet_alpha, seed)
    tpw = max(1, cpu_count()//num_clients)
    epoch_accs, epoch_times = [], []
    for ep in range(global_epochs):
        t0   = time.time()
        gs   = global_model.get_state_dict()
        args = [(gs,*client_data[i],i,HIDDEN_SIZE,input_size,1,local_epochs,tpw)
                for i in range(num_clients)]
        ctx  = mp.get_context("spawn")
        with ctx.Pool(processes=num_clients) as pool:
            results = pool.starmap(mlp_client_process, args)
        for r in results:
            if r.get("state_dict"): local_models[r["client_id"]].set_state_dict(r["state_dict"])
        global_model.set_state_dict(fedavg(local_models))
        m = global_model.evaluate(X_test, y_test.astype(int))
        epoch_accs.append(m["accuracy"]); epoch_times.append(time.time()-t0)
    final = global_model.evaluate(X_test, y_test.astype(int))
    final["epoch_accs"]  = epoch_accs
    final["epoch_times"] = epoch_times
    sim_acc, sim_f1 = simulate_accuracy(global_model, X_test, y_test, n_bits=6)
    final["sim_accuracy"] = sim_acc
    final["sim_f1"]       = sim_f1
    return final

def run_fl_lstm(X_train, y_train, X_test, y_test,
                num_clients, seed, global_epochs, local_epochs, input_size):
    torch.manual_seed(seed); np.random.seed(seed)
    global_model = LSTMModel(input_size, HIDDEN_SIZE, 1, 3)
    local_models = [LSTMModel(input_size, HIDDEN_SIZE, 1, 3) for _ in range(num_clients)]
    client_data  = _split(X_train, y_train, num_clients, "iid", None, seed)
    tpw = max(1, cpu_count()//num_clients)
    for ep in range(global_epochs):
        gs   = global_model.get_state_dict()
        args = [(gs,*client_data[i],i,HIDDEN_SIZE,input_size,1,local_epochs,tpw,3)
                for i in range(num_clients)]
        ctx  = mp.get_context("spawn")
        with ctx.Pool(processes=num_clients) as pool:
            results = pool.starmap(lstm_client_process, args)
        for r in results:
            if r.get("state_dict"): local_models[r["client_id"]].set_state_dict(r["state_dict"])
        global_model.set_state_dict(fedavg(local_models))
    global_model.eval()
    Xt = torch.FloatTensor(X_train); yt = torch.FloatTensor(y_train).view(-1,1)
    with torch.no_grad():
        teacher_preds = torch.sigmoid(global_model(Xt)/3.0).detach()
    child = ChildMLP(input_size, HIDDEN_SIZE, 1)
    loader = DataLoader(TensorDataset(Xt, teacher_preds, yt), batch_size=256, shuffle=True)
    opt = optim.Adam(child.parameters()); crit = DistillationLoss(3.0, 0.5)
    child.train()
    for _ in range(20):
        for bX,bsoft,bhard in loader:
            opt.zero_grad(); crit(child(bX),bsoft,bhard).backward(); opt.step()
    child.eval()
    with torch.no_grad():
        threshold = float(np.median(child(Xt).numpy().ravel()))
    return child.evaluate(X_test, y_test.astype(int), threshold=threshold)

def run_fl_svm(X_train, y_train, X_test, y_test,
               num_clients, seed, global_epochs, local_epochs, input_size):
    torch.manual_seed(seed); np.random.seed(seed)
    global_model = SVMModel(input_size)
    local_models = [SVMModel(input_size) for _ in range(num_clients)]
    client_data  = _split(X_train, y_train, num_clients, "iid", None, seed)
    tpw = max(1, cpu_count()//num_clients)
    for ep in range(global_epochs):
        gs   = global_model.get_state_dict()
        args = [(gs,*client_data[i],i,input_size,local_epochs,tpw)
                for i in range(num_clients)]
        ctx  = mp.get_context("spawn")
        with ctx.Pool(processes=num_clients) as pool:
            results = pool.starmap(svm_client_process, args)
        for r in results:
            if r.get("state_dict"): local_models[r["client_id"]].set_state_dict(r["state_dict"])
        global_model.set_state_dict(fedavg(local_models))
    return global_model.evaluate(X_test, y_test.astype(int))

def quantize_model_weights(state_dict, n_bits):
    q_sd = {}
    for key, tensor in state_dict.items():
        v_min, v_max = tensor.min(), tensor.max()
        if v_max == v_min:
            q_sd[key] = tensor.clone()
        else:
            scale = (v_max-v_min)/(2**n_bits-1)
            q_sd[key] = torch.round((tensor-v_min)/scale)*scale+v_min
    return q_sd

def simulate_accuracy(model, X_test, y_test, n_bits=6, n_samples=500):
    """Compile MLP and run fhe=simulate. Fast, no key gen."""
    try:
        X_cal = torch.FloatTensor(X_test[:70])
        model.model.eval()
        qm = compile_torch_model(model.model, X_cal, n_bits=n_bits, p_error=0.01)
        sim_logits = []
        for x in X_test[:n_samples]:
            out = qm.forward(x.reshape(1,-1), fhe="simulate")
            sim_logits.append(float(out.flatten()[0]))
        sim_logits = np.array(sim_logits)
        threshold  = float(np.median(sim_logits))
        preds      = (sim_logits > threshold).astype(int)
        y_true     = y_test[:n_samples].astype(int)
        return float(np.mean(preds == y_true)), float(f1_score(y_true, preds, zero_division=0))
    except Exception as e:
        print(f"    [simulate error: {e}]")
        return None, None

if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_dataset_results = {}

    for dataset_name, paths in DATASETS.items():
        print(f"\n\n{'#'*70}")
        print(f"# DATASET: {dataset_name.upper()}")
        print(f"{'#'*70}")

        X_train, y_train, raw_train = preprocess_data(paths["train"], k=4, verbose=True)
        X_test,  y_test,  raw_test  = preprocess_data(paths["test"],  k=4, verbose=True)
        y_train_int = y_train.astype(int)
        y_test_int  = y_test.astype(int)
        INPUT_SIZE  = X_train.shape[1]
        dataset_results = {}

        print(f"\n{'='*60}\nSEC A: DATASET STATISTICS — {dataset_name}\n{'='*60}")
        for split_name, raw_df in [("Train", raw_train), ("Test", raw_test)]:
            raw_df.columns = raw_df.columns.str.strip()
            n = len(raw_df); pos = int((raw_df["label"]==1).sum()); neg = n-pos
            sl = raw_df["sequence"].str.len()
            print(f"\n  {split_name}:")
            print(f"    Samples            : {n:,}")
            print(f"    Positive (label=1) : {pos:,} ({100*pos/n:.1f}%)")
            print(f"    Negative (label=0) : {neg:,} ({100*neg/n:.1f}%)")
            print(f"    Seq length mean    : {sl.mean():.1f}")
            print(f"    Seq length std     : {sl.std():.1f}")
            print(f"    Seq length range   : [{sl.min()}, {sl.max()}]")
            print(f"    k-mer feature dim  : {INPUT_SIZE}  (k=4)")
            all_seqs = " ".join(raw_df["sequence"].str.upper().tolist())
            for b in ['A','C','G','T']:
                print(f"    Nucleotide {b}       : {100*all_seqs.count(b)/len(all_seqs):.1f}%")


        if dataset_name == "Promoter":
            print(f"\n{'='*60}\nSEC B: MODEL ARCHITECTURE & PARAMETER COUNTS\n{'='*60}")
            mlp_p   = sum(p.numel() for p in build_mlp(INPUT_SIZE,32).parameters())
            lstm_p  = sum(p.numel() for p in LSTMModel(INPUT_SIZE,32,1,3).parameters())
            child_p = sum(p.numel() for p in ChildMLP(INPUT_SIZE,32).parameters())
            svm_p   = sum(p.numel() for p in build_svm(INPUT_SIZE).parameters())
            print(f"\n  {'Model':<30} {'Parameters':>12} {'FHE-compatible':>16}")
            print(f"  {'-'*60}")
            print(f"  {'FHE-FL MLP (proposed)':<30} {mlp_p:>12,} {'Yes (6-bit)':>16}")
            print(f"  {'FL-LSTM (teacher)':<30} {lstm_p:>12,} {'No':>16}")
            print(f"  {'FL-LSTM Child MLP':<30} {child_p:>12,} {'Yes (6-bit)':>16}")
            print(f"  {'FL-SVM (linear)':<30} {svm_p:>12,} {'Yes (via Concrete-ML)':>16}")

        print(f"\n{'='*60}\nSEC C: CONVERGENCE & TIMING — {dataset_name}\n{'='*60}")
        curve = run_fl_mlp(
            X_train, y_train_int, X_test, y_test_int,
            num_clients=5, distribution="iid", dirichlet_alpha=None,
            seed=42, global_epochs=GLOBAL_EPOCHS, local_epochs=LOCAL_EPOCHS,
            input_size=INPUT_SIZE,
        )
        print(f"\n  {'Epoch':<8} {'Test Accuracy':>14} {'Wall Time (s)':>14}")
        print(f"  {'-'*38}")
        for ep,(acc,t) in enumerate(zip(curve["epoch_accs"],curve["epoch_times"]),1):
            print(f"  {ep:<8} {acc:>14.4f} {t:>14.2f}")
        print(f"\n  Mean epoch time : {np.mean(curve['epoch_times']):.2f}s ± {np.std(curve['epoch_times']):.2f}s")
        print(f"  Total train time: {sum(curve['epoch_times']):.2f}s")
        dataset_results["convergence"] = curve

        print(f"\n{'='*60}\nSEC D: BIT-WIDTH ABLATION — {dataset_name}\n{'='*60}")
        ref_mlp = MLPModel(INPUT_SIZE, HIDDEN_SIZE)
        cd = _split(X_train, y_train_int, 5, "iid", None, 42)
        lms = [MLPModel(INPUT_SIZE, HIDDEN_SIZE) for _ in range(5)]
        tpw = max(1, cpu_count()//5)
        for ep in range(GLOBAL_EPOCHS):
            gs = ref_mlp.get_state_dict()
            args = [(gs,*cd[i],i,HIDDEN_SIZE,INPUT_SIZE,1,LOCAL_EPOCHS,tpw) for i in range(5)]
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=5) as pool:
                res = pool.starmap(mlp_client_process, args)
            for r in res:
                if r.get("state_dict"): lms[r["client_id"]].set_state_dict(r["state_dict"])
            ref_mlp.set_state_dict(fedavg(lms))
        plain_m   = ref_mlp.evaluate(X_test, y_test_int)
        plain_acc = plain_m["accuracy"]
        print(f"\n  Plaintext (FP32) accuracy: {plain_acc:.4f}")
        print(f"\n  {'Bits':<8} {'Plain':>8} {'Sim':>8} {'Delta(sim-plain)':>18} {'F1':>8} {'ROC-AUC':>10}")
        print(f"  {'-'*64}")
        bw_results = {}
        for n_bits in BIT_WIDTHS:
            q_sd = quantize_model_weights(ref_mlp.get_state_dict(), n_bits)
            q_sd = quantize_model_weights(ref_mlp.get_state_dict(), n_bits)
            q_mlp = MLPModel(INPUT_SIZE, HIDDEN_SIZE); q_mlp.set_state_dict(q_sd)
            qm = q_mlp.evaluate(X_test, y_test_int)
            sim_acc, sim_f1 = simulate_accuracy(q_mlp, X_test, y_test_int, n_bits=n_bits)
            delta = (sim_acc - qm['accuracy']) if sim_acc is not None else float('nan')
            qm['sim_accuracy'] = sim_acc
            bw_results[n_bits] = qm
            sim_s = f"{sim_acc:.4f}" if sim_acc is not None else 'N/A'
            print(f"  {n_bits:<8} {qm['accuracy']:>8.4f} {sim_s:>8} {delta:>+18.4f} {qm['f1']:>8.4f} {qm['roc_auc']:>10.4f}")
        dataset_results["plaintext_mlp"] = plain_m

        print(f"\n{'#'*60}\n# SEC E: CLIENT COUNT (IID, MLP) — {dataset_name}\n{'#'*60}")
        client_results = []
        for n_clients in [5, 10, 20]:
            label = f"clients={n_clients} | IID | {dataset_name}"
            print(f"\nRunning: {label} ...")
            t0 = time.time()
            s = multi_seed(
                lambda seed,**kw: run_fl_mlp(seed=seed,**kw), SEEDS,
                X_train=X_train, y_train=y_train_int, X_test=X_test, y_test=y_test_int,
                num_clients=n_clients, distribution="iid", dirichlet_alpha=None,
                global_epochs=GLOBAL_EPOCHS, local_epochs=LOCAL_EPOCHS, input_size=INPUT_SIZE,
            )
            s["label"]=label; s["elapsed"]=time.time()-t0
            client_results.append(s)
            print_summary(label, s); print(f"  Wall time: {s['elapsed']:.1f}s")
        dataset_results["client_count"] = client_results

        print(f"\n{'#'*60}\n# SEC F: IID vs NON-IID — {dataset_name}\n{'#'*60}")
        noniid_results = []
        for dist, alpha in [("iid",None),("non_iid",0.5),("non_iid",0.1)]:
            label = f"clients=5 | {dist}" + (f" α={alpha}" if dist=="non_iid" else "") + f" | {dataset_name}"
            if any(r["label"]==label for r in noniid_results): continue
            print(f"\nRunning: {label} ...")
            t0 = time.time()
            s = multi_seed(
                lambda seed,**kw: run_fl_mlp(seed=seed,**kw), SEEDS,
                X_train=X_train, y_train=y_train_int, X_test=X_test, y_test=y_test_int,
                num_clients=5, distribution=dist, dirichlet_alpha=alpha,
                global_epochs=GLOBAL_EPOCHS, local_epochs=LOCAL_EPOCHS, input_size=INPUT_SIZE,
            )
            s["label"]=label; s["elapsed"]=time.time()-t0
            noniid_results.append(s)
            print_summary(label, s); print(f"  Wall time: {s['elapsed']:.1f}s")
        dataset_results["noniid"] = noniid_results

        print(f"\n{'#'*60}\n# SEC G: MODEL COMPARISON — {dataset_name}\n{'#'*60}")
        model_results = {}

        model_results["MLP"] = client_results[0]

        print(f"\nLSTM → Child MLP (5 clients, IID, multi-seed)...")
        lstm_per = [run_fl_lstm(X_train,y_train_int,X_test,y_test_int,
                                5,seed,GLOBAL_EPOCHS,LOCAL_EPOCHS,INPUT_SIZE)
                    for seed in SEEDS]
        keys = ["accuracy","f1","precision","recall","roc_auc","pr_auc"]
        lstm_s = {f"{k}_mean":np.mean([p[k] for p in lstm_per if k in p]) for k in keys}
        lstm_s.update({f"{k}_std":np.std([p[k] for p in lstm_per if k in p]) for k in keys})
        lstm_s["confusion_matrix_mean"] = np.array([p["confusion_matrix"] for p in lstm_per]).mean(axis=0)
        lstm_s["label"] = f"LSTM→Child | {dataset_name}"
        model_results["LSTM_Child"] = lstm_s
        print_summary(lstm_s["label"], lstm_s)

        print(f"\nSVM (5 clients, IID, multi-seed)...")
        svm_per = [run_fl_svm(X_train,y_train_int,X_test,y_test_int,
                               5,seed,GLOBAL_EPOCHS,LOCAL_EPOCHS,INPUT_SIZE)
                   for seed in SEEDS]
        svm_s = {f"{k}_mean":np.mean([p[k] for p in svm_per if k in p]) for k in keys}
        svm_s.update({f"{k}_std":np.std([p[k] for p in svm_per if k in p]) for k in keys})
        svm_s["confusion_matrix_mean"] = np.array([p["confusion_matrix"] for p in svm_per]).mean(axis=0)
        svm_s["label"] = f"FL-SVM | {dataset_name}"
        model_results["SVM"] = svm_s
        print_summary(svm_s["label"], svm_s)

        dataset_results["model_comparison"] = model_results
        all_dataset_results[dataset_name]    = dataset_results

        print(f"\n\n{'='*80}\nABLATION SUMMARY TABLE — {dataset_name}\n{'='*80}")
        print(f"{'Configuration':<45} {'Plain Acc':>12} {'Sim Acc':>9} {'Delta':>8} {'F1':>12} {'ROC-AUC':>12}")
        print(f"{'-'*102}")
        for s in client_results + noniid_results:
            sim_a = s.get('sim_accuracy_mean')
            delta = f"{sim_a - s['accuracy_mean']:>+.4f}" if sim_a is not None else '  N/A'
            sim_s = f"{sim_a:.4f}" if sim_a is not None else ' N/A '
            print(f"{s['label']:<45} {s['accuracy_mean']:.4f}+/-{s['accuracy_std']:.3f}  {sim_s:>9}  {delta:>8}  {s['f1_mean']:.4f}+/-{s['f1_std']:.3f}  {s['roc_auc_mean']:.4f}+/-{s['roc_auc_std']:.3f}")
        print(f"{'='*102}")

    print(f"\n\n{'='*80}\nSEC H: CROSS-DATASET MODEL COMPARISON\n{'='*80}")
    print(f"{'Model':<22} {'Dataset':<12} {'Acc (mean±std)':>16} {'F1':>14} {'ROC-AUC':>14}")
    print(f"{'-'*80}")
    for ds_name, ds_res in all_dataset_results.items():
        for model_name, s in ds_res["model_comparison"].items():
            print(f"  {model_name:<20} {ds_name:<12} "
                  f"{s['accuracy_mean']:.4f}±{s['accuracy_std']:.3f}  "
                  f"{s['f1_mean']:.4f}±{s['f1_std']:.3f}  "
                  f"{s['roc_auc_mean']:.4f}±{s['roc_auc_std']:.3f}")
    print(f"{'='*80}")

    print("\n\nGenerating figures...")
    COLORS = {"MLP":"#185FA5","LSTM_Child":"#1D9E75","SVM":"#D4537E"}
    FONT   = {"title":14,"axis":12,"tick":11,"label":10,"annot":9}
    saved  = []

    def save_fig(fig, fname):
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(fname)
        print(f"  Saved: {fname}")
    for ds_name, ds_res in all_dataset_results.items():
        cr = ds_res["client_count"]
        x  = np.arange(len(cr)); w = 0.32
        fig, ax = plt.subplots(figsize=(10, 6))
        b1 = ax.bar(x-w/2, [s["accuracy_mean"] for s in cr], w,
                    yerr=[s["accuracy_std"] for s in cr], capsize=5,
                    color=COLORS["MLP"], alpha=0.88, label="Accuracy")
        b2 = ax.bar(x+w/2, [s["f1_mean"]       for s in cr], w,
                    yerr=[s["f1_std"]       for s in cr], capsize=5,
                    color=COLORS["LSTM_Child"], alpha=0.88, label="F1 Score")
        ax.set_title(f"{ds_name}: Client Count Ablation (MLP, IID, mean±std)",
                     fontsize=FONT["title"], fontweight="bold", pad=12)
        ax.set_xlabel("Number of Clients", fontsize=FONT["axis"])
        ax.set_ylabel("Score", fontsize=FONT["axis"])
        ax.set_xticks(x)
        ax.set_xticklabels(["5 clients","10 clients","20 clients"], fontsize=FONT["tick"])
        ax.set_ylim(0, 1.08)
        ax.yaxis.grid(True, alpha=0.35, linestyle="--")
        ax.legend(fontsize=FONT["label"], framealpha=0.9)
        for bar in list(b1)+list(b2):
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+0.012,
                    f"{h:.4f}", ha="center", va="bottom", fontsize=FONT["annot"])
        fig.tight_layout()
        save_fig(fig, f"fig_client_count_{ds_name.lower()}.png")

    for ds_name, ds_res in all_dataset_results.items():
        mc     = ds_res["model_comparison"]
        models = list(mc.keys())
        x      = np.arange(len(models)); w = 0.32
        fig, ax = plt.subplots(figsize=(10, 6))
        b1 = ax.bar(x-w/2, [mc[m]["accuracy_mean"] for m in models], w,
                    yerr=[mc[m]["accuracy_std"] for m in models], capsize=5,
                    color=[COLORS[m] for m in models], alpha=0.88, label="Accuracy")
        b2 = ax.bar(x+w/2, [mc[m]["f1_mean"]       for m in models], w,
                    yerr=[mc[m]["f1_std"]       for m in models], capsize=5,
                    color=[COLORS[m] for m in models], alpha=0.50, label="F1 Score")
        ax.set_title(f"{ds_name}: Model Comparison (5 clients, IID, mean±std)",
                     fontsize=FONT["title"], fontweight="bold", pad=12)
        ax.set_xlabel("Model", fontsize=FONT["axis"])
        ax.set_ylabel("Score", fontsize=FONT["axis"])
        ax.set_xticks(x)
        ax.set_xticklabels(["FHE-FL MLP","LSTM→Child","FL-SVM"], fontsize=FONT["tick"])
        ax.set_ylim(0, 1.08)
        ax.yaxis.grid(True, alpha=0.35, linestyle="--")
        ax.legend(fontsize=FONT["label"], framealpha=0.9)
        for bar in list(b1)+list(b2):
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+0.012,
                    f"{h:.4f}", ha="center", va="bottom", fontsize=FONT["annot"])
        fig.tight_layout()
        save_fig(fig, f"fig_model_comparison_{ds_name.lower()}.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    linestyles = ["-", "--"]
    markers    = ["o", "s"]
    for i, (ds_name, ds_res) in enumerate(all_dataset_results.items()):
        accs = ds_res["convergence"]["epoch_accs"]
        times= ds_res["convergence"]["epoch_times"]
        eps  = list(range(1, len(accs)+1))
        ax.plot(eps, accs, linestyle=linestyles[i], marker=markers[i],
                linewidth=2.5, markersize=8, label=f"{ds_name} accuracy")
    ax.set_title("Convergence Curves — FHE-FL MLP (5 clients, IID, seed=42)",
                 fontsize=FONT["title"], fontweight="bold", pad=12)
    ax.set_xlabel("Global Epoch", fontsize=FONT["axis"])
    ax.set_ylabel("Test Accuracy", fontsize=FONT["axis"])
    ax.set_xticks(eps)
    ax.tick_params(labelsize=FONT["tick"])
    ax.yaxis.grid(True, alpha=0.35, linestyle="--")
    ax.legend(fontsize=FONT["label"], framealpha=0.9)
    fig.tight_layout()
    save_fig(fig, "fig_convergence.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    bw_labels = ["FP32"] + [f"{b}-bit" for b in BIT_WIDTHS]
    bw_x      = np.arange(len(bw_labels))
    for i, (ds_name, ds_res) in enumerate(all_dataset_results.items()):
        plain = ds_res["plaintext_mlp"]["accuracy"]
        accs  = [plain] + [ds_res["bitwidth"][b]["accuracy"] for b in BIT_WIDTHS]
        ax.plot(bw_x, accs, linestyle=linestyles[i], marker=markers[i],
                linewidth=2.5, markersize=8, label=ds_name)
        for xi, acc in zip(bw_x, accs):
            ax.annotate(f"{acc:.4f}", (xi, acc),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=FONT["annot"])
    ax.set_title("Quantization Bit-Width Ablation — FHE-FL MLP (simulated)",
                 fontsize=FONT["title"], fontweight="bold", pad=12)
    ax.set_xlabel("Quantization", fontsize=FONT["axis"])
    ax.set_ylabel("Test Accuracy", fontsize=FONT["axis"])
    ax.set_xticks(bw_x)
    ax.set_xticklabels(bw_labels, fontsize=FONT["tick"])
    ax.yaxis.grid(True, alpha=0.35, linestyle="--")
    ax.legend(fontsize=FONT["label"], framealpha=0.9)
    fig.tight_layout()
    save_fig(fig, "fig_bitwidth.png")

    print("\n" + "="*60 + "\nALL SECTIONS COMPLETE\n" + "="*60)
    print("  Figures saved:")
    for f in saved:
        print(f"    {f}")
    print("  Stdout above — tables for supplementary material")
