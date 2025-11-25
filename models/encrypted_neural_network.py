import numpy as np
import time
import copy
from concrete.ml.sklearn import NeuralNetClassifier
from concrete.ml.deployment import FHEModelDev
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import f1_score

class FHEModel:
    def __init__(self, input_size=228, hidden_size=32, output_size=2, n_layers=3,
                 n_w_bits=8, maxiter=2, activation='relu', verbose=1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.n_w_bits = n_w_bits
        self.maxiter = maxiter
        self.activation = activation
        self.verbose = verbose
        self.model = NeuralNetClassifier(
            module__n_layers=self.n_layers,
            module__n_w_bits=self.n_w_bits,
            verbose=self.verbose
        )
        self.is_compiled = False
        self.fhe_circuit = None

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def compile_weights(self, X_sample):
        if not self.is_compiled:
            self.fhe_circuit = self.model.compile(X_sample)
            self.is_compiled = True
        return self

    def predict(self, X, encrypted=False):
        if not self.is_compiled:
            self.compile_weights(X[:1])

        return self.model.predict(X, fhe="execute")

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        f1_score = f1_score(y_test, y_pred)
        return {"accuracy": accuracy, "f1" : f1_score}

    def get_weights(self):
        try:
            params = {}
            if hasattr(self.model, "module_"):
                for name, param in self.model.module_.named_parameters():
                    params[name] = param.detach().cpu().numpy()
            return params
        except:
            return None

    def set_weights(self, weights):
        try:
            if hasattr(self.model, "module_"):
                for name, param in self.model.module_.named_parameters():
                    if name in weights:
                        param.data = weights[name]
            return True
        except:
            return False


class FederatedLearning:
    def __init__(self, num_clients, model_class, input_size=228, hidden_size=32, output_size=2,
                 n_layers=3, n_w_bits=3, maxiter=1, activation='relu', verbose=0,
                 global_epochs=5, local_epochs=1, alpha=0.01):
        self.num_clients = num_clients
        self.global_epochs = global_epochs
        self.local_epochs = local_epochs
        self.alpha = alpha
        self.verbose = verbose


        self.global_model = model_class(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            n_layers=n_layers,
            n_w_bits=n_w_bits,
            maxiter=maxiter,
            activation=activation,
            verbose=verbose
        )


        self.local_models = [
            model_class(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
                n_layers=n_layers,
                n_w_bits=n_w_bits,
                maxiter=maxiter,
                activation=activation,
                verbose=0
            ) for _ in range(num_clients)
        ]

        self.client_data = []

    def split_data(self, X, y, method="horizontal", non_iid_factor=0):
        self.client_data = []

        if method == "horizontal":
            n_samples = X.shape[0]
            samples_per_client = n_samples // self.num_clients

            if non_iid_factor > 0:
               X, y = shuffle(X, y, random_state=42)
            for i in range(self.num_clients):
                start_idx = i * samples_per_client
                end_idx = start_idx + samples_per_client if i < self.num_clients - 1 else n_samples
                self.client_data.append((X[start_idx:end_idx], y[start_idx:end_idx]))

        return self.client_data

    def distribute_global_weights(self):
        global_weights = self.global_model.get_weights()
        if global_weights:
            for model in self.local_models:
                model.set_weights(copy.deepcopy(global_weights))

    def aggregate_local_weights(self, aggregation_method="average", weights=None):
        if not weights:
            weights = [1/self.num_clients] * self.num_clients


        local_weights_list = [model.get_weights() for model in self.local_models]


        if None in local_weights_list or not local_weights_list:
            if self.verbose > 0:
                print("Warning: Some models don't have accessible weights. Skipping aggregation.")
            return


        if aggregation_method == "average":

            new_weights = {}
            for key in local_weights_list[0].keys():
                new_weights[key] = np.zeros_like(local_weights_list[0][key])

            for i, local_weights in enumerate(local_weights_list):
                for key in new_weights:
                    new_weights[key] += weights[i] * local_weights[key]

            self.global_model.set_weights(new_weights)


    def train_local_models(self):
        for i, model in enumerate(self.local_models):
            if i < len(self.client_data):
                X_local, y_local = self.client_data[i]
                model.fit(X_local, y_local)

                if self.verbose > 1:
                    print(f"Client {i+1} training complete")

    def train(self, X, y, split_method="horizontal", non_iid_factor=0,
          aggregation_method="average", compile_for_fhe=True,
          encryption_test=False, X_test=None, y_test=None):

      self.split_data(X, y, method=split_method, non_iid_factor=non_iid_factor)

      metrics_history = []

      sample_X = X[:min(100, len(X))]
      sample_y = y[:min(100, len(y))]
      self.global_model.fit(sample_X, sample_y)

      for epoch in range(self.global_epochs):
        if self.verbose > 0:
            print(f"\nGlobal Epoch {epoch+1}/{self.global_epochs}")

        self.distribute_global_weights()

        for i, model in enumerate(self.local_models):
            X_local, y_local = self.client_data[i]
            model.fit(X_local, y_local)

        self.aggregate_local_weights(aggregation_method=aggregation_method)
      if compile_for_fhe:
        if self.verbose > 0:
            print("\nCompiling global model for FHE execution...")
        self.global_model.compile_weights(X[:1])

        if  X_test is not None and y_test is not None:
            if self.verbose > 0:
                print("Testing encrypted prediction...")
            start_time = time.time()
            test_subset = min(30, len(X_test))
            encrypted_preds = self.global_model.predict(X_test[:test_subset], encrypted=True)
            if self.verbose > 0:
                end_time = time.time()
                print(f"Encrypted prediction successful on {test_subset} samples")
                accuracy = np.mean(encrypted_preds == y_test[:test_subset])
                print(f"Accuracy: {accuracy}")
                print(f"Runtime: {end_time-start_time} seconds")
                print(f"F1 Score: {f1_score(y_test[:test_subset], encrypted_preds)}")



        else:
            standard_preds = self.global_model.predict(X_test, encrypted=False)
            if self.verbose > 0:
                print("Standard prediction successful")

      return {"metrics_history": metrics_history, "final_model": self.global_model}





    def get_global_model(self):
        return self.global_model

    def get_prediction_function(self):
        model = self.global_model

        def predict_fn(X, encrypted=False):
            return model.predict(X, encrypted=encrypted)

        return predict_fn



if __name__ == "__main__":
    def preprocess_data(csv_path):
        encodings = {'a': [1, 0, 0, 0], 't': [0, 1, 0, 0], 'c': [0, 0, 1, 0], 'g': [0, 0, 0, 1]}
        headers = ["outcome", "id", "nucleotides"]

        # Load data
        data = pd.read_csv(csv_path)
        data.to_csv(csv_path, header=headers, index=False)
        data = pd.read_csv(csv_path)

        # Encode nucleotides
        def encode(nucleotides):
            return np.array([encodings[nucleotide] for nucleotide in nucleotides])

        X = data['nucleotides']
        X = np.array([encode(nucleotide) for nucleotide in X]).reshape(len(X), -1)

        y = data['outcome'].map({'+': 1, '-': 0}).values

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=42)
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)

        return X_train, X_test, y_train, y_test

    X_train, X_test, y_train, y_test = preprocess_data("data.csv")
    fed_learning = FederatedLearning(
         num_clients=5,
         model_class=FHEModel,
         global_epochs=10,
         local_epochs=2,
         verbose=1
     )
    print(y_test)
    result = fed_learning.train(
         X_train, y_train,
         split_method="horizontal",
         non_iid_factor=0.2,
         X_test=X_test,
         y_test=y_test
     )
    final_model = result["final_model"]
    start_time = time.time()
    encrypted_predictions = final_model.predict(X_train[:20], encrypted=True)
    end_time = time.time()
    encrypted_accuracy = np.mean(encrypted_predictions == y_train[:20])
    print(f"Encrypted Test Accuracy (first 20 samples): {encrypted_accuracy:.4f}")
    print(f"Encryped Test f1 (first 20 samples): {f1_score(y_train[:20], encrypted_predictions):.4f}")
    print(f"Encryped runtime (first 20 samples): {end_time-start_time}")
