import pickle
from pathlib import Path
import numpy as np
import scipy as sp
import os
import json
import torch
from sklearn.metrics import roc_auc_score
import gc
from tqdm import tqdm

gpu = "0"
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
dropout_mlp = 0.5
dropout_gru = 0.25

MODEL_PATH = "falcon-7b"
RNN_MODEL_ATTRIBUTION_NAME = "rnn_hallucination_detection_attribution.pt"
FFN_MODEL_LOGITS_NAME = "ffn_hallucination_detection_logits.pt"
FFN_MODEL_LAYER_NAME = "ffn_hallucination_detection_{activation}_layer{layer}.pt"




class FFHallucinationClassifier(torch.nn.Module):
    def __init__(self, input_shape, dropout=dropout_mlp):
        super().__init__()
        self.dropout = dropout
        
        self.linear_relu_stack =torch.nn.Sequential(
            torch.nn.Linear(input_shape, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(256, 2)
            )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    

class RNNHallucinationClassifier(torch.nn.Module):
    def __init__(self, dropout=dropout_gru):
        super().__init__()
        hidden_dim = 128
        num_layers = 4
        self.gru = torch.nn.GRU(1, hidden_dim, num_layers, dropout=dropout, batch_first=True, bidirectional=False)
        self.linear = torch.nn.Linear(hidden_dim, 2)
    
    def forward(self, seq):
        gru_out, _ = self.gru(seq)
        return self.linear(gru_out)[-1, -1, :]


def load_probing_model(model_name, model):
        model_path = os.path.join(MODEL_PATH, model_name)
        saved_data = torch.load(model_path, weights_only=True)

        if isinstance(saved_data, list):
            if len(saved_data) > 0:
                if hasattr(saved_data[0], 'state_dict'):
                    model.load_state_dict(saved_data[0].state_dict())
                elif isinstance(saved_data[0], dict):
                    model.load_state_dict(saved_data[0])
                else:
                    print(f"Unexpected format in list: {type(saved_data[0])}")
                    return
            else:
                print("Empty list found in saved file")
                return
        elif isinstance(saved_data, dict):
            model.load_state_dict(saved_data)
        else:
            print(f"Unexpected save format: {type(saved_data)}")
            return
        
        model.eval()
        print("Model loaded successfully. Ready for inference.")

        return model


def predict(predict_logits, predict_fully_connected, predict_attention, artifact_dict):
    correct = torch.tensor(artifact_dict['none_conflict']).long().to(device)

    print(artifact_dict.keys())

    if predict_logits:
        logits_correct = correct[:len(artifact_dict['logits'])]
    else:
        logits_correct = []
    
    if predict_fully_connected:
        mlp_correct = correct[:len(artifact_dict['first_fully_connected'])]
    else:
        mlp_correct = []

    if predict_attention:
        attn_correct = correct[:len(artifact_dict['first_attention'])]
    else:
        attn_correct = []

    logits_preds = []
    mlp_preds = {}
    attn_preds = {}

    with torch.no_grad():
        # --- Logits ---
        if predict_logits:
            print("Processing logits...")                
            first_logits = np.stack([
                sp.special.softmax(i[j].to(torch.float32).cpu().numpy()) 
                for i, j in zip(artifact_dict['logits'], artifact_dict['start_pos'])
            ])

            first_logits = torch.tensor(first_logits).float().to(device)

            logits_model = FFHallucinationClassifier(first_logits.shape[1])
            logits_model = load_probing_model(FFN_MODEL_LOGITS_NAME, logits_model).to(device)

            pred = torch.nn.functional.softmax(logits_model(first_logits), dim=1)
            logits_preds = (pred[:, 1] > 0.5).long().cpu()

        # --- Fully Connected (MLP) ---
        if predict_fully_connected:
            print("Processing fully connected layers...")
            for layer in range(artifact_dict['first_fully_connected'][0].shape[0]):
                
                inputs = np.stack([i[layer] for i in artifact_dict['first_fully_connected']])
                inputs = torch.tensor(inputs).float().to(device)

                mlp_layer_model = FFHallucinationClassifier(inputs.shape[1])
                mlp_layer_model = load_probing_model(
                    FFN_MODEL_LAYER_NAME.format(activation="fully", layer=layer),
                    mlp_layer_model
                ).to(device)

                pred = torch.nn.functional.softmax(mlp_layer_model(inputs), dim=1)
                mlp_preds[layer] = (pred[:, 1] > 0.5).long().cpu()

        # --- Attention Layers ---
        if predict_attention:
            print("Processing attention layers...")
            for layer in range(artifact_dict['first_attention'][0].shape[0]):
                attn_correct = correct[:len(artifact_dict['first_attention'])]
                inputs = np.stack([i[layer] for i in artifact_dict['first_attention']])
                inputs = torch.tensor(inputs).float().to(device)

                attn_layer_model = FFHallucinationClassifier(inputs.shape[1])
                attn_layer_model = load_probing_model(
                    FFN_MODEL_LAYER_NAME.format(activation="attn", layer=layer),
                    attn_layer_model
                ).to(device)

                pred = torch.nn.functional.softmax(attn_layer_model(inputs), dim=1)
                attn_preds[layer] = (pred[:, 1] > 0.5).long().cpu()

    return logits_correct, logits_preds, mlp_correct, mlp_preds, attn_correct, attn_preds


def to_numpy_array(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, list):
        return np.array([to_numpy_array(i) for i in x])
    else:
        return np.array(x)


def main(predict_logits=True, predict_fully_connected=True, predict_attention=True):
    print("---------- Classifier Model Prediction ---------")
    print(f"Predicting logits: {predict_logits}, fully connected layers: {predict_fully_connected}, attention layers: {predict_attention}")

    all_results = {}
    all_logits_correct, all_logits_preds, all_mlp_correct, all_mlp_preds, all_attn_correct, all_attn_preds = [], [], [], {}, [], {}

    # Load and merge all pickle files into a single 'results' dict    
    artifacts_dir = "./results/kc"
    labels_dirs = os.listdir(artifacts_dir)
    print(f"Found {len(labels_dirs)} activation directories: {labels_dirs}")


    for label_dir in labels_dirs:
        print(f"\nProcessing label directory: {label_dir}")
        activations = os.listdir(os.path.join(artifacts_dir, label_dir))

        for activation_dir in activations:
            if not predict_fully_connected and "mlp" in activation_dir:
                print(f"Skipping fully connected activation directory: {activation_dir}")
                continue
            if not predict_attention and "attn" in activation_dir:
                print(f"Skipping attention activation directory: {activation_dir}")
                continue

            print(f"\t - Activation directory: {activation_dir}")
            
            pickle_dir = os.path.join(artifacts_dir, label_dir, activation_dir)
            inference_results = list(Path(pickle_dir).glob("*.pickle"))

            print(f"\t\t -> Found {len(inference_results)} pickle files in {pickle_dir}")

            artifact_dict = {}
            for results_file in tqdm(inference_results, desc="Loading pickle files"):
                try:
                    with open(results_file, "rb") as infile:
                        batch_results = pickle.load(infile)

                    for k, v in batch_results.items():
                        if k not in artifact_dict:
                            artifact_dict[k] = v
                        else:
                            # Extend lists, concatenate arrays, or handle as needed
                            if isinstance(artifact_dict[k], np.ndarray):
                                artifact_dict[k] = np.concatenate([artifact_dict[k], v], axis=0)
                            elif isinstance(artifact_dict[k], list):
                                artifact_dict[k].extend(v)
                            else:
                                # For other types, you may need to handle accordingly
                                pass
                except EOFError:
                    print(f"Error: Ran out of input while reading {results_file}. Skipping this file.")
                    continue
                except Exception as e:
                    print(f"Error processing file {e}")
                    continue
                
                gc.collect()  # Clear memory after processing each file

            # Predict
            logits_correct, logits_preds, mlp_correct, mlp_preds, attn_correct, attn_preds = predict(predict_logits, predict_fully_connected, predict_attention, artifact_dict)
            
            all_logits_correct.extend(logits_correct)
            all_logits_preds.extend(logits_preds)

            all_mlp_correct.extend(mlp_correct)
            for layer, preds in mlp_preds.items():
                if layer not in all_mlp_preds:
                    all_mlp_preds[layer] = preds
                else:
                    all_mlp_preds[layer] = np.concatenate([all_mlp_preds[layer], preds], axis=0)

            all_attn_correct.extend(attn_correct)
            for layer, preds in attn_preds.items():
                if layer not in all_attn_preds:
                    all_attn_preds[layer] = preds
                else:
                    all_attn_preds[layer] = np.concatenate([all_attn_preds[layer], preds], axis=0)

            gc.collect()  # Clear memory after processing each file

            print(f" ---- Check:{artifact_dict.keys()} ----")
    
    print("---------- Compute Metrics ---------")

    # -- Logits Metrics ---
    if predict_logits:
        print("Processing logits metrics...")
        all_logits_correct = to_numpy_array(all_logits_correct)
        all_logits_preds = to_numpy_array(all_logits_preds)

        first_logits_roc = roc_auc_score(all_logits_correct, all_logits_preds)
        print("all_logits_preds shape:", all_logits_preds.shape)
        first_logits_acc = (all_logits_preds == all_logits_correct).mean()

        all_results['first_logits_roc'] = first_logits_roc
        all_results['first_logits_acc'] = first_logits_acc

    # -- Fully Connected Metrics ---
    if predict_fully_connected:
        print("Processing fully connected layers...")
        all_mlp_correct = to_numpy_array(all_mlp_correct)
        for layer, preds in all_mlp_preds.items():
            preds = to_numpy_array(preds)

            layer_roc = roc_auc_score(all_mlp_correct, preds)
            layer_acc = (preds == all_mlp_correct).mean()

            all_results[f'first_fully_connected_roc_{layer}'] = layer_roc
            all_results[f'first_fully_connected_acc_{layer}'] = layer_acc

    # -- Attention Metrics ---
    if predict_attention:
        print("Processing attention layers...")
        all_attn_correct = to_numpy_array(all_attn_correct)
        for layer, preds in all_attn_preds.items():
            preds = to_numpy_array(preds)

            layer_roc = roc_auc_score(all_attn_correct, preds)
            layer_acc = (preds == all_attn_correct).mean()

            all_results[f'first_attention_roc_{layer}'] = layer_roc
            all_results[f'first_attention_acc_{layer}'] = layer_acc

    print("---------- Results Summary ---------")
    print("All results collected:")
    print(all_results.keys())

    for k,v in all_results.items():
        print(k, v)

    # Save the results to a pickle file
    metrics_filename = ""
    if predict_logits:
        metrics_filename += "logits_"
    if predict_fully_connected:
        metrics_filename += "fully_connected_"
    if predict_attention:
        metrics_filename += "attention_"
    metrics_filename += "kc_metrics.json"
    print(f"Saving metrics to {metrics_filename}")

    with open(metrics_filename, "w") as f:
        json.dump(all_results, f)


if __name__ == "__main__":
    #main(predict_attention=False, predict_fully_connected=True, predict_logits=True)
    main(predict_attention=True, predict_fully_connected=False, predict_logits=False)