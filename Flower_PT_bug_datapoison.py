# In myfl7.py I reoved so many of the print functions to clear out the output
# In myfl6.py I added the main() function to read parameters from input

# import packages
print("Importing packages...")
import numpy as np
import random
import os
import argparse
from pathlib import Path
import copy
import json

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader, random_split, Dataset
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW

import time

import flwr as fl
from flwr.common import Metrics

print("flwr", fl.__version__)

VERBOSE = 0
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

print("######### Packages imported!")

DEVICE = torch.device("cuda")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)
disable_progress_bar()


# Reading the data
############################# ###############################################################################
############################# ###############################################################################
# Constants


# # Padding function
# def pad_sequence(seq, max_length, pad_value=0):
#     return seq + [pad_value] * (max_length - len(seq))

# # Tokenizer and vectorizer functions
# def tokenize_code(code):
#     tokens = code.split()
#     return tokens

# def vectorize_tokens(tokens, vocab):
#     return [vocab.get(token, vocab['<UNK>']) for token in tokens]

# class CodeDataset(Dataset):
#     def __init__(self, codes, labels, vocab, max_length):
#         self.codes = codes
#         self.labels = labels
#         self.vocab = vocab
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.codes)

#     def __getitem__(self, idx):
#         tokens = tokenize_code(self.codes[idx])
#         vectorized_code = vectorize_tokens(tokens, self.vocab)
#         padded_code = pad_sequence(vectorized_code, self.max_length)
#         return torch.tensor(padded_code, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)

class CodeDataset(Dataset):
    def __init__(self, codes, labels, tokenizer, client_idx, max_length=256):  # Reduced max_length
        self.codes = codes
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.client = client_idx

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        code = self.codes[idx]
        label = self.labels[idx]

        # Tokenize the code using CodeBERT tokenizer
        tokens = self.tokenizer(
            code,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = tokens["input_ids"].squeeze()
        
        if(self.client == 0):
            # print("self.client -------------------------------------------", self.client)
            input_ids = torch.add(input_ids, 100)
        # print(input_ids)
        attention_mask = tokens["attention_mask"].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.long)
        }


def load_dataset():     
    # Load the dataset from JSON file
    with open('./dataset_pairs.json', 'r') as file:
        data = json.load(file)

    # Extracting buggy and non-buggy code pairs
    codes = []
    labels = []
   
    for item in data:
        for key in item.keys():
            if 'buggy' in item[key] and 'fixed' in item[key]:
                codes.append(item[key]['buggy'])
                labels.append(1)  # Label 1 for buggy code
                codes.append(item[key]['fixed'])
                labels.append(0)  # Label 0 for non-buggy code
            
    
    # Ensure labels and codes have consistent lengths
    assert len(codes) == len(labels), "Codes and labels must have the same length"
    train_codes, test_codes, train_labels, test_labels = train_test_split(codes, labels, test_size=0.1, random_state=42)
    
    return ((train_codes, train_labels), (test_codes, test_labels))




def partition_training_data(X_train, Y_train, tokenizer, num_clients, valid_percent=0.1):
    
    
     # Split training set into `num_clients` partitions to simulate different local datasets
    print(num_clients)    
    # print(len(X_train))    
    partition_size = len(X_train) // num_clients
    
    lengths = [partition_size] * num_clients
    trainset = list(zip(X_train, Y_train))
    datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for i in range(len(datasets)):
        len_val = int(len(datasets[i]) * valid_percent)  # 10 % validation set
        len_train = len(datasets[i]) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(datasets[i], lengths, torch.Generator().manual_seed(42))

        train_codes, train_labels = list(zip(*ds_train))
        valid_codes, valid_labels = list(zip(*ds_val))

        train_dataset = CodeDataset(train_codes, train_labels, tokenizer, i)
        valid_dataset = CodeDataset(valid_codes, valid_labels, tokenizer, i)

        trainloaders.append(DataLoader(train_dataset, batch_size=8, shuffle=True))
        valloaders.append(DataLoader(valid_dataset, batch_size=8))
        
    return trainloaders, valloaders


#############################################################
# FL
#############################################################

def deepcopy_dict(d):
    new_dict = {}
    for key, value in d.items():
        if isinstance(value, dict):
            new_dict[key] = deepcopy_dict(value)
        else:
            new_dict[key] = value
    return new_dict

def train(model, trainloader, epochs: int):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=5e-5)
    # optimizer = torch.optim.Adam(model.parameters())
    model.train()
    
    
    all_epochs_history = []
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        epoch_results = {}
        for i, batch in enumerate(trainloader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.logits, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        
        epoch_results['loss'] = epoch_loss.cpu().detach().numpy()
        epoch_results['accuracy'] = epoch_acc
        
        # Append the average ACC and LOSS (both Training and Validation) for each epoch into a list
        all_epochs_history.append(epoch_results)
    
        print(f"Epoch {epoch+1}: train loss {epoch_results['loss']}, accuracy {epoch_results['accuracy']}")
    
    # Calculate the average of accuracy and loss for all epochs
    results_avg = deepcopy_dict(all_epochs_history[0])
    for k in results_avg.keys():
        for i in range(1, len(all_epochs_history)):
            results_avg[k] += all_epochs_history[i][k]
        results_avg[k] = results_avg[k]/len(all_epochs_history)
    
    return results_avg
        



        
def test(model, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    model.eval()
    with torch.no_grad():          
        for batch in testloader:
            # print(batch)
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # print(outputs)
            # loss += outputs.loss
            loss += criterion(outputs.logits, labels).item()
            
            _, predicted = torch.max(outputs.logits, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy




################### client.py
#**Client Configuration:**

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, valloader, device_ID) -> None:
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.device_ID = device_ID

        
    def get_parameters(self, config):
        '''
        Get parameters of the local model.
        '''
        return get_parameters(self.model)

    
    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""
        print("*********************** FITING THE MODEL ON TRAINING SET **********************************")
        print()
        print(f"-------------------- Training Device_ID: {self.device_ID} ---------------------")  
        set_parameters(self.model, parameters)
        train_history = train(self.model, self.trainloader, epochs=config["local_epochs"])     
        
        print("Training results (history) for this fitting round : ", train_history)  
        print(" ----------------------------------------------------------------")
        print()   
        # Return updated model parameters and results
        return get_parameters(self.model), len(self.trainloader), train_history

    
    def evaluate(self, parameters, config):
        '''
        Evaluate parameters on the locally held test set.
        '''
        print("********************** EVALUATING THE MODEL ON VALIDATION SET ****************************")
        print()
        print(f"------------------- Evaluating Device_ID: {self.device_ID} -------------------") 
        set_parameters(self.model, parameters)
        val_loss, val_acc = test(self.model, self.valloader)
        
        print("**EVALUATION LOSS : ", val_loss)
        print("**EVALUATION ACCURACY : " , val_acc)
        print(" ----------------------------------------------------------------")
        print()
        return float(val_loss), len(self.valloader), {"accuracy": float(val_acc)}
       
    
#####################

def get_client_fn(trainloaders, valloaders):
    
    def client_fn(cid: str) -> fl.client.Client:
        # Create model
        cmodel = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2).to(DEVICE)  
        # cmodel = Net(input_size).to(DEVICE) 
        
        # cid starts from 0
        # Here, I'll choose/create the dataset for one client (cid)
        device_ID = int(cid) + 1
        print("cid = ", cid , "     device_ID = ",  device_ID) 

        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]

        # Create and return client    
        return FlowerClient(cmodel, trainloader, valloader, device_ID).to_client()   
    return client_fn




######### **Server Configuration:**

# Create an evaluation function for the server (to evaluate the model parameters on a piece of data)
def get_evaluate_fn(testloader):
    """Return an evaluation function for server-side evaluation."""
    # Create a model for the server:

    # Load the CodeBERT model for binary classification
    model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2).to(DEVICE) 

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:

        set_parameters(model, parameters)  # Update model with the latest parameters
        loss, accuracy = test(model, testloader)
        print()
        print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
        print()
        return loss, {"accuracy": accuracy}

    return evaluate



######### **Configuration:**

def fit_config(server_round: int):
    """Return training configuration dict for each round.
    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 16,
        "local_epochs": 2 , # ORIGINAL: "local_epochs": 1 if server_round < 2 else 2,
        #"learning_rate": str(0.001),
    }
    return config


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    # To rtake care of : No evaluate_metrics_aggregation_fn provided
    # which takes care of : app_fit: metrics_distributed {}
    """
    print()
    print("Number of clients used:", len(metrics))
    
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    # Aggregate and return custom metric (weighted average)
    result = sum(accuracies) / sum(examples)
    return {"accuracy": result}


def get_parameters(smodel) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in smodel.state_dict().items()]
    # Can Apply Encryption here


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
    

    

######### **IMPLEMENTAION:**

################################################# main()
def main():
    print("\nStarting the main()...")

    # Timer starts
    start = time.time()
    
    global NUM_CLIENTS
    NUM_CLIENTS = args.num_clients
    
    global INPUT_SHAPE  
    INPUT_SHAPE = (32, 32, 3)
    
    global NUM_CLASS
    NUM_CLASS = 2
    
    global MAX_SEQ_LENGTH
    MAX_SEQ_LENGTH = 713  # This should match the maximum length expected by the model
    ########################################################################

    ## 2. Prepare your dataset
    ((X_train, y_train), (X_test, y_test)) = load_dataset()
    
    # Load the CodeBERT tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    
    test_dataset = CodeDataset(X_test, y_test, tokenizer, 1000)
    testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    valid_percent = 0.1
    trainloaders, valloaders = partition_training_data(X_train, y_train, tokenizer, NUM_CLIENTS, valid_percent)
    
    
    # 4. Create FedAvg strategy
    ''' For the purpose of simulation, all clients are available.
    So, we put "fraction_fit" to a really small number and then control the number of training clients
    by setting "min_fit_clients". This is true for the "fraction_evaluate" and "min_evaluate_clients" as well.
    '''
    num_server_rounds = args.num_server_rounds
    num_clients_per_round_fit = args.num_cl_fit
    num_clients_per_round_eval = args.num_cl_eval

    print("########## Parameters:")
    print("Total number of clients: ", NUM_CLIENTS)
    print("Number of clients for training: ", num_clients_per_round_fit)
    print("Number of clients for evaluation: ", num_clients_per_round_eval)
    print("Number of server rounds: ", num_server_rounds)
    print("######################################### ")
    print()
    print()
    
    print("########## Set the Strategy...")
    strategy = fl.server.strategy.FedAvg(
            fraction_fit = 0.000001,  # Sample 70% of available clients for training
            fraction_evaluate = 0.000001,  # Sample 10% of available clients for evaluation

            min_fit_clients = num_clients_per_round_fit,  # Never sample less than 4 clients for training
            min_evaluate_clients = num_clients_per_round_eval,  # Never sample less than 2 clients for evaluation
            min_available_clients = NUM_CLIENTS, 
            
            # Server-side evaluation can be enabled by passing an evaluation function to evaluate_fn.
            evaluate_fn = get_evaluate_fn(testloader),  
            on_fit_config_fn = fit_config,
        
            initial_parameters = fl.common.ndarrays_to_parameters(get_parameters(RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2))),
        
            fit_metrics_aggregation_fn = weighted_average,
            evaluate_metrics_aggregation_fn = weighted_average, 
    )
    
    print()
    print()
    
    # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
    client_resources = {"num_cpus": 1, "num_gpus": 1.0}
    if DEVICE.type == "cuda":
        print("CUDA CUDA CUDA CUDA CUDA CUDA CUDA CUDA CUDA ")
        
        client_resources = {"num_cpus": 1, "num_gpus": 1.0}

    
    print("########## Starting simulation...")   
    ## 5. Start Simulation
    final_result = fl.simulation.start_simulation(
        client_fn = get_client_fn(trainloaders, valloaders),
        num_clients = NUM_CLIENTS,
        config = fl.server.ServerConfig(num_rounds=num_server_rounds),
        strategy = strategy,
        client_resources=client_resources,
    )


    ## 6. Save your results
    print(3*"\n")
    print("########## Printing Final Results: ")
    print(final_result)

    # Updating the previous total time and lap number
    end = time.time()
    print("Program Execution Time (Sec)", end - start)


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()
    # Add the arguments
    parser.add_argument('-p', '--data_path', type=str, default='.')
    parser.add_argument('-n', '--num_clients', type=int, default=100, help='Number of clients')
    parser.add_argument('-s', '--num_server_rounds', type=int, default=3, help='number of server rounds')
    parser.add_argument('-t', '--num_cl_fit', type=int, default=3, help='Number of clients for training')
    parser.add_argument('-e', '--num_cl_eval', type=int, default=2, help='Number of clients for evaluating')

    args = parser.parse_args()    
    main()


"""

######################################################################
NOTE: After each run, remove the log files in the following directory:
ls -l /tmp/ray/
rm -fr /tmp/ray/*

Note: due to space error, I moved the /ray folder. from now on:
rm -fr ray/*
ls -l ray/

######################################################################

# run like this:

CUDA_VISIBLE_DEVICES=2 python FL_PT_bug.py --num_clients 5 --num_cl_fit 2 --num_cl_eval 2 --num_server_rounds 10

"""




'''
fit_round 1: strategy sampled 5 clients (out of 10)

**************************** FITING THE MODEL:**********************************
-------------------- Training Device_ID: 5 ---------------------
-------------------- Training Device_ID: 7 ---------------------
-------------------- Training Device_ID: 3 ---------------------
-------------------- Training Device_ID: 2 ---------------------
-------------------- Training Device_ID: 8 ---------------------
...

Server-side evaluation loss 0.059791420376300815 / accuracy 0.3125

evaluate_round 1: strategy sampled 5 clients (out of 10)

**************************** EVALUATING THE MODEL:****************************
------------------- Evaluating Device_ID: 5 -------------------
------------------- Evaluating Device_ID: 2 -------------------
------------------- Evaluating Device_ID: 1 -------------------
------------------- Evaluating Device_ID: 6 -------------------
------------------- Evaluating Device_ID: 9 -------------------



fit_round 2: strategy sampled 5 clients (out of 10)

**************************** FITING THE MODEL:**********************************
-------------------- Training Device_ID: 5 ---------------------
-------------------- Training Device_ID: 7 ---------------------
-------------------- Training Device_ID: 3 ---------------------
-------------------- Training Device_ID: 2 ---------------------
-------------------- Training Device_ID: 8 ---------------------
...

Server-side evaluation loss 0.04894713588953018 / accuracy 0.4259

evaluate_round 2: strategy sampled 5 clients (out of 10)

**************************** EVALUATING THE MODEL:****************************
------------------- Evaluating Device_ID: 5 -------------------
------------------- Evaluating Device_ID: 2 -------------------
------------------- Evaluating Device_ID: 1 -------------------
------------------- Evaluating Device_ID: 6 -------------------
------------------- Evaluating Device_ID: 9 -------------------


'''
