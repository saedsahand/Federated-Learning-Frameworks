import tensorflow as tf
import tensorflow_federated as tff
import numpy as np
from sklearn.model_selection import train_test_split
import json
from transformers import RobertaTokenizer

# Load dataset
def load_dataset():
    with open('./dataset_pairs.json', 'r') as file:
        data = json.load(file)

    codes = []
    labels = []
   
    for item in data:
        for key in item.keys():
            if 'buggy' in item[key] and 'fixed' in item[key]:
                codes.append(item[key]['buggy'])
                labels.append(1)  # Label 1 for buggy code
                codes.append(item[key]['fixed'])
                labels.append(0)  # Label 0 for non-buggy code

    assert len(codes) == len(labels), "Codes and labels must have the same length"
    train_codes, test_codes, train_labels, test_labels = train_test_split(codes, labels, test_size=0.1, random_state=42)
    
    return ((train_codes, train_labels), (test_codes, test_labels))


# Preprocess data
def preprocess_data(codes, labels, tokenizer, max_length=256):
    input_ids = []
    attention_masks = []
    
    for code in codes:
        tokens = tokenizer(
            code,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="tf"
        )
        input_ids.append(tokens['input_ids'])
        attention_masks.append(tokens['attention_mask'])
    
    dataset = tf.data.Dataset.from_tensor_slices(({
        'input_ids': tf.concat(input_ids, axis=0),
        'attention_mask': tf.concat(attention_masks, axis=0)
    }, labels))
    
    return dataset


# Build Keras model
def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(256,)),
        tf.keras.layers.Embedding(input_dim=30522, output_dim=128),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    return model


# Define the TFF model function
def model_fn():
    keras_model = build_model()
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]

    # Wrapping the Keras model in TFF's API
    return tff.learning.from_keras_model(
        keras_model=keras_model,
        input_spec=input_spec,
        loss=loss,
        metrics=metrics
    )


# Create the federated averaging process
def create_federated_process(input_spec):
    iterative_process = tff.learning.build_federated_averaging_process(
        model_fn=lambda: model_fn(),
        client_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=0.001),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0)
    )
    return iterative_process


# Create federated data from client data
def make_federated_data(client_data, client_ids):
    return [
        preprocess_data(client_data.create_tf_dataset_for_client(x), tokenizer)
        for x in client_ids
    ]


def main():
    # Load the dataset
    ((X_train, y_train), (X_test, y_test)) = load_dataset()
    
    # Load the tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    
    # Prepare the dataset for federated learning
    train_dataset = preprocess_data(X_train, y_train, tokenizer)
    test_dataset = preprocess_data(X_test, y_test, tokenizer)
    
    # Extract input specification from the dataset
    input_spec = train_dataset.element_spec
    
    # Build and initialize the federated learning process
    iterative_process = create_federated_process(input_spec)
    state = iterative_process.initialize()
    
    # Simulate federated learning with 5 clients
    federated_train_data = make_federated_data(train_dataset, list(range(5)))
    
    # Training for multiple rounds
    NUM_ROUNDS = 10
    for round_num in range(1, NUM_ROUNDS + 1):
        state, metrics = iterative_process.next(state, federated_train_data)
        print(f'Round {round_num}, Metrics={metrics}')


if __name__ == '__main__':
    main()
