# Load necessary libraries
import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.initializers import glorot_uniform
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.models import Model, load_model, Sequential
from keras.layers import (
    Input, Conv1D, Dense, Dropout, MaxPooling1D, BatchNormalization, Activation, Reshape, Embedding, 
    GlobalAveragePooling1D, Concatenate, LayerNormalization, Attention, Add, Permute, ZeroPadding1D, 
    AveragePooling1D, Flatten, GlobalMaxPooling1D, MaxPool1D, MultiHeadAttention
)
# Calculate and print various performance metrics for the best model
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report


# Set random seed for reproducibility
seed = 42
np.random.seed(seed)

# Install gdown library using pip
!pip install gdown

# Download necessary files using gdown
!gdown --id 14bHRkSgtEq-6exDKI5g9qJvfQ2708f8f
!gdown --id 1AuEFWdl1PumKEjv9z4lJ_JH78KRWdQ8g
!gdown --id 1A4cmVyIDTq23yHt-8IeIUDAJNzaLhmGw
!gdown --id 1AH3tuqJuKxZGPl2S6FBUD7KSmQbDUNPa
!gdown --id 1Ag_k3lrVjehCY-yraQJbw9Bxci0A_7Ni
!gdown --id 1Aoxrvc1ffIHZ4aM0azlU877ELLn9Lzw2
!gdown --id 1AEjAyc2mRbAWXgEOR1KujTZxh6997cdg
!gdown --id 1A-7ztu0KEG5opaOFBXHyg0XKV9yiraO-
!gdown --id 19viRSGCZ0dEtV7E2AS1Vn8vJbwH3tUKf
!gdown --id 1AQGDZtSkp-QNQNQUwl7MKz0iKmDZBFrJ
!gdown --id 1AAem6_anj1HM55_G09S4-HdZZPFelEjg
!gdown --id 1AWE3bJBuvALv75WcKb2yu8QAPq4cudei


# Define the input shape and number of classes
input_shape = (400, 1)
num_classes = 4

# Function to prepare the dataset from compressed CSV file
def prepare_df(f):
    # Read CSV file and shuffle the rows
    df = pd.read_csv(f, compression='gzip')
    df = df.sample(frac=1).reset_index(drop=True)

    # Split the data into input X and output y
    X = df.drop(columns='output')
    y = df['output']

    # Encode the output classes as integers
    y = pd.get_dummies(y, ['output'])
    y = y.astype(float)

    return [X, y]

# Load the compressed CSV files into a list and prepare the dataset
csv_files = ['train_data0.csv.gz', 'train_data1.csv.gz', 'train_data2.csv.gz', 'train_data3.csv.gz', 'train_data4.csv.gz', 
             'train_data5.csv.gz', 'train_data6.csv.gz', 'train_data7.csv.gz', 'train_data8.csv.gz', 'train_data9.csv.gz']
train_data_list = [prepare_df(f) for f in csv_files]

# Function to define callback functions for model training
def getCallbacks(checkpoint_filename):
    # Early stopping callback to stop the training if the validation accuracy doesn't improve for 3 epochs
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        verbose=1
    ) 
    return [early_stopping]

# Function to build the deep learning model
class PositionalEncoding(Layer):
    def __init__(self, sequence_length, embedding_dimension):
        super(PositionalEncoding, self).__init__()
        self.encoding = self.positional_encoding(sequence_length, embedding_dimension)
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'encoding': self.encoding
        })
        return config
        
    def call(self, inputs):
        return inputs + self.encoding[:, :tf.shape(inputs)[1], :]
    
    def positional_encoding(self, sequence_length, embedding_dimension):
        encoding = np.zeros((sequence_length, embedding_dimension))
        for pos in range(sequence_length):
            for i in range(embedding_dimension):
                if i % 2 == 0:
                    encoding[pos, i] = np.sin(pos / 10000**(2*i/embedding_dimension))
                else:
                    encoding[pos, i] = np.cos(pos / 10000**(2*(i-1)/embedding_dimension))
        encoding = tf.constant(encoding, dtype=tf.float32)
        return tf.expand_dims(encoding, axis=0)
    
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout_rate):
    # Input Embedding
    x = Embedding(input_dim=800, output_dim=16)(inputs)
    
    # positional encoding
    x = PositionalEncoding(800, 16)(x)
 
    # Multi-Head Attention
    for i in range(num_heads):
        # Multi-Head Attention
        Q = Conv1D(head_size, 1, activation=None)(x)
        K = Conv1D(head_size, 1, activation=None)(x)
        V = Conv1D(head_size, 1, activation=None)(x)
        attention_out = Attention(use_scale=True, dropout=dropout_rate)([Q, K, V])
        attention_out = Dropout(dropout_rate)(attention_out)
        # Add and Norm
        attention_out = Dense(x.shape[-1])(attention_out)
        attention_out = Add()([x, attention_out])
        attention_out = LayerNormalization()(attention_out)
        # Feedforward
        ff_out = Conv1D(ff_dim, 1, activation='gelu')(attention_out)
        ff_out = Dropout(dropout_rate)(ff_out)
        ff_out = Conv1D(ff_dim, 1, activation=None)(ff_out)
        ff_out = Dropout(dropout_rate)(ff_out)
        # Add and Norm
        ff_out = Dense(attention_out.shape[-1])(ff_out)
        x = Add()([attention_out, ff_out])
        x = LayerNormalization()(x)
    return x

def build_model(input_shape, num_classes):
    # Define the hyperparameters for the Transformer encoder
    head_size = 32
    num_heads = 8
    ff_dim = 64
    dropout_rate = 0.1
    
    # Define the input layer
    inputs = Input(shape=input_shape)

    # Define the convolutional layers
    x = Conv1D(8, 5, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)

    x = Conv1D(10, 5, padding='same')(x)
   # Add BatchNormalization and Activation layers to improve the accuracy of the model
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)

    x = Conv1D(16, 5, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)

    # Reshape the output of the convolutional layers for the Transformer encoder
    x = Reshape((800,))(x)

    # Apply the Transformer encoder
    x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout_rate)

    # Pool over the time dimension and apply a fully connected layer
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax')(x)

    # Create the Keras model
    model = Model(inputs=inputs, outputs=x)
    return model


# Initialize the KFold cross-validator and other parameters
kfold = KFold(n_splits=10, shuffle=True) 
val_acc_list = []
models = []
lr = 1e-5
batch_size = 50
num_epochs = 6
opt = Adam(lr)

# Build the model
model = build_model(input_shape, num_classes)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Define the number of folds for cross-validation
n_folds = 10

# Use KFold to split the data into n_folds number of train-test pairs
kf = KFold(n_splits=n_folds)

# Loop through each fold and split the data
for i, (train_index, val_index) in enumerate(kf.split(train_data_list)):
    
    # Split the data into train and test sets
    train_files = [train_data_list[j] for j in train_index]
    val_file = train_data_list[val_index[0]] 

    # Load the train and test data
    X_train, y_train = pd.concat([X for X, y in train_files]), pd.concat([y for X, y in train_files])
    X_val, y_val = val_file  
    
    # Define the filename for the model checkpoint
    checkpoint_filename = f'W_model_fold{i+1}.h5'
    checkfile_exit = f'/kaggle/input/pretrained/W_model_fold{i+1}.h5'
    
    if not os.path.isfile(checkfile_exit):
        # Train the model on the train data
        model.fit(X_train, y_train, callbacks=getCallbacks(checkpoint_filename), validation_data=(X_val, y_val), epochs=num_epochs, batch_size=batch_size)
        model.save_weights(checkpoint_filename)
    else:
        # Load the best model from the checkpoint file
        # Load the saved weights into a new instance of the model
        model = build_model(input_shape, num_classes)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        model.load_weights(checkfile_exit)  

    # Evaluate the model on the validation data
    scores = model.evaluate(X_val, y_val, verbose=0)
    print(f'Fold {i+1}: Validation loss: {scores[0]} - Validation accuracy: {scores[1]}')    
    
    # Append the accuracy score and model to the list of scores and models
    val_acc_list.append(scores[1] * 100)
    models.append(model)
    
    # Print the results 
    print(f"Fold {i+1}: Trained on {len(train_files)} files, tested on {val_index}")

# Calculate overall performance metrics by averaging the results from all folds 
avg_val_acc = np.mean(val_acc_list) 
print(f'Overall validation accuracy: {avg_val_acc:.4f}')

# Save the best model
val_best_model_index = np.argmax(val_acc_list)
val_best_model = models[val_best_model_index]
val_best_model.save_weights("val_best_model.h5")

# Plot the validation accuracy for each fold
x = np.arange(0,10,1) 
plt.plot(val_acc_list, label='Validation Accuracy')
plt.scatter(x, val_acc_list, label='Validation Accuracy')
plt.xlabel('Fold')
plt.ylabel('Val Accuracy') 
plt.xticks(x) 
plt.show()

# Load the best model and visualize its architecture
best_model = build_model(input_shape, num_classes)
best_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
best_model.load_weights('/kaggle/input/pretrained/W_model_fold7.h5')

print(best_model.summary())

# Visualize the model architecture
plot_model(best_model, to_file='model2.png', show_shapes=True)  

# Evaluate the model on the test data for each fold
test_acc_list = []
for i in range(10):
    # Load the saved weights into a new instance of the model
    test_model = build_model(input_shape, num_classes)
    test_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    test_model.load_weights(f'/kaggle/input/pretrained/W_model_fold{i+1}.h5')
    
    # Load the test data
    Test_df = pd.read_csv('test_data.csv.gz', compression='gzip') 
    # Split the test data into input X and output y    
    X_test = Test_df.drop(columns='output')
    y_test = Test_df['output']
    # Encode class values as integers
    y_test = pd.get_dummies(y_test, ['output'])
    # Convert all columns to floats
    y_test = y_test.astype(float)
    
    # Evaluate the model on the test data and print the results
    scores = test_model.evaluate(X_test, y_test, verbose=0)
    print(f'Testing loss: {scores[0]} - Testing accuracy: {scores[1]}')
    print("==========================================================")
    
    # Append the accuracy score to the list of test accuracy scores
    test_acc_list.append(scores[1] * 100)
    

# Load the test data and prepare it for testing
Test_df = pd.read_csv('test_data.csv.gz', compression='gzip') 
X_test = Test_df.drop(columns='output')
y_test = Test_df['output']
y_test = pd.get_dummies(y_test, ['output'])
y_test = y_test.astype(float)

# Evaluate the best model on the test data and print the results
scores = best_model.evaluate(X_test, y_test, verbose=0)
print(f'Testing loss: {scores[0]} - Testing accuracy: {scores[1]}')

# Plot the validation and test accuracy for each fold
x = np.arange(0,10,1) 
plt.plot(val_acc_list, label='Validation Accuracy')
plt.scatter(x, val_acc_list, label='Validation Accuracy')
plt.plot(test_acc_list, label='Testing Accuracy', c='r')
plt.scatter(x, test_acc_list, label='Testing Accuracy', c='r')
plt.xlabel('Fold')
plt.ylabel('Accuracy') 
plt.xticks(x) 
plt.legend()
plt.show()


# Get the predicted output of the best model on the test data
y_pred = best_model.predict(X_test).round()

# Calculate the confusion matrix for the predicted output and print it
confusion = multilabel_confusion_matrix(y_test, y_pred)
print('Confusion Matrix\n')
print(confusion)

# Calculate and print various performance metrics for the predicted output
print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_test, y_pred)))
print('Micro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='micro')))
print('Micro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='micro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='micro')))
print('Macro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='macro')))
print('Macro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='macro')))
print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='macro')))
print('Weighted Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='weighted')))
print('Weighted Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='weighted')))
print('Weighted F1-score: {:.2f}'.format(f1_score(y_test, y_pred, average='weighted')))
print('\nClassification Report\n')
print(classification_report(y_test, y_pred, target_names=['output_CAD', 'output_CHF', 'output_MI', 'output_Normal']))

# Create a heatmap of the confusion matrix for each label
labels = ['CAD', 'CHF', 'MI', 'Normal']
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
for i, ax in enumerate(axs.flat):
    sns.heatmap(confusion[i], annot=True, cmap='Blues', fmt='g', xticklabels=['True', 'False'], yticklabels=['True', 'False'], ax=ax)
    ax.set_title('Confusion Matrix for Label ' + labels[i])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
plt.tight_layout()
plt.show()

# Load a single test case and print its input and output
X_test.iloc[300:301,:]
y_test.iloc[300:301,:]

# Predict the output for the test case using the best model
pred = best_model.predict(X_test.iloc[300:301,:]).round()

# Map the predicted output to the corresponding class label
out_col = ['output_CAD', 'output_CHF', 'output_MI', 'output_Normal']
output = {out_col[i]:pred[0][i] for i in range(4)}
output_class = max(output, key=output.get).split('_')[-1]

# Print the predicted class label
print("output_class = " + output_class)

# End of the model training notebook