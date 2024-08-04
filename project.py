import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Helper function to plot and save samples of text data
def plot_samples_text(df, class_names, samples_per_class=3, file_name="example_classes_text.png"):
    num_classes = len(class_names)
    plt.figure(figsize=(12, 12))
    for cls in range(num_classes):
        idxs = np.flatnonzero(df['Genre'] == cls)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + cls + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.text(0.5, 0.5, df.loc[idx, 'Title'], ha='center', va='center', wrap=True)
            plt.axis('off')
            if i == 0:
                plt.title(class_names[cls])
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

# Load and preprocess the data
def load_data(filepath):
    with open(filepath, 'r') as f:
        books = f.read().split('======')
    data = []
    for book in books:
        lines = book.strip().split('\n')
        if len(lines) < 3:
            continue
        title = lines[0].strip()
        genre = lines[1].strip()
        summary = ' '.join(lines[2:]).strip()
        data.append((title, genre, summary))
    return data

print("Data loading function defined.")

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_filepath = os.path.join(BASE_DIR, 'books', 'books-train.txt')
val_filepath = os.path.join(BASE_DIR, 'books', 'books-validation.txt')
test_filepath = os.path.join(BASE_DIR, 'books', 'books-test.txt')

# Load datasets
training_data = load_data(train_filepath)
validation_data = load_data(val_filepath)
test_data = load_data(test_filepath)

print("Data loaded successfully.")

# Convert data to DataFrame for easier manipulation
train_df = pd.DataFrame(training_data, columns=['Title', 'Genre', 'Summary'])
val_df = pd.DataFrame(validation_data, columns=['Title', 'Genre', 'Summary'])
test_df = pd.DataFrame(test_data, columns=['Title', 'Genre', 'Summary'])

print("DataFrames created successfully.")

# Combine title and summary for text features
train_df['Text'] = train_df['Title'] + ' ' + train_df['Summary']
val_df['Text'] = val_df['Title'] + ' ' + val_df['Summary']
test_df['Text'] = test_df['Title'] + ' ' + test_df['Summary']

print("Text combined successfully.")

# Encode genres as numeric labels
label_encoder = LabelEncoder()
train_df['Genre'] = label_encoder.fit_transform(train_df['Genre'])
val_df['Genre'] = label_encoder.transform(val_df['Genre'])
test_df['Genre'] = label_encoder.transform(test_df['Genre'])

print("Labels encoded successfully.")

# Extract features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_df['Text']).toarray()
X_val = vectorizer.transform(val_df['Text']).toarray()
X_test = vectorizer.transform(test_df['Text']).toarray()

y_train = to_categorical(train_df['Genre'])
y_val = to_categorical(val_df['Genre'])
y_test = to_categorical(test_df['Genre'])

print("Features extracted successfully.")

# Print basic statistics
print(f'Training data shape: {X_train.shape}')
print(f'Training labels shape: {y_train.shape}')
print(f'Validation data shape: {X_val.shape}')
print(f'Validation labels shape: {y_val.shape}')
print(f'Test data shape: {X_test.shape}')
print(f'Test labels shape: {y_test.shape}')

# Number of samples per class
classes, counts = np.unique(train_df['Genre'], return_counts=True)
for cls, count in zip(classes, counts):
    print(f'Class {cls} ({label_encoder.inverse_transform([cls])[0]}): {count} samples')

print("Basic statistics printed successfully.")

# Plot samples from each class (text data)
class_names = label_encoder.classes_
plot_samples_text(train_df, class_names)

# Build the model
model = Sequential([
    Dense(512, activation='relu', input_shape=(5000,)),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

print("Model built successfully.")

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Model compiled successfully.")

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=128)

print("Model trained successfully.")

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test)
print(f'Test accuracy: {test_accuracy:.4f}')

# Plot training history
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()

plot_training_history(history)

# Confusion matrix
y_pred = model.predict(X_test).argmax(axis=1)
y_true = y_test.argmax(axis=1)
cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.show()

print("Script executed successfully.")
