import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
import uuid

# Suppress TensorFlow warnings
tf.config.experimental.enable_op_determinism()
tf.get_logger().setLevel('ERROR')

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load and preprocess the Cleveland heart disease dataset
def load_data():
    print("\nStep 1: Loading and preprocessing data...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
               'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    try:
        data = pd.read_csv(url, names=columns, na_values='?')
        print(f"Dataset loaded successfully. Shape: {data.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

    # Remove instances with missing values
    data = data.dropna()
    print(f"Dataset after removing missing values: {data.shape}")

    # Convert target to binary (0: healthy, 1: heart disease)
    data['target'] = data['target'].apply(lambda x: 1 if x > 0 else 0)
    print(f"Target distribution: \n{data['target'].value_counts()}")

    X = data.drop('target', axis=1)
    y = data['target']

    return X, y

# Feature selection using chi-square
def select_features(X_train, X_test, y_train, k=11):
    print("\nStep 2: Performing feature selection with chi-square...")
    try:
        selector = SelectKBest(score_func=chi2, k=k)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)

        # Get selected feature indices
        selected_features = selector.get_support()
        feature_names = X_train.columns[selected_features].tolist()
        print(f"Selected {k} features: {feature_names}")

        return X_train_selected, X_test_selected, feature_names
    except Exception as e:
        print(f"Error in feature selection: {e}")
        return None, None, None

# Build χ²-DNN model
def build_chi2_dnn_model(input_dim, neurons1=50, neurons2=2):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(neurons1, activation='relu'),
        Dense(neurons2, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    return model

# Build χ²-ANN model
def build_chi2_ann_model(input_dim, neurons=50):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(neurons, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    return model

# Build conventional DNN model
def build_dnn_model(input_dim, neurons1=2, neurons2=4):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(neurons1, activation='relu'),
        Dense(neurons2, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    return model

# Build conventional ANN model
def build_ann_model(input_dim, neurons=50):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(neurons, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    return model

# Calculate evaluation metrics
def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TP = cm[1,1]
    TN = cm[0,0]
    FP = cm[0,1]
    FN = cm[1,0]

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
    mcc_denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    mcc = ((TP * TN) - (FP * FN)) / mcc_denominator if mcc_denominator != 0 else 0

    return accuracy, sensitivity, specificity, mcc

# Grid search for hyperparameter tuning
def grid_search_dnn(X_train, y_train, X_val, y_val, input_dim, model_type='dnn'):
    print(f"\nStep 4: Performing grid search for {model_type.upper()}...")
    neuron_configs = [(50, 2), (30, 5), (20, 10)] if model_type == 'dnn' else [(50,), (30,), (20,)]
    best_accuracy = 0
    best_config = None
    best_model = None

    for config in neuron_configs:
        if model_type == 'dnn':
            model = build_chi2_dnn_model(input_dim, config[0], config[1]) if 'chi2' in model_type else build_dnn_model(input_dim, config[0], config[1])
        else:
            model = build_chi2_ann_model(input_dim, config[0]) if 'chi2' in model_type else build_ann_model(input_dim, config[0])

        history = model.fit(X_train, y_train,
                          epochs=200,
                          batch_size=32,
                          validation_data=(X_val, y_val),
                          callbacks=[EarlyStopping(patience=20, restore_best_weights=True)],
                          verbose=0)

        y_pred_prob = model.predict(X_val, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        accuracy = accuracy_score(y_val, y_pred)
        print(f"Config {config}: Validation Accuracy = {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_config = config
            best_model = model

    print(f"Best {model_type.upper()} config: {best_config}, Validation Accuracy: {best_accuracy:.4f}")
    return best_model, best_config, best_accuracy

# Plot ROC curves for all models
def plot_roc_curves(results):
    print("\nStep 5: Generating ROC curves...")
    plt.figure()
    for model_name, (fpr, tpr, roc_auc) in results.items():
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
        print(f"{model_name} AUC: {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc='lower right')
    plt.savefig('roc_curves_comparison.png')
    plt.close()

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, model_name):
    print(f"\nGenerating confusion matrix for {model_name}...")
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix for {model_name}:\n{cm}")
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrix_{model_name.lower().replace("χ²-", "chi2_")}.png')
    plt.close()

# Plot training history
def plot_training_history(history, model_name):
    print(f"\nGenerating training history plot for {model_name}...")
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'training_history_{model_name.lower().replace("χ²-", "chi2_")}.png')
    plt.close()

# Main execution
def main():
    # Step 1: Load data
    X, y = load_data()
    if X is None or y is None:
        print("Failed to load data. Exiting.")
        return

    # Step 2: Split data
    print("\nStep 2: Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Test set shape: {X_test.shape}")

    # Step 3: Normalize features for chi-square
    print("\nStep 3: Normalizing features for χ² models...")
    chi2_scaler = MinMaxScaler()
    X_train_chi2 = chi2_scaler.fit_transform(X_train)
    X_test_chi2 = chi2_scaler.transform(X_test)
    X_val_chi2 = chi2_scaler.transform(X_val)

    # Convert to DataFrame to preserve column names
    X_train_chi2 = pd.DataFrame(X_train_chi2, columns=X_train.columns)
    X_test_chi2 = pd.DataFrame(X_test_chi2, columns=X_test.columns)
    X_val_chi2 = pd.DataFrame(X_val_chi2, columns=X_val.columns)
    print("Features normalized for χ² models.")

    # Step 4: Feature selection for χ² models
    X_train_selected, X_test_selected, selected_features = select_features(X_train_chi2, X_test_chi2, y_train)
    if X_train_selected is None:
        print("Feature selection failed. Exiting.")
        return
    X_val_selected = chi2_scaler.transform(X_val)[:, [X_train.columns.get_loc(col) for col in selected_features]]

    # Step 5: Standardize features
    print("\nStep 5: Standardizing features...")
    nn_scaler = StandardScaler()
    X_train_selected_scaled = nn_scaler.fit_transform(X_train_selected)
    X_test_selected_scaled = nn_scaler.transform(X_test_selected)
    X_val_selected_scaled = nn_scaler.transform(X_val_selected)

    full_scaler = StandardScaler()
    X_train_full = full_scaler.fit_transform(X_train)
    X_test_full = full_scaler.transform(X_test)
    X_val_full = full_scaler.transform(X_val)
    print("Features standardized for all models.")

    # Step 6: Compute class weights
    print("\nStep 6: Computing class weights...")
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    print(f"Class weights: {class_weight_dict}")

    # Step 7: Train and evaluate models
    results = {}

    # χ²-DNN
    print("\nStep 7.1: Training χ²-DNN...")
    chi2_dnn_model, chi2_dnn_config, chi2_dnn_val_acc = grid_search_dnn(X_train_selected_scaled, y_train, X_val_selected_scaled, y_val, X_train_selected_scaled.shape[1], 'chi2_dnn')

    y_pred_prob_chi2_dnn = chi2_dnn_model.predict(X_test_selected_scaled, verbose=0)
    y_pred_chi2_dnn = (y_pred_prob_chi2_dnn > 0.5).astype(int).flatten()

    accuracy_chi2_dnn, sensitivity_chi2_dnn, specificity_chi2_dnn, mcc_chi2_dnn = calculate_metrics(y_test, y_pred_chi2_dnn)
    fpr_chi2_dnn, tpr_chi2_dnn, _ = roc_curve(y_test, y_pred_prob_chi2_dnn)
    roc_auc_chi2_dnn = auc(fpr_chi2_dnn, tpr_chi2_dnn)
    results['χ²-DNN'] = (fpr_chi2_dnn, tpr_chi2_dnn, roc_auc_chi2_dnn)

    print("\nχ²-DNN Model Results:")
    print(f"Accuracy: {accuracy_chi2_dnn:.4f}")
    print(f"Sensitivity: {sensitivity_chi2_dnn:.4f}")
    print(f"Specificity: {specificity_chi2_dnn:.4f}")
    print(f"MCC: {mcc_chi2_dnn:.4f}")
    print(f"AUC: {roc_auc_chi2_dnn:.4f}")

    # χ²-ANN
    print("\nStep 7.2: Training χ²-ANN...")
    chi2_ann_model, chi2_ann_config, chi2_ann_val_acc = grid_search_dnn(X_train_selected_scaled, y_train, X_val_selected_scaled, y_val, X_train_selected_scaled.shape[1], 'chi2_ann')

    y_pred_prob_chi2_ann = chi2_ann_model.predict(X_test_selected_scaled, verbose=0)
    y_pred_chi2_ann = (y_pred_prob_chi2_ann > 0.5).astype(int).flatten()

    accuracy_chi2_ann, sensitivity_chi2_ann, specificity_chi2_ann, mcc_chi2_ann = calculate_metrics(y_test, y_pred_chi2_ann)
    fpr_chi2_ann, tpr_chi2_ann, _ = roc_curve(y_test, y_pred_prob_chi2_ann)
    roc_auc_chi2_ann = auc(fpr_chi2_ann, tpr_chi2_ann)
    results['χ²-ANN'] = (fpr_chi2_ann, tpr_chi2_ann, roc_auc_chi2_ann)

    print("\nχ²-ANN Model Results:")
    print(f"Accuracy: {accuracy_chi2_ann:.4f}")
    print(f"Sensitivity: {sensitivity_chi2_ann:.4f}")
    print(f"Specificity: {specificity_chi2_ann:.4f}")
    print(f"MCC: {mcc_chi2_ann:.4f}")
    print(f"AUC: {roc_auc_chi2_ann:.4f}")

    # Conventional DNN
    print("\nStep 7.3: Training Conventional DNN...")
    dnn_model, dnn_config, dnn_val_acc = grid_search_dnn(X_train_full, y_train, X_val_full, y_val, X_train_full.shape[1], 'dnn')

    y_pred_prob_dnn = dnn_model.predict(X_test_full, verbose=0)
    y_pred_dnn = (y_pred_prob_dnn > 0.5).astype(int).flatten()

    accuracy_dnn, sensitivity_dnn, specificity_dnn, mcc_dnn = calculate_metrics(y_test, y_pred_dnn)
    fpr_dnn, tpr_dnn, _ = roc_curve(y_test, y_pred_prob_dnn)
    roc_auc_dnn = auc(fpr_dnn, tpr_dnn)
    results['DNN'] = (fpr_dnn, tpr_dnn, roc_auc_dnn)

    print("\nConventional DNN Model Results:")
    print(f"Accuracy: {accuracy_dnn:.4f}")
    print(f"Sensitivity: {sensitivity_dnn:.4f}")
    print(f"Specificity: {specificity_dnn:.4f}")
    print(f"MCC: {mcc_dnn:.4f}")
    print(f"AUC: {roc_auc_dnn:.4f}")

    # Conventional ANN
    print("\nStep 7.4: Training Conventional ANN...")
    ann_model, ann_config, ann_val_acc = grid_search_dnn(X_train_full, y_train, X_val_full, y_val, X_train_full.shape[1], 'ann')

    y_pred_prob_ann = ann_model.predict(X_test_full, verbose=0)
    y_pred_ann = (y_pred_prob_ann > 0.5).astype(int).flatten()

    accuracy_ann, sensitivity_ann, specificity_ann, mcc_ann = calculate_metrics(y_test, y_pred_ann)
    fpr_ann, tpr_ann, _ = roc_curve(y_test, y_pred_prob_ann)
    roc_auc_ann = auc(fpr_ann, tpr_ann)
    results['ANN'] = (fpr_ann, tpr_ann, roc_auc_ann)

    print("\nConventional ANN Model Results:")
    print(f"Accuracy: {accuracy_ann:.4f}")
    print(f"Sensitivity: {sensitivity_ann:.4f}")
    print(f"Specificity: {specificity_ann:.4f}")
    print(f"MCC: {mcc_ann:.4f}")
    print(f"AUC: {roc_auc_ann:.4f}")

    # Step 8: Generate visualizations
    plot_roc_curves(results)
    plot_confusion_matrix(y_test, y_pred_chi2_dnn, 'χ²-DNN')
    plot_confusion_matrix(y_test, y_pred_chi2_ann, 'χ²-ANN')
    plot_confusion_matrix(y_test, y_pred_dnn, 'DNN')
    plot_confusion_matrix(y_test, y_pred_ann, 'ANN')

    # Step 9: Train models again to capture history for plotting
    print("\nStep 9: Retraining models to capture training history...")
    chi2_dnn_history = chi2_dnn_model.fit(X_train_selected_scaled, y_train,
                                        epochs=200,
                                        batch_size=32,
                                        validation_data=(X_val_selected_scaled, y_val),
                                        callbacks=[EarlyStopping(patience=20, restore_best_weights=True)],
                                        verbose=0)
    chi2_ann_history = chi2_ann_model.fit(X_train_selected_scaled, y_train,
                                        epochs=200,
                                        batch_size=32,
                                        validation_data=(X_val_selected_scaled, y_val),
                                        callbacks=[EarlyStopping(patience=20, restore_best_weights=True)],
                                        verbose=0)
    dnn_history = dnn_model.fit(X_train_full, y_train,
                               epochs=200,
                               batch_size=32,
                               validation_data=(X_val_full, y_val),
                               callbacks=[EarlyStopping(patience=20, restore_best_weights=True)],
                               verbose=0)
    ann_history = ann_model.fit(X_train_full, y_train,
                               epochs=200,
                               batch_size=32,
                               validation_data=(X_val_full, y_val),
                               callbacks=[EarlyStopping(patience=20, restore_best_weights=True)],
                               verbose=0)

    plot_training_history(chi2_dnn_history, 'χ²-DNN')
    plot_training_history(chi2_ann_history, 'χ²-ANN')
    plot_training_history(dnn_history, 'DNN')
    plot_training_history(ann_history, 'ANN')
    
    # Save the models
    chi2_dnn_model.save("chi2_dnn_model.h5")
    chi2_ann_model.save("chi2_ann_model.h5")
    dnn_model.save("dnn_model.h5")
    ann_model.save("ann_model.h5")

if __name__ == "__main__":
    main()