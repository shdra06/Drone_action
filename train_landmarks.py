import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

DATA_FILE = "hand_data.csv"
MODEL_FILE = "gesture_model.pkl"

def main():
    try:
        df = pd.read_csv(DATA_FILE)
        if df.empty:
            print("Error: Dataset is empty.")
            return

        print(f"Loaded {len(df)} samples.")
        print("Class distribution:")
        print(df['label'].value_counts())

        # 2. Preprocess
        X = df.drop("label", axis=1).values
        y = df["label"].values

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

       
        print("Training Neural Network (MLPClassifier)...")
        model = MLPClassifier(hidden_layer_sizes=(128, 64, 32),
                              activation='relu',
                              solver='adam',
                              max_iter=500,
                              random_state=42,
                              verbose=True)

       
        model.fit(X_train, y_train)

        
        accuracy = model.score(X_test, y_test)
        print(f"\nFinal Test Accuracy: {accuracy*100:.2f}%")

        
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {MODEL_FILE}")

    except FileNotFoundError:
        print(f"Error: {DATA_FILE} not found. Run collect_landmarks.py first.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
