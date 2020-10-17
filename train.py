import argparse
import h5py
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--db", required=True, help="Path to HDF5 file")
    args = parser.parse_args()

    db = h5py.File(args.db, "r")

    # Pointer for train/test split
    i = int(db["data"].shape[0] * 0.75)

    print("[INFO] Training classifier ...")
    params = { "C": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0] }
    model = LogisticRegression(solver="lbfgs")
    classifier = GridSearchCV(model, params, cv=3, n_jobs=-1)
    classifier.fit(db["data"][:i], db["labels"][:i])

    print(f"Best hyperparameters: {classifier.best_params_}")

    print("[INFO] Evaluating classifier ...")
    preds = classifier.predict(db["data"][i:])
    report = classification_report(db["labels"][i:], preds, target_names=db["class_names"])
    print(report)

    db.close()

if __name__ == '__main__':
    main()
