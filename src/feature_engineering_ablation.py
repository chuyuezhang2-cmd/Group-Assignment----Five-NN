import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"


def extract_title(name: str) -> str:
    parts = name.split(',')
    if len(parts) < 2:
        return "Rare"
    title_part = parts[1].split(".")[0].strip()
    title_map = {
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
    }
    title_part = title_map.get(title_part, title_part)
    if title_part in ["Mr", "Mrs", "Miss", "Master"]:
        return title_part
    return "Rare"


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Age"] = out["Age"].fillna(out["Age"].median())
    out["Embarked"] = out["Embarked"].fillna(out["Embarked"].mode()[0])
    out["Cabin"] = out["Cabin"].fillna("Unknown")
    out["Fare"] = out["Fare"].fillna(out["Fare"].median())
    return out


def engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = base_features(df)
    out["Title"] = out["Name"].apply(extract_title)
    out["FamilySize"] = out["SibSp"] + out["Parch"] + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)
    out["FamilySizeGroup"] = pd.cut(
        out["FamilySize"], bins=[0, 1, 4, 11], labels=["Solo", "Small", "Large"]
    )
    out["CabinDeck"] = out["Cabin"].str[0]
    out["TicketGroupSize"] = out.groupby("Ticket")["Ticket"].transform("count")
    out["HasCabin"] = (out["Cabin"] != "Unknown").astype(int)
    out["FarePerPerson"] = out["Fare"] / out["FamilySize"]
    out["LogFare"] = np.log1p(out["Fare"])
    out["AgePclass"] = out["Age"] * out["Pclass"]
    out["IsChild"] = (out["Age"] < 16).astype(int)
    return out


def prepare_data(df: pd.DataFrame, mode: str):
    if mode == "base":
        data = base_features(df)
        drop_cols = ["Name", "Ticket", "Cabin", "PassengerId"]
        cat_cols = ["Sex", "Embarked"]
        num_cols = ["Age", "Fare", "SibSp", "Parch", "Pclass"]
    elif mode == "engineered":
        data = engineered_features(df)
        drop_cols = ["Name", "Ticket", "Cabin", "PassengerId"]
        cat_cols = ["Sex", "Embarked", "Title", "CabinDeck", "FamilySizeGroup"]
        num_cols = [
            "Age",
            "Fare",
            "SibSp",
            "Parch",
            "Pclass",
            "FamilySize",
            "IsAlone",
            "TicketGroupSize",
            "HasCabin",
            "FarePerPerson",
            "LogFare",
            "AgePclass",
            "IsChild",
        ]
    else:
        raise ValueError("mode must be 'base' or 'engineered'")

    data = data.drop(columns=drop_cols)
    X = data.drop(columns=["Survived"])
    y = data["Survived"]

    return X, y, cat_cols, num_cols


def encode_scale(X_train, X_valid, cat_cols, num_cols):
    encoder = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
    scaler = StandardScaler()

    X_train_cat = encoder.fit_transform(X_train[cat_cols])
    X_valid_cat = encoder.transform(X_valid[cat_cols])

    X_train_num = scaler.fit_transform(X_train[num_cols])
    X_valid_num = scaler.transform(X_valid[num_cols])

    X_train_out = np.hstack([X_train_num, X_train_cat])
    X_valid_out = np.hstack([X_valid_num, X_valid_cat])

    return X_train_out, X_valid_out


def evaluate_mode(df: pd.DataFrame, mode: str):
    X, y, cat_cols, num_cols = prepare_data(df, mode)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_out, X_valid_out = encode_scale(X_train, X_valid, cat_cols, num_cols)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train_out, y_train)

    preds = model.predict(X_valid_out)
    acc = accuracy_score(y_valid, preds)
    f1 = f1_score(y_valid, preds)

    return acc, f1


def main():
    train_path = DATA_DIR / "train.csv"
    df = pd.read_csv(train_path)

    base_acc, base_f1 = evaluate_mode(df, "base")
    eng_acc, eng_f1 = evaluate_mode(df, "engineered")

    print("Base features:")
    print(f"  Accuracy: {base_acc:.4f}")
    print(f"  F1-score: {base_f1:.4f}")
    print("Engineered features:")
    print(f"  Accuracy: {eng_acc:.4f}")
    print(f"  F1-score: {eng_f1:.4f}")


if __name__ == "__main__":
    main()
