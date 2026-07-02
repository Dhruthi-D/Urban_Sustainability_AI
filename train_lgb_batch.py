import lightgbm as lgb
import pandas as pd

# --- File paths ---
csv_file = "training_dataset.csv"
model_file = "delta_score_lgb_chunked.txt"

# --- Parameters ---
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 7,
    'verbose': 1
}

# --- Categorical columns ---
categorical_cols = ["Landuse", "Project_Type", "Material"]

# --- Chunk settings ---
chunk_size = 200_000  # adjust based on RAM
first_chunk = True
iteration = 0

# --- Read CSV in chunks ---
reader = pd.read_csv(csv_file, chunksize=chunk_size)

for chunk in reader:
    iteration += 1
    print(f"Processing chunk {iteration}...")

    # Ensure categorical columns are category type
    for col in categorical_cols:
        if col in chunk.columns:
            chunk[col] = chunk[col].astype('category')

    # Features and label
    X = chunk.drop(columns=["Delta_Score"])
    y = chunk["Delta_Score"]

    # Create LightGBM Dataset
    train_data = lgb.Dataset(X, label=y, categorical_feature=categorical_cols, free_raw_data=False)

    if first_chunk:
        print("Training on first chunk...")
        bst = lgb.train(
            params,
            train_data,
            num_boost_round=500
        )
        first_chunk = False
    else:
        print("Incremental training on next chunk...")
        bst = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            init_model=bst
        )

# --- Save model ---
bst.save_model(model_file)
print(f"Model saved successfully as {model_file}")
