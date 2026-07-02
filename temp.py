import lightgbm as lgb

# Load the model
bst = lgb.Booster(model_file="delta_score_lgb_chunked.txt")

# Print categorical feature names
print("Categorical features:", bst.pandas_categorical)
print("\nAll feature names:", bst.feature_name())

# If 'Project_Type' was categorical, inspect its values
try:
    print("\nUnique project types seen during training:")
    for cat in bst.pandas_categorical['Project_Type']:
        print("-", cat)
except Exception as e:
    print("\nCan't directly extract Project_Type categories — model may not store them:", e)
