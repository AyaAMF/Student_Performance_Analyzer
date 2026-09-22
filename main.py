# ============================================
# STUDENT PERFORMANCE AI PROJECT
# MODEL OPTIMIZATION VERSION
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score
)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================
# 1. LOAD DATASET
# ============================================

df = pd.read_csv(
    "data/student-mat.csv",
    sep=";"
)

print("\n================================")
print("DATASET LOADED")
print("================================")

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nFirst 5 Rows:")
print(df.head())


# ============================================
# 2. DATA INFORMATION
# ============================================

print("\n================================")
print("DATASET INFORMATION")
print("================================")

df.info()


# ============================================
# 3. MISSING VALUES
# ============================================

print("\n================================")
print("MISSING VALUES")
print("================================")

print(df.isnull().sum())


# ============================================
# 4. TARGET
# ============================================

target = "G3"

print("\n================================")
print("TARGET INFORMATION")
print("================================")

print("Target:", target)

print(
    "Average Final Grade:",
    round(df[target].mean(), 2)
)

print(
    "Highest Final Grade:",
    df[target].max()
)

print(
    "Lowest Final Grade:",
    df[target].min()
)


# ============================================
# 5. CORRELATION ANALYSIS
# ============================================

print("\n================================")
print("CORRELATION ANALYSIS")
print("================================")

study_correlation = df["studytime"].corr(df["G3"])

absence_correlation = df["absences"].corr(df["G3"])

failure_correlation = df["failures"].corr(df["G3"])

print(
    "Study Time vs Final Grade:",
    round(study_correlation, 3)
)

print(
    "Absences vs Final Grade:",
    round(absence_correlation, 3)
)

print(
    "Failures vs Final Grade:",
    round(failure_correlation, 3)
)


# ============================================
# 6. VISUALIZATION
# ============================================

plt.figure(figsize=(8, 5))

plt.scatter(
    df["studytime"],
    df["G3"]
)

plt.xlabel("Study Time")
plt.ylabel("Final Grade")

plt.title(
    "Study Time vs Final Grade"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================
# 7. PREPARE X AND Y
# ============================================

X = df.drop(
    columns=["G3"]
)

y = df["G3"]


# ============================================
# 8. REMOVE G1 AND G2
# ============================================

X = X.drop(
    columns=["G1", "G2"]
)


# ============================================
# 9. COLUMN TYPES
# ============================================

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    exclude=["object"]
).columns.tolist()


print("\n================================")
print("FEATURE TYPES")
print("================================")

print("\nCategorical Columns:")
print(categorical_columns)

print("\nNumerical Columns:")
print(numerical_columns)


# ============================================
# 10. TRAIN / TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42
)


print("\n================================")
print("TRAIN / TEST SPLIT")
print("================================")

print(
    "Training Samples:",
    len(X_train)
)

print(
    "Testing Samples:",
    len(X_test)
)


# ============================================
# 11. PREPROCESSOR
# ============================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "categorical",

            OneHotEncoder(
                handle_unknown="ignore"
            ),

            categorical_columns
        )

    ],

    remainder="passthrough"
)


# ============================================
# 12. BASELINE RANDOM FOREST
# ============================================

baseline_model = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",

            RandomForestRegressor(

                n_estimators=200,

                random_state=42

            )
        )

    ]
)


print("\n================================")
print("TRAINING BASELINE RANDOM FOREST")
print("================================")


baseline_model.fit(
    X_train,
    y_train
)


# ============================================
# 13. BASELINE PREDICTION
# ============================================

baseline_predictions = baseline_model.predict(
    X_test
)


# ============================================
# 14. BASELINE EVALUATION
# ============================================

baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions
)

baseline_mse = mean_squared_error(
    y_test,
    baseline_predictions
)

baseline_rmse = baseline_mse ** 0.5

baseline_r2 = r2_score(
    y_test,
    baseline_predictions
)


print("\n================================")
print("BASELINE RESULTS")
print("================================")

print(
    "MAE:",
    round(baseline_mae, 3)
)

print(
    "RMSE:",
    round(baseline_rmse, 3)
)

print(
    "R2:",
    round(baseline_r2, 3)
)


# ============================================
# 15. CROSS VALIDATION
# ============================================

print("\n================================")
print("CROSS VALIDATION")
print("================================")

cv_scores = cross_val_score(

    baseline_model,

    X_train,

    y_train,

    cv=5,

    scoring="r2"

)


print(
    "Fold Scores:",
    cv_scores.round(3)
)

print(
    "Average CV R2:",
    round(cv_scores.mean(), 3)
)

print(
    "CV Standard Deviation:",
    round(cv_scores.std(), 3)
)


# ============================================
# 16. HYPERPARAMETER TUNING
# ============================================

print("\n================================")
print("HYPERPARAMETER TUNING")
print("================================")


param_grid = {

    "model__n_estimators": [
        100,
        200,
        300
    ],

    "model__max_depth": [
        None,
        5,
        10,
        15
    ],

    "model__min_samples_split": [
        2,
        5,
        10
    ],

    "model__min_samples_leaf": [
        1,
        2,
        4
    ]

}


grid_search = GridSearchCV(

    estimator=baseline_model,

    param_grid=param_grid,

    cv=5,

    scoring="r2",

    n_jobs=-1,

    verbose=1

)


print(
    "Searching for best parameters..."
)


grid_search.fit(
    X_train,
    y_train
)


# ============================================
# 17. BEST PARAMETERS
# ============================================

print("\n================================")
print("BEST PARAMETERS")
print("================================")


print(
    grid_search.best_params_
)


print(
    "Best CV R2:",
    round(
        grid_search.best_score_,
        3
    )
)


# ============================================
# 18. BEST MODEL
# ============================================

best_model = grid_search.best_estimator_


# ============================================
# 19. BEST MODEL PREDICTION
# ============================================

best_predictions = best_model.predict(
    X_test
)


# ============================================
# 20. BEST MODEL EVALUATION
# ============================================

best_mae = mean_absolute_error(
    y_test,
    best_predictions
)

best_mse = mean_squared_error(
    y_test,
    best_predictions
)

best_rmse = best_mse ** 0.5

best_r2 = r2_score(
    y_test,
    best_predictions
)


print("\n================================")
print("OPTIMIZED RANDOM FOREST RESULTS")
print("================================")

print(
    "MAE:",
    round(best_mae, 3)
)

print(
    "RMSE:",
    round(best_rmse, 3)
)

print(
    "R2:",
    round(best_r2, 3)
)


# ============================================
# 21. MODEL COMPARISON
# ============================================

print("\n================================")
print("MODEL COMPARISON")
print("================================")


comparison_results = pd.DataFrame({

    "Model": [

        "Baseline Random Forest",

        "Optimized Random Forest"

    ],

    "MAE": [

        baseline_mae,

        best_mae

    ],

    "RMSE": [

        baseline_rmse,

        best_rmse

    ],

    "R2": [

        baseline_r2,

        best_r2

    ]

})


print(
    comparison_results.round(3)
)


# ============================================
# 22. FEATURE IMPORTANCE
# ============================================

print("\n================================")
print("FEATURE IMPORTANCE")
print("================================")


rf_model = best_model.named_steps["model"]

fitted_preprocessor = (
    best_model
    .named_steps["preprocessor"]
)


feature_names = (
    fitted_preprocessor
    .get_feature_names_out()
)


importances = rf_model.feature_importances_


feature_importance_df = pd.DataFrame({

    "Feature": feature_names,

    "Importance": importances

})


feature_importance_df = (
    feature_importance_df
    .sort_values(
        by="Importance",
        ascending=False
    )
)


print(
    feature_importance_df
    .head(15)
    .round(4)
)


# ============================================
# 23. FEATURE IMPORTANCE VISUALIZATION
# ============================================

top_features = (
    feature_importance_df
    .head(15)
)


plt.figure(figsize=(10, 7))

plt.barh(

    top_features["Feature"],

    top_features["Importance"]

)

plt.xlabel("Importance")

plt.ylabel("Feature")

plt.title(
    "Top 15 Features - Optimized Random Forest"
)

plt.gca().invert_yaxis()

plt.tight_layout()

plt.show()


# ============================================
# 24. ACTUAL VS PREDICTED
# ============================================

plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    best_predictions
)

plt.xlabel("Actual Grade")

plt.ylabel("Predicted Grade")

plt.title(
    "Actual vs Predicted Grades - Optimized Model"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================
# 25. ACTUAL VS PREDICTED TABLE
# ============================================

prediction_table = pd.DataFrame({

    "Actual": y_test.values,

    "Predicted": best_predictions

})


print("\n================================")
print("ACTUAL VS PREDICTED")
print("================================")

print(
    prediction_table
    .head(10)
    .round(2)
)


# ============================================
# 26. NEW STUDENT
# ============================================

new_student = pd.DataFrame({

    "school": ["GP"],

    "sex": ["F"],

    "age": [17],

    "address": ["U"],

    "famsize": ["GT3"],

    "Pstatus": ["T"],

    "Medu": [4],

    "Fedu": [4],

    "Mjob": ["teacher"],

    "Fjob": ["teacher"],

    "reason": ["course"],

    "guardian": ["mother"],

    "traveltime": [1],

    "studytime": [3],

    "failures": [0],

    "schoolsup": ["yes"],

    "famsup": ["yes"],

    "paid": ["no"],

    "activities": ["yes"],

    "nursery": ["yes"],

    "higher": ["yes"],

    "internet": ["yes"],

    "romantic": ["no"],

    "famrel": [5],

    "freetime": [3],

    "goout": [3],

    "Dalc": [1],

    "Walc": [1],

    "health": [5],

    "absences": [4]

})


# ============================================
# 27. NEW STUDENT PREDICTION
# ============================================

new_prediction = best_model.predict(
    new_student
)


print("\n================================")
print("NEW STUDENT PREDICTION")
print("================================")


print(
    "Predicted Final Grade:",
    round(new_prediction[0], 2),
    "/ 20"
)


# ============================================
# 28. FINAL SUMMARY
# ============================================

print("\n================================")
print("FINAL PROJECT SUMMARY")
print("================================")

print(
    "Dataset:",
    "UCI Student Performance"
)

print(
    "Rows:",
    df.shape[0]
)

print(
    "Features Used:",
    X.shape[1]
)

print(
    "Baseline R2:",
    round(baseline_r2, 3)
)

print(
    "Optimized R2:",
    round(best_r2, 3)
)

print(
    "Baseline MAE:",
    round(baseline_mae, 3)
)

print(
    "Optimized MAE:",
    round(best_mae, 3)
)

print(
    "Baseline RMSE:",
    round(baseline_rmse, 3)
)

print(
    "Optimized RMSE:",
    round(best_rmse, 3)
)

print(
    "Best Parameters:",
    grid_search.best_params_
)

print(
    "New Student Prediction:",
    round(new_prediction[0], 2),
    "/ 20"
)


print("\n================================")
print("PROJECT FINISHED")
print("================================")
# =========================
# 19. SAVE FINAL MODEL
# =========================

os.makedirs("models", exist_ok=True)

joblib.dump(
    baseline_model,
    "models/student_performance_model.pkl"
)

print("\nModel saved successfully!")
print("Saved to: models/student_performance_model.pkl")