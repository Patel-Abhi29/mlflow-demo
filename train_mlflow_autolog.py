import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Diabetes_Regression_Comparison")


data = load_diabetes()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
}

best_mse = float("inf")
best_run_id = None
best_model_name = None


for model_name, model in models.items():

    with mlflow.start_run(run_name=model_name):

        # Train
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)

        print(f"{model_name} MSE: {mse:.4f}")

        # Log params
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("dataset", "Diabetes")

        # Log metric
        mlflow.log_metric("mse", mse)

        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )

        # Track best model
        if mse < best_mse:
            best_mse = mse
            best_run_id = mlflow.active_run().info.run_id
            best_model_name = model_name


# Register BEST model automatically

model_uri = f"runs:/{best_run_id}/model"
registered_model_name = "Best_Diabetes_Model"

mlflow.register_model(model_uri, registered_model_name)

print("\n BEST MODEL REGISTERED")
print(f"Best Model: {best_model_name}")
print(f"Best MSE: {best_mse:.4f}")
print(f"Registered as: {registered_model_name}")
print("\nOpen MLflow UI at http://127.0.0.1:5000")
