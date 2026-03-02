import mlflow


def test_log_and_query(tmp_path):
    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("smoke-test")

    with mlflow.start_run():
        mlflow.log_param("batch_size", 32)
        mlflow.log_metric("throughput", 1234.5)

    runs = mlflow.search_runs(experiment_names=["smoke-test"])
    assert len(runs) == 1
    assert runs.iloc[0]["params.batch_size"] == "32"
    assert runs.iloc[0]["metrics.throughput"] == 1234.5
