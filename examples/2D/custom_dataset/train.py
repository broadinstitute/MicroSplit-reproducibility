import neptune

run = neptune.init_run(
    project="BroadImagingPlatform/jump-qc-microsplit",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0OGVhMGJlZS1kZTRlLTQ2NjAtOWY2Ny00YmE1YmE4MjkyZTYifQ=="  # uses NEPTUNE_API_TOKEN env var
)

# Log parameters
run["parameters/lr"] = 0.001
run["parameters/epochs"] = 10

# Log metrics
for epoch in range(10):
    loss = 1 / (epoch + 1)
    run["train/loss"].append(loss)

run.stop()