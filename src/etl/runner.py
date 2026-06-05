from ray.job_submission import JobSubmissionClient
import ray
from .pipeline import ETLWorker


def run_etl(config: dict, video_paths: list[str], config_path: str):
    mode = config.get("ray", {}).get("mode", "local")
    if mode == "local":
        ray.init(ignore_reinit_error=True)
        num_workers = config.get("ray", {}).get("num_workers", 4)

        workers = [ETLWorker.remote(config) for _ in range(num_workers)]

        futures = []
        for i, path in enumerate(video_paths):
            worker = workers[i % num_workers]
            futures.append(worker.process_video.remote(path))

        ray.get(futures)
        return {"videos_processed": len(video_paths)}
    else:
        # Cloud option
        return _run_cloud(config=config, video_paths=video_paths, config_path=config_path)


def _run_cloud(config: dict, video_paths: list[str], config_path: str):
    client = _get_client(config=config)
    job_id = client.submit_job(
        entrypoint=f"python -m src.cli.etl --config {config_path} --videos {' '.join(video_paths)}"
    )

    return {"job_id": job_id, "videos_submitted": len(video_paths)}


def _get_client(config: dict):
    dashboard_url = config.get("ray", {}).get("dashboard_url", "http://localhost:8265")
    client = JobSubmissionClient(dashboard_url)
    return client
