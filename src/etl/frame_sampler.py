from dataclasses import dataclass
from typing import Any
import tempfile
import boto3
import numpy as np
import cv2 as cv


@dataclass
class SampledFrame:
    frame_index: int
    timestamp_s: float
    image: np.ndarray
    scene_change_score: float
    width: int
    height: int


class FrameSampler:
    def __init__(self, config: dict[str, Any]):
        self.stride_frames = config.get("sampling", {}).get("stride_frames", 30)
        self.scene_change_threshold = config.get("sampling", {}).get("scene_change_threshold", 0.35)
        self.max_frames_per_video = config.get("sampling", {}).get("max_frames_per_video", 100)

    def sample(self, video_path: str):

        if video_path.startswith("s3://"):
            path_without_prefix = video_path[5:]
            bucket, key = path_without_prefix.split("/", 1)
            temp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            boto3.client("s3").download_file(bucket, key, temp.name)
            video_path = temp.name

        video = cv.VideoCapture(video_path)

        fps = video.get(cv.CAP_PROP_FPS)

        total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))

        if fps == 0 or total_frames == 0:
            raise RuntimeError(f"Could not read video (fps={fps}, frames={total_frames}): {video_path}")

        width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))

        duration_seconds = total_frames / fps

        previous_hist = None
        score = 0.0

        sampled_frames = []

        for frame_index in range(0, total_frames, self.stride_frames):
            video.set(cv.CAP_PROP_POS_FRAMES, frame_index)

            ret, frame = video.read()

            if not ret:
                continue

            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            grey_frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

            hist = cv.calcHist([grey_frame], [0], None, [256], [0, 256])
            cv.normalize(hist, hist, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

            if previous_hist is None:
                score = 1.0
            else:
                score = cv.compareHist(hist, previous_hist, cv.HISTCMP_BHATTACHARYYA)

            if score > self.scene_change_threshold:
                sampled_frames.append(
                    SampledFrame(
                        frame_index=frame_index,
                        timestamp_s=(frame_index / total_frames) * duration_seconds,
                        image=frame,
                        scene_change_score=score,
                        width=width,
                        height=height,
                    )
                )

            if len(sampled_frames) > self.max_frames_per_video:
                break

            previous_hist = hist

        video.release()

        return sampled_frames, {
            "duration": duration_seconds,
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": total_frames,
        }
