const MOCK_MODE = false;

const MOCK_EXPERIMENTS = [
  {
    experiment_id: 'exp002_cloud_run',
    fingerprint: 'a1b2c3d4',
    status: 'success',
    region: 'us-east-1',
    best_metric: 0.766,
    best_epoch: 200,
    total_steps: 45000,
    checkpoint_s3_path: 's3://akhilesh-ml-checkpoints/runs/exp002_cloud_run/best',
    config_filename: 'exp002_cloud_run.yaml',
    completed_at: '2026-03-02T04:37:49Z',
  },
  {
    experiment_id: 'exp003_visdrone_run',
    fingerprint: 'e5f6a7b8',
    status: 'running',
    region: 'ap-southeast-1',
    ec2_instance: 'i-0abc123def456789',
    total_steps: 12500,
    checkpoint_s3_path: 's3://akhilesh-ml-checkpoints/runs/exp003_visdrone_run/best',
    config_filename: 'exp003_visdrone_run.yaml',
    claimed: '2026-05-10T08:00:00Z',
  },
  {
    experiment_id: 'exp004_augment_v2',
    fingerprint: 'c9d0e1f2',
    status: 'pending',
    config_filename: 'exp004_augment_v2.yaml',
  },
  {
    experiment_id: 'exp001_baseline',
    fingerprint: 'g3h4i5j6',
    status: 'failed',
    total_steps: 3200,
    failure_reason: 'CUDA OOM: batch size 32 exceeds GPU memory on p2.xlarge',
    config_filename: 'exp001_baseline.yaml',
  },
];

const MOCK_INSTANCE = {
  message: {
    name: 'running',
    architecture: 'x86_64',
    instance_type: 'p3.2xlarge',
    public_ip: '54.123.45.67',
    launch_time: '2026-05-10T08:00:00+00:00',
    tensorboard_url: 'http://54.123.45.67:6006',
  }
};

const MOCK_ARTIFACTS = {
  'exp002_cloud_run': { saved_model: true,  onnx: true,  priors: true  },
  'exp003_visdrone_run': { saved_model: false, onnx: false, priors: false },
};

const MOCK_ETL_STATS =
{ total_videos: 3, total_frames: 247, total_annotations: 1842,
  class_distribution: [
    { class_name: 'pedestrian', count: 712 },
    { class_name: 'car', count: 541 },
    { class_name: 'bicycle', count: 310 },
    { class_name: 'truck', count: 182 },
    { class_name: 'van', count: 97 },
  ]
}

const MOCK_ETL_FRAMES = {
  1: [
    { id: 101, frame_index: 0,   timestamp_s: 0.0,  scene_change_score: 1.00, annotation_count: 8  },
    { id: 102, frame_index: 30,  timestamp_s: 1.0,  scene_change_score: 0.72, annotation_count: 12 },
    { id: 103, frame_index: 60,  timestamp_s: 2.0,  scene_change_score: 0.61, annotation_count: 6  },
    { id: 104, frame_index: 90,  timestamp_s: 3.0,  scene_change_score: 0.58, annotation_count: 9  },
    { id: 105, frame_index: 120, timestamp_s: 4.0,  scene_change_score: 0.44, annotation_count: 5  },
  ],
  2: [
    { id: 201, frame_index: 0,  timestamp_s: 0.0, scene_change_score: 1.00, annotation_count: 5 },
    { id: 202, frame_index: 30, timestamp_s: 1.0, scene_change_score: 0.68, annotation_count: 7 },
    { id: 203, frame_index: 60, timestamp_s: 2.0, scene_change_score: 0.55, annotation_count: 4 },
  ],
};

const MOCK_ETL_ANNOTATIONS = {
  101: [
    { id: 1001, class_name: 'pedestrian', votes: 3, consensus_score: 0.874, x1: 0.12, y1: 0.35, x2: 0.18, y2: 0.52, model_confidences: { yolov8: 0.91, rtdetr: 0.84, grounding_dino: 0.85 } },
    { id: 1002, class_name: 'car',        votes: 3, consensus_score: 0.812, x1: 0.45, y1: 0.55, x2: 0.68, y2: 0.78, model_confidences: { yolov8: 0.88, rtdetr: 0.79, grounding_dino: 0.76 } },
    { id: 1003, class_name: 'pedestrian', votes: 2, consensus_score: 0.731, x1: 0.72, y1: 0.38, x2: 0.77, y2: 0.55, model_confidences: { yolov8: 0.78, rtdetr: 0.69 } },
    { id: 1004, class_name: 'bicycle',    votes: 2, consensus_score: 0.689, x1: 0.28, y1: 0.60, x2: 0.36, y2: 0.75, model_confidences: { rtdetr: 0.71, grounding_dino: 0.67 } },
    { id: 1005, class_name: 'van',        votes: 3, consensus_score: 0.923, x1: 0.55, y1: 0.42, x2: 0.82, y2: 0.71, model_confidences: { yolov8: 0.95, rtdetr: 0.91, grounding_dino: 0.90 } },
    { id: 1006, class_name: 'truck',      votes: 2, consensus_score: 0.654, x1: 0.05, y1: 0.48, x2: 0.22, y2: 0.68, model_confidences: { yolov8: 0.70, grounding_dino: 0.61 } },
    { id: 1007, class_name: 'pedestrian', votes: 3, consensus_score: 0.796, x1: 0.88, y1: 0.32, x2: 0.93, y2: 0.48, model_confidences: { yolov8: 0.82, rtdetr: 0.76, grounding_dino: 0.79 } },
    { id: 1008, class_name: 'car',        votes: 2, consensus_score: 0.701, x1: 0.33, y1: 0.62, x2: 0.52, y2: 0.82, model_confidences: { yolov8: 0.74, rtdetr: 0.66 } },
  ],
  102: [
    { id: 1009, class_name: 'pedestrian', votes: 3, consensus_score: 0.911, x1: 0.08, y1: 0.28, x2: 0.14, y2: 0.46, model_confidences: { yolov8: 0.94, rtdetr: 0.89, grounding_dino: 0.91 } },
    { id: 1010, class_name: 'car',        votes: 3, consensus_score: 0.856, x1: 0.38, y1: 0.50, x2: 0.63, y2: 0.74, model_confidences: { yolov8: 0.90, rtdetr: 0.83, grounding_dino: 0.82 } },
    { id: 1011, class_name: 'truck',      votes: 3, consensus_score: 0.841, x1: 0.65, y1: 0.45, x2: 0.88, y2: 0.72, model_confidences: { yolov8: 0.87, rtdetr: 0.82, grounding_dino: 0.84 } },
    { id: 1012, class_name: 'pedestrian', votes: 2, consensus_score: 0.714, x1: 0.22, y1: 0.32, x2: 0.28, y2: 0.50, model_confidences: { yolov8: 0.76, grounding_dino: 0.67 } },
    { id: 1013, class_name: 'bicycle',    votes: 2, consensus_score: 0.672, x1: 0.48, y1: 0.62, x2: 0.57, y2: 0.78, model_confidences: { rtdetr: 0.70, grounding_dino: 0.64 } },
    { id: 1014, class_name: 'van',        votes: 3, consensus_score: 0.889, x1: 0.70, y1: 0.30, x2: 0.90, y2: 0.55, model_confidences: { yolov8: 0.92, rtdetr: 0.87, grounding_dino: 0.88 } },
    { id: 1015, class_name: 'car',        votes: 2, consensus_score: 0.698, x1: 0.10, y1: 0.55, x2: 0.30, y2: 0.75, model_confidences: { yolov8: 0.73, rtdetr: 0.67 } },
    { id: 1016, class_name: 'pedestrian', votes: 3, consensus_score: 0.833, x1: 0.42, y1: 0.25, x2: 0.47, y2: 0.42, model_confidences: { yolov8: 0.86, rtdetr: 0.81, grounding_dino: 0.83 } },
    { id: 1017, class_name: 'car',        votes: 2, consensus_score: 0.711, x1: 0.15, y1: 0.60, x2: 0.35, y2: 0.80, model_confidences: { rtdetr: 0.74, grounding_dino: 0.68 } },
    { id: 1018, class_name: 'pedestrian', votes: 2, consensus_score: 0.687, x1: 0.78, y1: 0.34, x2: 0.83, y2: 0.52, model_confidences: { yolov8: 0.71, rtdetr: 0.66 } },
    { id: 1019, class_name: 'bicycle',    votes: 3, consensus_score: 0.752, x1: 0.55, y1: 0.58, x2: 0.64, y2: 0.76, model_confidences: { yolov8: 0.78, rtdetr: 0.72, grounding_dino: 0.75 } },
    { id: 1020, class_name: 'truck',      votes: 2, consensus_score: 0.643, x1: 0.02, y1: 0.52, x2: 0.18, y2: 0.76, model_confidences: { yolov8: 0.68, grounding_dino: 0.61 } },
  ],
  201: [
    { id: 2001, class_name: 'car',        votes: 3, consensus_score: 0.901, x1: 0.20, y1: 0.45, x2: 0.45, y2: 0.70, model_confidences: { yolov8: 0.93, rtdetr: 0.89, grounding_dino: 0.91 } },
    { id: 2002, class_name: 'pedestrian', votes: 2, consensus_score: 0.742, x1: 0.60, y1: 0.30, x2: 0.66, y2: 0.48, model_confidences: { yolov8: 0.79, rtdetr: 0.70 } },
    { id: 2003, class_name: 'bicycle',    votes: 3, consensus_score: 0.811, x1: 0.78, y1: 0.50, x2: 0.86, y2: 0.68, model_confidences: { yolov8: 0.84, rtdetr: 0.80, grounding_dino: 0.81 } },
    { id: 2004, class_name: 'van',        votes: 2, consensus_score: 0.668, x1: 0.05, y1: 0.55, x2: 0.24, y2: 0.78, model_confidences: { rtdetr: 0.70, grounding_dino: 0.64 } },
    { id: 2005, class_name: 'car',        votes: 2, consensus_score: 0.695, x1: 0.48, y1: 0.58, x2: 0.68, y2: 0.80, model_confidences: { yolov8: 0.72, grounding_dino: 0.67 } },
  ],
};

const MOCK_ETL_VIDEOS = [
  { id: 1, filename: 'DJI_0001.mp4',     duration: 62.4,  fps: 30.0, width: 3840, height: 2160, status: 'completed',  frames: 104, annotations: 891 },
  { id: 2, filename: 'DJI_0002.mp4',     duration: 48.1,  fps: 30.0, width: 3840, height: 2160, status: 'completed',  frames:  83, annotations: 621 },
  { id: 3, filename: 'DJI_0003_raw.mp4', duration: 115.7, fps: 25.0, width: 1920, height: 1080, status: 'processing', frames:  60, annotations: 330 },
];

const MOCK_CICD = {
  current_run: {
    id: 'run_20260523_a1b2',
    branch: 'dev',
    commit: '84514d5',
    commit_message: 'Adds multi-model object detection and consensus ETL components',
    trigger: 'push',
    started_at: '2026-05-23T14:22:10Z',
    status: 'running',
    stages: [
      { id: 'lint',        name: 'Lint & Types',   status: 'success', duration: 18.4 },
      { id: 'unit',        name: 'Unit Tests',      status: 'success', duration: 42.1 },
      { id: 'integration', name: 'Integration',     status: 'running', duration: null },
      { id: 'eval',        name: 'Model Eval',      status: 'pending', duration: null },
      { id: 'build',       name: 'Docker Build',    status: 'pending', duration: null },
      { id: 'staging',     name: 'Deploy Staging',  status: 'pending', duration: null },
      { id: 'prod',        name: 'Deploy Prod',     status: 'pending', duration: null },
    ],
  },
  recent_runs: [
    { id: 'run_20260522_e5f6', branch: 'main', commit: '561e158', status: 'success', duration: 892, started_at: '2026-05-22T09:15:00Z' },
    { id: 'run_20260521_c9d0', branch: 'main', commit: '0478399', status: 'failed',  duration: 234, started_at: '2026-05-21T16:40:00Z' },
    { id: 'run_20260520_g3h4', branch: 'main', commit: 'c4bf5e1', status: 'success', duration: 910, started_at: '2026-05-20T11:22:00Z' },
  ],
};

const MOCK_RELEASES = [
  {
    version: 'v1.2.0',
    model: 'mobilenetv2_ssd_voc',
    experiment: 'exp002_cloud_run',
    map_score: 0.766,
    released_at: '2026-03-15T10:00:00Z',
    status: 'current',
    targets: ['jetson', 'api'],
    artifacts: { saved_model: true, onnx: true, tensorrt: true },
  },
  {
    version: 'v1.1.0',
    model: 'mobilenetv2_ssd_voc',
    experiment: 'exp001_baseline',
    map_score: 0.721,
    released_at: '2026-01-20T14:30:00Z',
    status: 'deprecated',
    targets: ['api'],
    artifacts: { saved_model: true, onnx: true, tensorrt: false },
  },
  {
    version: 'v1.0.0',
    model: 'mobilenetv2_ssd_voc',
    experiment: 'exp000_initial',
    map_score: 0.682,
    released_at: '2025-11-05T09:00:00Z',
    status: 'archived',
    targets: ['api'],
    artifacts: { saved_model: true, onnx: false, tensorrt: false },
  },
];

const MOCK_AIRFLOW = {
  dag_id: 'etl_pipeline',
  schedule: '0 2 * * *',
  is_paused: false,
  last_run: {
    run_id: 'scheduled__2026-05-23T02:00:00+00:00',
    state: 'success',
    start_date: '2026-05-23T02:00:02Z',
    end_date:   '2026-05-23T02:08:23Z',
  },
  tasks: [
    { task_id: 'provision_ec2', state: 'success', duration: 0.8,   start_date: '2026-05-23T02:00:02Z' },
    { task_id: 'wait_for_ray',  state: 'success', duration: 12.3,  start_date: '2026-05-23T02:00:03Z' },
    { task_id: 'run_etl_job',   state: 'success', duration: 483.2, start_date: '2026-05-23T02:00:16Z' },
    { task_id: 'teardown_ec2',  state: 'success', duration: 1.1,   start_date: '2026-05-23T02:08:19Z' },
    { task_id: 'email_summary', state: 'success', duration: 2.4,   start_date: '2026-05-23T02:08:21Z' },
  ],
};

const MOCK_AIRFLOW_RUNS = [
  { run_id: 'scheduled__2026-05-29T02:00:00+00:00', state: 'success', run_type: 'scheduled', duration: 503, start_date: '2026-05-29T02:00:02Z' },
  { run_id: 'manual__2026-05-28T14:22:00+00:00',    state: 'failed',  run_type: 'manual',    duration: 47,  start_date: '2026-05-28T14:22:00Z' },
  { run_id: 'scheduled__2026-05-28T02:00:00+00:00', state: 'success', run_type: 'scheduled', duration: 491, start_date: '2026-05-28T02:00:01Z' },
];

const MOCK_RAY = {
  status: 'running',
  dashboard_url: 'http://localhost:8265',
  resources: {
    cpu_used: 2.1, cpu_total: 4,
    memory_used_gb: 8.3, memory_total_gb: 16,
  },
  nodes: [
    { id: 'node-001', ip: '127.0.0.1', status: 'alive', cpu_pct: 52, memory_pct: 52 },
  ],
};

const mockDelay = (data, ms = 400) =>
  new Promise(resolve => setTimeout(() => resolve(data), ms));

const API_BASE = '';

async function apiFetch(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, options);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status}: ${text}`);
  }
  return res.json();
}

function apiPost(path, body) {
  return apiFetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

function fetchExperiments() {
  if (MOCK_MODE) return mockDelay(MOCK_EXPERIMENTS);
  return apiFetch('/api/experiments');
}

function fetchInstanceStatus(instanceId) {
  if (MOCK_MODE) return mockDelay(MOCK_INSTANCE);
  return apiFetch(`/api/training/${instanceId}/status`);
}

function fetchArtifacts(experimentId) {
  if (MOCK_MODE) return mockDelay({
    artifat_status: MOCK_ARTIFACTS[experimentId] || { saved_model: false, onnx: false, priors: false }
  });
  return apiFetch(`/api/export/${experimentId}/artifacts`);
}

function fetchJobStatus(jobId) {
  if (MOCK_MODE) return mockDelay({ job_status: { status: 'success' } }, 1500);
  return apiFetch(`/api/export/jobs/${jobId}`);
}

function launchTraining(req) {
  if (MOCK_MODE) return mockDelay({ status: 'ok', message: 'Launched (mock)' }, 800);
  return apiPost('/api/training/launch', req);
}

function stopTraining(req) {
  if (MOCK_MODE) return mockDelay({ status: 'ok', message: 'Stopped (mock)' }, 800);
  return apiPost('/api/training/stop', req);
}

function exportSavedModel(req) {
  if (MOCK_MODE) return mockDelay({ job_id: `export_mock_${Date.now()}`, status: 'running' }, 600);
  return apiPost('/api/export/savedmodel', req);
}

function exportOnnx(req) {
  if (MOCK_MODE) return mockDelay({ job_id: `onnx_mock_${Date.now()}`, status: 'running' }, 600);
  return apiPost('/api/export/onnx', req);
}

function fetchMetrics(experimentId) {
  if (MOCK_MODE) return mockDelay({ experiment_id: experimentId }, 600);
  return apiFetch(`/api/experiments/${experimentId}/metrics`);
}

function fetchEtlFrames(videoId) {
  if (MOCK_MODE) return mockDelay(MOCK_ETL_FRAMES[videoId] || []);
  return apiFetch(`/api/etl/videos/${videoId}/frames`);
}

function fetchEtlAnnotations(frameId) {
  if (MOCK_MODE) return mockDelay(MOCK_ETL_ANNOTATIONS[frameId] || []);
  return apiFetch(`/api/etl/frames/${frameId}/annotations`);
}

function fetchCicd() {
  if (MOCK_MODE) return mockDelay(MOCK_CICD);
  return apiFetch('/api/deploy/cicd');
}

function fetchReleases() {
  if (MOCK_MODE) return mockDelay(MOCK_RELEASES);
  return apiFetch('/api/deploy/releases');
}

function fetchAirflow() {
  if (MOCK_MODE) return mockDelay(MOCK_AIRFLOW);
  return apiFetch('/api/ops/airflow');
}

function fetchAirflowRuns(){ 
  if (MOCK_MODE) return mockDelay(MOCK_AIRFLOW_RUNS); 
  return apiFetch('/api/ops/airflow/runs'); 
}
function fetchAirflowRunTasks(id){ 
  if (MOCK_MODE) return mockDelay(MOCK_AIRFLOW.tasks); 
  return apiFetch(`/api/ops/airflow/runs/${id}/tasks`); 
}

function fetchRay() {
  if (MOCK_MODE) return mockDelay(MOCK_RAY);
  return apiFetch('/api/ops/ray');
}

function fetchEtlStats() {
  if (MOCK_MODE) return mockDelay(MOCK_ETL_STATS);
  return apiFetch('/api/etl/stats');
}

function fetchEtlVideos() {
  if (MOCK_MODE) return mockDelay(MOCK_ETL_VIDEOS);
  return apiFetch('/api/etl/videos');
}

function usePolling(fn, interval = 5000, enabled = true) {
  const [data, setData]       = React.useState(null);
  const [loading, setLoading] = React.useState(false);
  const [error, setError]     = React.useState(null);
  const fnRef = React.useRef(fn);
  fnRef.current = fn;

  const call = React.useCallback(async () => {
    setLoading(true);
    try {
      const result = await fnRef.current();
      setData(result);
      setError(null);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  React.useEffect(() => {
    if (!enabled) return;
    call();
    const id = setInterval(call, interval);
    return () => clearInterval(id);
  }, [enabled, interval, call]);

  return { data, loading, error, refresh: call };
}

Object.assign(window, {
  fetchExperiments, fetchInstanceStatus, fetchArtifacts, fetchJobStatus,
  launchTraining, stopTraining, exportSavedModel, exportOnnx,
  fetchMetrics, usePolling,
  fetchEtlStats, fetchEtlVideos, fetchEtlFrames, fetchEtlAnnotations,
  fetchAirflow, fetchRay,
  fetchCicd, fetchReleases,fetchAirflowRuns, fetchAirflowRunTasks,
});
