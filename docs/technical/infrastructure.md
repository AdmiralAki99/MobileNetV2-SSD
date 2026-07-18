# Infrastructure: DynamoDB Ledger & S3 Sync

`src/infrastructure/{dynamodb_ledger,s3_sync,util}.py`

These are the pieces that connect a training run on an ephemeral EC2 instance back to
the ml-platform's registry (see [../ARCHITECTURE.md §5](../ARCHITECTURE.md) — "Registry
(cloud, machine-independent)"). Neither is required for local training — `train.py`
degrades gracefully to "no ledger / no S3 sync" if unconfigured (see below).

## Experiment ledger (`dynamodb_ledger.py`)

`ExperimentLedger` wraps one DynamoDB table, keyed by `(experiment_id, fingerprint)`.
This is the **single source of truth for run status** the platform's dashboard reads.

State machine (each transition is a conditional `update_item`, atomic against races):

```
register_experiment()  → pending      (put_item, fails silently if item already exists)
claim_experiment()     → pending|failed → running   (an EC2 instance claims the job; ConditionExpression enforces only one claimer wins)
update_checkpoint_pointer()            (running, called repeatedly — just updates checkpoint_s3_path + total_steps)
mark_success()         → running → success            (best_metric, best_epoch, checkpoint/artifact S3 paths)
mark_failure()         → running → failed              (failure_reason recorded)
reset_failed()         → failed → pending              (manual retry hook — scans all items, resets matching experiment_id)
```

Every write is guarded by a `ConditionExpression` on the current `status` — e.g.
`claim_experiment` only succeeds if status is currently `pending` or `failed`; if two
EC2 instances race to claim the same experiment, exactly one `update_item` succeeds and
the other gets `ConditionalCheckFailedException` (caught, returns `False`). This is the
"cheap atomic claim without a distributed lock" pattern DynamoDB conditional writes are
good for — no separate locking service needed.

`get_ec2_instance_id()` — reads the instance's own ID via the **IMDSv2** metadata
endpoint (`http://169.254.169.254/...`, token-based, 2s timeout) — returns `None`
outside EC2 (e.g. local dev) rather than raising, since the ledger update should degrade
gracefully rather than crash training on a laptop run.

`build_dynamodb_ledger(config, logger)` — returns `None` (not an exception) if
`config["infrastructure"]["dynamodb_table"]` is unset, or if the `boto3` client fails to
initialize — this is why `training/engine.py::fit`'s ledger-writing code is always
guarded by `_ledger_available()` (checks `DYNAMODB_TABLE`/`EXPERIMENT_ID` env vars) —
the ledger is opportunistic infrastructure, never a hard training dependency.

Note: `training/engine.py` and `cli/train.py` actually call a module-level
`ledger_writer.write_status`/`write_metric`/`write_checkpoint` (imported as bare
`import ledger_writer`) rather than instantiating `ExperimentLedger` directly in the hot
path — that's a thin wrapper module (not covered in this pass) that presumably
constructs an `ExperimentLedger` from env vars once. If tracing ledger writes from
`engine.py`, start there, not in this file.

## S3 sync (`s3_sync.py`, `util.py`)

`S3SyncClient` manages **three separate S3 buckets/prefixes** by role, all optional
except `checkpoint_bucket`:

- **checkpoint_bucket** (required) — rolling `last`/`best` checkpoints, TensorBoard
  events, `training.log`, `metric_history.json` — everything needed to resume or inspect
  an in-progress run. `upload_training_artifacts(log_dir, run_root)` is the one-shot
  "sync everything training-related" call used after every epoch/best-checkpoint save
  in `training/engine.py::fit`.
- **artifact_bucket** (optional) — final published outputs: `weights/` dir,
  `training_summary.json`. `upload_final_artifacts` — called once, at the very end of
  `cli/train.py::train()`, distinct from the per-epoch checkpoint sync above.
- **dataset_bucket** (optional) — parsed but not exercised by any `upload_*`/`download_*`
  method in this file — likely a placeholder for a dataset-fetch path implemented
  elsewhere (or a TODO).

All upload paths are **best-effort**: every method catches exceptions and logs a
warning rather than raising — an S3 outage should degrade training to "local-only
checkpoints," not kill the run. `ProgressBar` (a `Callback` for `boto3`'s
`download_file`) renders a text progress bar for large checkpoint downloads.

`util.py` — a near-duplicate of `S3SyncClient.upload_training_artifacts`/
`upload_final_artifacts` but as free functions taking an explicit `s3_client` argument
instead of methods — this looks like an older/alternate API surface for the same upload
logic (both are actively imported: `cli/train.py` uses `s3_client.upload_final_artifacts`
directly on the `S3SyncClient` instance, while `util.download_checkpoint_from_s3` is the
one actually used, for `--run_from s3://...` resume). If you're modifying upload
behavior, check both files for the logic you're changing — they can drift out of sync.

`download_checkpoint_from_s3(s3_client, s3_checkpoint_prefix, checkpoint_step=None)` —
lists remote keys first (cheap) to discover which checkpoint steps actually exist
without downloading everything, picks the requested step (or latest if unavailable/
unspecified), then downloads only files matching that step's prefix (`ckpt-<N>.*`) plus
the `checkpoint` metadata file — avoids pulling the entire `keep_last_k` checkpoint
history over the network just to resume from one step.

## Where this plugs into training

`cli/train.py::initialize_run_settings` builds `s3_sync_client` once via `build_s3_sync`
and threads it through the whole run; `training/engine.py::fit` checks `_ledger_available()`
via env vars (`DYNAMODB_TABLE`, `EXPERIMENT_ID`) before every ledger write. Both
infra pieces are env/config-gated so the exact same code path runs identically whether
you're on a laptop with nothing configured or an EC2 spot instance wired into the full
platform (see [../ARCHITECTURE.md](../ARCHITECTURE.md) for why: "the dashboard never
touches a machine... the registry is the source of truth").
