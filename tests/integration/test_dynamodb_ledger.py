import pytest
import boto3

from infrastructure.dynamodb_ledger import ExperimentLedger

TABLE_NAME = "ml-experiment-ledger"
REGION     = "us-east-1"
EXP_ID     = "test_exp"
FP         = "aabbccddee11"
TIMESTAMP  = "20260222_000000"


@pytest.fixture(scope="module")
def ledger():
    return ExperimentLedger(table_name=TABLE_NAME, region=REGION)


@pytest.fixture(scope="module")
def seeded_item(ledger):
    raw_table = boto3.resource("dynamodb", region_name=REGION).Table(TABLE_NAME)
    raw_table.put_item(Item={
        "experiment_id": EXP_ID,
        "fingerprint":   FP,
        "status":        "pending",
        "priority":      50,
    })
    yield
    raw_table.delete_item(Key={"experiment_id": EXP_ID, "fingerprint": FP})


@pytest.mark.integration
def test_get_experiment_state_returns_pending(ledger, seeded_item):
    state = ledger.get_experiment_state(EXP_ID, FP)
    assert state is not None
    assert state["status"] == "pending"


@pytest.mark.integration
def test_claim_experiment_transitions_to_running(ledger, seeded_item):
    claimed = ledger.claim_experiment(EXP_ID, FP, TIMESTAMP, instance_id="i-test001")
    assert claimed is True
    state = ledger.get_experiment_state(EXP_ID, FP)
    assert state["status"] == "running"


@pytest.mark.integration
def test_second_claim_rejected(ledger, seeded_item):
    claimed_again = ledger.claim_experiment(EXP_ID, FP, TIMESTAMP, instance_id="i-test002")
    assert claimed_again is False


@pytest.mark.integration
def test_update_checkpoint_pointer(ledger, seeded_item):
    ledger.update_checkpoint_pointer(EXP_ID, FP, "s3://akhilesh-ml-checkpoints/test/ckpt", step=500)
    state = ledger.get_experiment_state(EXP_ID, FP)
    assert state["checkpoint_s3_path"] == "s3://akhilesh-ml-checkpoints/test/ckpt"


@pytest.mark.integration
def test_mark_success_transitions_to_success(ledger, seeded_item):
    ledger.mark_success(
        EXP_ID, FP,
        checkpoint_s3_path="s3://akhilesh-ml-checkpoints/test/ckpt",
        artifact_s3_path="s3://akhilesh-ml-artifacts/test/weights",
        best_epoch=20,
        total_steps=5000,
        best_metric=0.742,
    )
    state = ledger.get_experiment_state(EXP_ID, FP)
    assert state["status"] == "success"


@pytest.mark.integration
def test_mark_success_on_non_running_item_does_not_raise(ledger, seeded_item):
    # Item is already 'success' — ConditionalCheckFailed should be swallowed, not raised
    ledger.mark_success(EXP_ID, FP, "", "", 0, 0, 0.0)
