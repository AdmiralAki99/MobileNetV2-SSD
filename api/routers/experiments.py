from fastapi import APIRouter, HTTPException
from decimal import Decimal
from ..services.ledger import get_ledger
# Creating the router to use in the api server

router = APIRouter()

def _to_json(item: dict):
    json_dict = {}
    for obj in item:
        if isinstance(item[obj], Decimal):
            json_dict[obj] = float(item[obj])
        elif isinstance(item[obj], set):
            json_dict[obj] = list(item[obj])
        else:
            json_dict[obj] = item[obj]
            
    return json_dict

@router.get("")
def get_experiments():
    # Need to get the ledger and then the experiments from it
    experiment_ledger = get_ledger()
    experiments = experiment_ledger.list_experiments()
    # Now need to make the decimals floats to make the serializable
    return [_to_json(experiment) for experiment in experiments]

@router.post("/{experiment_id}/{fingerprint}/reset")
def reset_experiment(experiment_id: str, fingerprint: str):
    # Need to get the ledger and then the reset from it
    experiment_ledger = get_ledger()
    reset_counter = experiment_ledger.reset_failed(experiment_id= experiment_id)
    
    if reset_counter == 0:
        raise HTTPException(status_code=400, detail="No failed runs to reset")
    else:
        return {
            'status': 200,
            'Message': f"Reset Complete! Took {reset_counter} tries"
        }