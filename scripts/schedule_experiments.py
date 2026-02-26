from mobilenetv2ssd.core.config import load_config
from cli.train import compute_fingerprint
from infrastructure.dynamodb_ledger import ExperimentLedger

from pathlib import Path
from datetime import datetime, timezone
import argparse
import re

def load_experiments(config_root: str, git_commit: str | None = None):
    config_path = Path(config_root)
    experiment_path = config_path / "experiments"
    
    experiments = []
    
    # Now need to iterate over them 
    for path in experiment_path.glob("*.yaml"):
        if re.match(r'^_',path.name):
            continue
        
        # Need only the stuff that does not start with '_' to stop template from being detected
        exp_config = load_config(path, config_root= config_root)
        
        # Now creating a fingerprint
        exp_fingerprint = compute_fingerprint(exp_config, git_commit= git_commit)
        
        # Now need to create the dict
        experiments.append({
            'config': exp_config,
            'yaml_path': str(path),
            'fingerprint': exp_fingerprint,
            'id': exp_config.get('experiment',{}).get('id','exp'),
            'priority': exp_config.get('experiment',{}).get('priority',0),
            'depends_on':  exp_config.get('experiment',{}).get('depends_on',[]),
        })
        
    return experiments
        
def topological_sort(experiments: list[dict]):
    # TODO: Add DAG Scheduling using Kahn's algorithm or something similar for later
    
    return sorted(experiments, key=lambda x: -x['priority'])

def register_experiments(ledger: ExperimentLedger | None, experiments: list[dict], dry_run: bool = False):
    items = []
    for experiment in experiments:
        item = {
            'experiment_id':experiment['id'],
            'fingerprint': experiment['fingerprint'].short,
            'fingerprint_hex': experiment['fingerprint'].hex,
            'status': 'pending',
            'priority': experiment['priority'],
            'depends_on': experiment['depends_on'],
            'experiment_config_path': experiment['yaml_path'],
            "instance_type": experiment['config'].get('infrastructure',{}).get('instance_type', 'NA'),
            'registered_at': datetime.now(timezone.utc).isoformat()
        }
        
        items.append(item)
        
    if dry_run:
        print(items)
    else:
        # Need a ledger
        if not ledger:
            raise ValueError("Ledger is missing!")
        
        # Log it
        for item in items:
            if not ledger.register_experiment(item= item):
                print(f"Experiment: {item['experiment_id']}, Fingerprint: {item['fingerprint']} already exists")
        

def list_experiments_table(ledger: ExperimentLedger):
    items = ledger.list_experiments()
    if not items:
        print("Ledger is empty.")
        return

    items = sorted(items, key=lambda x: (x.get('experiment_id', ''), str(x.get('fingerprint', ''))))

    header = f"{'ID':<10} {'FINGERPRINT':<14} {'STATUS':<10} {'PRIORITY':<10} {'STEPS':<8} {'METRIC':<10} {'INSTANCE'}"
    print(header)
    print('-' * len(header))

    for item in items:
        metric = item.get('best_metric')
        metric_str = f"{float(metric):.4f}" if metric is not None else '-'
        print(f"{item.get('experiment_id',''):<10} {item.get('fingerprint',''):<14} {item.get('status',''):<10} {str(item.get('priority','-')):<10} {str(item.get('total_steps','-')):<8} {metric_str:<10} {item.get('ec2_instance','-')}")
        if item.get('failure_reason'):
            print(f"  failure: {item['failure_reason']}")


def parse_args():
    parser = argparse.ArgumentParser(description="Schedule experiments into the DynamoDB ledger.")
    parser.add_argument('--config_root', type=str, default=None, help='Path to configs root. Defaults to <project_root>/configs.')
    parser.add_argument('--table_name', type=str, default='ml-experiment-ledger', help='DynamoDB table name.')
    parser.add_argument('--region', type=str, default='us-east-1', help='AWS region.')
    parser.add_argument('--git_commit', type=str, default=None, help='Git commit hash (must match what train.py will use).')
    parser.add_argument('--dry_run', action='store_true', help='Preview registrations without writing to DynamoDB.')
    parser.add_argument('--list', action='store_true', help='Print current table state and exit.')
    parser.add_argument('--reset_failed', type=str, default=None, metavar='EXPERIMENT_ID', help='Reset all failed runs for an experiment_id to pending.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config_root = args.config_root or 'configs'

    if args.list:
        ledger = ExperimentLedger(table_name=args.table_name, region=args.region)
        list_experiments_table(ledger)
    elif args.reset_failed:
        ledger = ExperimentLedger(table_name=args.table_name, region=args.region)
        n = ledger.reset_failed(args.reset_failed)
        print(f"Reset {n} failed run(s) for '{args.reset_failed}' to pending.")
    else:
        experiments = load_experiments(config_root, git_commit=args.git_commit)
        ordered = topological_sort(experiments)
        ledger = None if args.dry_run else ExperimentLedger(table_name=args.table_name, region=args.region)
        register_experiments(ledger, ordered, dry_run=args.dry_run)