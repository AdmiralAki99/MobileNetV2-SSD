import argparse
import json

from ....config import ETL_DB_URL
from .runner import derive_priors, export_priors

def parse_args():
    parser = argparse.ArgumentParser(description="Derive SSD priors from dataset box stats via clustering")
    parser.add_argument("--dataset", required=True, type=str, help="Dataset name in the ledger (e.g. vis_drone)")
    parser.add_argument("--split", required=True, type=str, help="Split to cluster on (e.g. train)")
    parser.add_argument("--algorithm", default="kmeans", type=str, help="Clustering algorithm")
    parser.add_argument("--num-aspect-ratios", default=3, type=int, help="Number of aspect-ratio clusters")
    parser.add_argument("--priors", required=True, type=str, help="Path to the SSD priors YAML (template)")
    parser.add_argument("--out", default=None, type=str, help="Output YAML path (defaults to --priors, in place)")
    parser.add_argument("--json", action="store_true", help="Print result as JSON to stdout")
    
    args = parser.parse_args()
    
    return {
        "dataset": args.dataset,
        "split": args.split,
        "algorithm": args.algorithm,
        "num_aspect_ratios": args.num_aspect_ratios,
        "priors_path": args.priors,
        "out_path": args.out,
        "json": args.json,
    }
    
def execute_clustering():
    args = parse_args()
    result = derive_priors(
        args["dataset"], args["split"], args["algorithm"],
        args["num_aspect_ratios"], args["priors_path"], ETL_DB_URL,   # ← priors_path
    )
    if args["out_path"]:                                              # ← only writes if --out given
        export_priors(result, args["priors_path"], args["out_path"])
    if args["json"]:
        print(json.dumps(result.to_dict()))
    else:
        print(f"min_scale={result.min_scale:.4f} max_scale={result.max_scale:.4f} "
              f"ratios={result.aspect_ratios} fitness={result.fitness}")
    
    
if __name__ == "__main__":
    execute_clustering()
