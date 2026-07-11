import yaml
from .reader import read_box_dims
from .strategies.registry import create_clusterer
from .derivers.ssd import SSDPriorDeriver

class _FlowList(list):
    """A list subtype that YAML will serialize in flow style: [a, b, c]"""

def _flow_representer(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

yaml.add_representer(_FlowList, _flow_representer)

def derive_priors(dataset, split, algorithm, num_aspect_ratios, priors_path, db_url):
    boxes = read_box_dims(db_url=db_url, dataset_name=dataset, split=split)
    with open(priors_path) as file:
        priors_yaml = yaml.safe_load(file)
    num_levels = len(priors_yaml["priors"]["strides"])

    strategy = create_clusterer(algorithm)
    return SSDPriorDeriver().derive_priors(
        boxes=boxes,
        strategy=strategy,
        params={"dataset": dataset, "num_aspect_ratios": num_aspect_ratios, "num_levels": num_levels},
    )
    
def export_priors(result, priors_path, out_path):
    with open(priors_path) as file:
        priors_yaml = yaml.safe_load(file)
        
    p = priors_yaml["priors"]
    num_levels = len(p["strides"])
    p["image_size"]    = _FlowList(p["image_size"])
    p["strides"]       = _FlowList(p["strides"])
    p["aspect_ratios"] = [_FlowList(result.aspect_ratios) for _ in range(num_levels)]
    p["min_scale"]     = round(result.min_scale, 4)
    p["max_scale"]     = round(result.max_scale, 4)
    p["variances"]     = _FlowList(p["variances"])

    with open(out_path, "w") as file:
        yaml.dump(priors_yaml, file, sort_keys=False, allow_unicode=True)