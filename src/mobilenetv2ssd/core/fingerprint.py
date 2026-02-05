from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import json, hashlib, math
from typing import Any, Mapping

@dataclass
class Fingerprint:
    algo: str
    schema_version: int
    hex: str
    short: str
    canonical_json: Optional[str] = None

    def s3_prefix(self, namespace: str = "runs") -> str:
        # e.g. runs/<hex>/
        return f"{namespace}/{self.hex}/"

    def lmdb_key(self, namespace: str = "run") -> bytes:
        # e.g. b"run:<hex>"
        return f"{namespace}:{self.hex}".encode("utf-8")
    
    
class Fingerprinter:
    def __init__(self, algo: str = "sha256", short_len: int = 12, schema_version: int = 1, include_canonical_json: bool = False):
        self._algo = algo
        self._short_len = short_len
        self._schema_version = schema_version
        self._include_canonical_json = include_canonical_json
    
    
    def _to_json(self, obj: Any):
        if obj is None or isinstance(obj, (bool, int, str)):
            return obj
        
        if isinstance(obj, float):
            if math.isnan(obj):
                return "NaN"
            if math.isinf(obj):
                return "__Inf__" if obj > 0 else "__-Inf__"
            return float(f"{obj:.17g}")
        
        # Handling the mapping type
        if isinstance(obj, Mapping):
            return {str(k): self._to_json(v) for k, v in sorted(obj.items())}
        
        # Handling iterable types
        if isinstance(obj, (list, tuple)):
            return [self._to_json(v) for v in obj]
        
        # Handling a set type
        if isinstance(obj, set):
            return sorted([self._to_json(v) for v in obj])
        
        if hasattr(obj, "item") and callable(getattr(obj, "item")):
            try:
                return self._to_json(obj.item())
            except Exception:
                pass
        
        return str(obj)
    
    def _canonicalize(self, obj: Mapping[str, Any]):
        if not isinstance(obj, Mapping):
            raise TypeError("Fingerprinter.fingerprint expects a Mapping (dict-like)")
        
        json_obj = self._to_json(obj)
        if not isinstance(json_obj, dict):
            raise TypeError("Fingerprinter.fingerprint expects a Mapping (dict-like)")
        
        json_obj['_schema_version'] = self._schema_version
        
        return json_obj    
    
    def fingerprint(self, obj: Mapping[str, Any]):
        payload = self._canonicalize(obj)
        
        canonical_json = json.dumps(payload, separators=(',', ':'), ensure_ascii=True)
        
        hasher = hashlib.new(self._algo)
        hasher.update(canonical_json.encode('utf-8'))
        hex_digest = hasher.hexdigest()
        
        short_digest = hex_digest[:self._short_len]
        
        return Fingerprint(
            algo = self._algo,
            schema_version = self._schema_version,
            hex = hex_digest,
            short = short_digest,
            canonical_json = canonical_json if self._include_canonical_json else None
        )