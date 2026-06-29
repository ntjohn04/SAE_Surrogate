import re
from attr import dataclass
import requests
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable
import pandas as pd
from pathlib import Path

DEFAULT_RELEASE = "gemma-scope-2b-pt-res"
DEFAULT_SAE_ID = "layer_20/width_16k/average_l0_71"
NEURONPEDIA_FEATURE_API = "https://www.neuronpedia.org/api/feature"

@dataclass(frozen=True)
class FeatureHit:
    feature: int
    activation: float
    token_index: int | None = None
    token: str | None = None

@dataclass(frozen=True)
class FeatureInfo:
    description: str | None
    url: str | None
    error: str | None = None

def neuronpedia_source(release: str, sae_id: str) -> tuple[str, str] | None:
    if release != "gemma-scope-2b-pt-res":
        return None
    match = re.search(r"layer_(\d+)/width_(\d+)k/", sae_id)
    if not match:
        return None
    layer, width = match.groups()
    return "gemma-2-2b", f"{layer}-gemmascope-res-{width}k"

def neuronpedia_url(release: str, sae_id: str, feature: int) -> str | None:
    source = neuronpedia_source(release, sae_id)
    if source is None:
        return None
    model_id, source_id = source
    return f"https://www.neuronpedia.org/{model_id}/{source_id}/{feature}"

def neuronpedia_api_url(release: str, sae_id: str, feature: int) -> str | None:
    source = neuronpedia_source(release, sae_id)
    if source is None:
        return None
    model_id, source_id = source
    return f"{NEURONPEDIA_FEATURE_API}/{model_id}/{source_id}/{feature}"

def clean_description(description: str | None) -> str | None:
    if not description:
        return None
    return " ".join(description.split())

def explanation_sort_key(explanation: dict) -> tuple[int, int]:
    type_name = str(explanation.get("typeName", ""))
    model_name = str(explanation.get("explanationModelName", ""))
    scores = explanation.get("scores") or []

    score_bonus = 0
    for score in scores:
        value = score.get("value") if isinstance(score, dict) else None
        if isinstance(value, (int, float)):
            score_bonus = max(score_bonus, int(value))

    preferred_type = 2 if "np_max-act-logits" in type_name else 0
    preferred_model = 1 if "gemini" in model_name.lower() or "gpt-4" in model_name.lower() else 0
    return preferred_type + preferred_model, score_bonus


def best_feature_description(feature_json: dict) -> str | None:
    explanations = feature_json.get("explanations") or []
    if explanations:
        best = max(explanations, key=explanation_sort_key)
        description = clean_description(best.get("description"))
        if description:
            return description

    vector_label = clean_description(feature_json.get("vectorLabel"))
    if vector_label:
        return vector_label

    return None

def fetch_feature_info(release: str, sae_id: str, feature: int, timeout_s: float = 10.0) -> FeatureInfo:
    url = neuronpedia_url(release, sae_id, feature)
    api_url = neuronpedia_api_url(release, sae_id, feature)
    if api_url is None:
        return FeatureInfo(description=None, url=url, error="unsupported Neuronpedia source")

    try:
        response = requests.get(api_url, timeout=timeout_s)
        response.raise_for_status()
        description = best_feature_description(response.json())
        return FeatureInfo(description=description, url=url)
    except Exception as exc:
        return FeatureInfo(description=None, url=url, error=str(exc))

def fetch_feature_infos(
    release: str,
    sae_id: str,
    features: Iterable[int],
    enabled: bool,
    max_workers: int = 8,
) -> dict[int, FeatureInfo]:
    unique_features = sorted(set(features))
    if not enabled:
        return {
            feature: FeatureInfo(description=None, url=neuronpedia_url(release, sae_id, feature))
            for feature in unique_features
        }

    infos: dict[int, FeatureInfo] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_feature_info, release, sae_id, feature): feature
            for feature in unique_features
        }
        for future in as_completed(futures):
            feature = futures[future]
            try:
                infos[feature] = future.result()
            except Exception as exc:
                infos[feature] = FeatureInfo(
                    description=None,
                    url=neuronpedia_url(release, sae_id, feature),
                    error=str(exc),
                )
    return infos

def visible_token(token: str, max_len: int = 32) -> str:
    text = repr(token)
    if len(text) <= max_len:
        return text
    return text[: max_len - 4] + "...'"


def top_features_for_vector(
    values: torch.Tensor,
    top_k: int,
    token_index: int | None = None,
    token: str | None = None,
) -> list[FeatureHit]:
    values = values.detach().float().cpu()
    active = torch.nonzero(values > 0, as_tuple=False).flatten()
    if active.numel() == 0:
        return []

    active_values = values[active]
    k = min(top_k, active_values.numel())
    top_values, top_positions = torch.topk(active_values, k=k)
    top_indices = active[top_positions]

    return [
        FeatureHit(
            feature=int(feature.item()),
            activation=float(value.item()),
            token_index=token_index,
            token=token,
        )
        for feature, value in zip(top_indices, top_values, strict=True)
    ]

def top_features_across_prompt(
    feature_acts: torch.Tensor,
    tokens: list[str],
    top_k: int,
) -> list[FeatureHit]:
    acts = feature_acts[0].detach().float().cpu()
    max_values, max_token_indices = acts.max(dim=0)
    hits = top_features_for_vector(max_values, top_k)
    return [
        FeatureHit(
            feature=hit.feature,
            activation=hit.activation,
            token_index=int(max_token_indices[hit.feature].item()),
            token=tokens[int(max_token_indices[hit.feature].item())],
        )
        for hit in hits
    ]

def print_hits(
    title: str,
    hits: Iterable[FeatureHit],
    release: str,
    sae_id: str,
    include_links: bool,
    feature_infos: dict[int, FeatureInfo] | None = None,
    show_descriptions: bool = True,
) -> None:
    print(f"\n{title}")
    hits = list(hits)
    if not hits:
        print("  No active features.")
        return

    for rank, hit in enumerate(hits, start=1):
        token_part = ""
        if hit.token_index is not None and hit.token is not None:
            token_part = f" token={hit.token_index} {visible_token(hit.token)}"
        print(
            f"  {rank:>2}. feature={hit.feature:<7} "
            f"activation={hit.activation:>9.4f}{token_part}"
        )
        info = feature_infos.get(hit.feature) if feature_infos else None
        if include_links:
            url = info.url if info else neuronpedia_url(release, sae_id, hit.feature)
            if url:
                print(f"      link: {url}")
        if not show_descriptions:
            continue
        if info and info.description:
            print(f"      represents: {info.description}")
        elif info and info.error:
            print(f"      represents: unavailable ({info.error})")
        elif feature_infos is not None:
            print("      represents: no Neuronpedia explanation found")

def build_feature_catalog(
    release: str = DEFAULT_RELEASE,
    sae_id: str = DEFAULT_SAE_ID,
    num_features: int = 16384,
    max_workers: int = 16,
    save_path: str | None = "feature_catalog.csv",
) -> pd.DataFrame:
    """Fetch all feature descriptions and links from Neuronpedia into a DataFrame.
    
    Saves to CSV so subsequent calls just load from disk.
    """
    if save_path and Path(save_path).exists():
        print(f"Loading cached catalog from {save_path}")
        return pd.read_csv(save_path)

    feature_ids = list(range(num_features))
    print(f"Fetching {num_features} features from Neuronpedia (this may take a while)...")
    infos = fetch_feature_infos(release, sae_id, feature_ids, enabled=True, max_workers=max_workers)

    rows = []
    for fid in feature_ids:
        info = infos.get(fid)
        rows.append({
            "feature_id": fid,
            "feature_desc": info.description if info else None,
            "feature_link": info.url if info else neuronpedia_url(release, sae_id, fid),
        })

    df = pd.DataFrame(rows)
    print(f"Catalog: {len(df)} features, {df['feature_desc'].notna().sum()} with descriptions")
    return df