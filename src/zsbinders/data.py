"""Feature table construction from the Anthropic claude-protein-binder-design dataset."""

from __future__ import annotations

import pandas as pd
from datasets import load_dataset

HF_DATASET = "Anthropic/claude-protein-binder-design"

LABEL_COLUMNS = {"adaptyv": "adaptyv_binding", "twist": "twist_binding"}

DESIGN_COLUMNS = [
    "uuid",
    "target",
    "binder_length",
    "generator",
    "sequence_design_method",
    "epitope_n_residues",
]

# Interface-quality and confidence metrics emitted per cofolding run.
COFOLD_METRICS = [
    "sc_dockq",
    "dockq_fnat",
    "dockq_irms",
    "dockq_lrms",
    "ipsae_min",
    "ipsae_max",
    "ipsae_binder_to_target",
    "ipsae_target_to_binder",
    "iptm_pae",
    "ptm_pae",
    "lis",
    "plddt_binder",
    "plddt_target",
    "pae_interface_mean",
    "pae_interface_min",
    "n_interface_contacts",
]

AGGREGATIONS = ["min", "max", "mean", "std", "median"]

COFOLDING_MODEL = "boltz2"


def load_design_summary(label: str) -> pd.DataFrame:
    """Design metadata plus the wet-lab label, restricted to binder/non_binder rows."""
    label_column = LABEL_COLUMNS[label]
    df = (
        load_dataset(HF_DATASET, "design_summary", split="full")
        .to_pandas()
        .loc[:, DESIGN_COLUMNS + [label_column]]
        .dropna()
        .rename(columns={label_column: "y_true"})
    )
    return df[df["y_true"].isin(["non_binder", "binder"])]


def load_cofold_features(cofolding_model: str = COFOLDING_MODEL) -> pd.DataFrame:
    """Per-design summary statistics over the repeated cofolding runs.

    Each design is folded several times with different seeds, so the spread across runs
    carries signal that any single run does not.
    """
    df = load_dataset(HF_DATASET, "insilico_cofold_predictions", split="full").to_pandas()
    df = df[df["cofolding_model"] == cofolding_model]
    grouped = df[["uuid"] + COFOLD_METRICS].groupby("uuid").agg(AGGREGATIONS)
    grouped.columns = ["_".join(col) for col in grouped.columns]
    return grouped.reset_index()


def build_feature_table(label: str, cofolding_model: str = COFOLDING_MODEL) -> pd.DataFrame:
    """One row per design: aggregated cofolding metrics, design metadata, and `y_true`."""
    features = load_cofold_features(cofolding_model)
    designs = load_design_summary(label)
    return features.merge(designs, on="uuid", how="inner").dropna().reset_index(drop=True)


def stratified_sample(df: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    """Class-balanced sample, shuffled so progress logs are not ordered by class."""
    per_class = min(size // 2, *df["y_true"].value_counts())
    return (
        df.groupby("y_true", group_keys=False)
        .sample(n=per_class, random_state=seed)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric model inputs: everything but the identifier, the label, and scoring output."""
    excluded = {"uuid", "y_true", "y_pred", "p_binder", "error"}
    return [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
