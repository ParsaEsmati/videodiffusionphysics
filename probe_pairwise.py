#!/usr/bin/env python3
"""
Pair-wise linear probe: one linear classifier per transformer block on saved
features, with a train/validation split grouped by scenario and per-video and
pair-wise validation accuracy.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from datetime import datetime as dt
from pathlib import Path
from typing import Optional

import lightning as L
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset, Subset

from data_pipeline.probe_dataset import ProbeDataset, build_probe_csv

# Optional: only needed when wandb logging is on (the default; see --no-wandb).
try:
    import wandb
except ImportError:
    wandb = None


DEFAULT_WANDB_PROJECT_NAME = "videophys-probe-pairwise"


# ============================================================================
# Helpers: scenario extraction + pair-wise metric
# ============================================================================


def _scenario_from_path(scene_dir_path: str | Path, mode: str) -> str:
    """Scenario id of a scene: IntPhys ``O1_01_1`` gives ``O1_01``; InfLevel names drop the event code."""
    p = Path(scene_dir_path)
    stem = p.parent.name if p.name.startswith("step_") else p.name
    if mode == "intphys":
        parts = stem.rsplit("_", 1)
        return parts[0] if len(parts) > 1 else stem
    if mode == "inflevel":
        # File naming:
        #   gravity/solidity (5 parts):  view__principle__obj_a__obj_b__code
        #   continuity      (6 parts):  view__continuity__obj_a__obj_b__code__direction
        # InfLevel's official evaluator groups by (camera, cover, obj[, dir]) — i.e.
        # direction matters for continuity. Drop the event-code, keep direction.
        parts = stem.split("__")
        if len(parts) == 6:
            return "__".join([parts[0], parts[1], parts[2], parts[3], parts[5]])
        if len(parts) == 5:
            return "__".join(parts[:4])
        return stem
    if mode == "scene":
        return stem    # each scene is its own scenario (degenerates to per-video metric)
    raise ValueError(f"Unknown scenario_mode: {mode!r}")


def pairwise_accuracy(samples: list[tuple[float, int, int]]) -> Optional[float]:
    """Share of (plausible, impossible) pairs within a scenario where the plausible video scores
    higher; ties count 0.5. Returns ``None`` if there are no pairs."""
    by_scenario: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for s, l, sc in samples:
        by_scenario[sc].append((s, l))

    correct = 0.0
    total = 0
    for items in by_scenario.values():
        pos = [s for s, l in items if l == 1]
        neg = [s for s, l in items if l == 0]
        for p in pos:
            for n in neg:
                total += 1
                if p > n:
                    correct += 1.0
                elif p == n:
                    correct += 0.5
    return correct / total if total > 0 else None


def pairwise_correctness_per_block(
    per_block_scores: list[list[tuple[float, int, int]]],
    scene_dirs: Optional[list[str]] = None,
    scenario_strs: Optional[list[str]] = None,
) -> list[dict]:
    """One row per (plausible, impossible) pair within a scenario, scored 1 / 0.5 / 0 for every block."""
    if not per_block_scores or not per_block_scores[0]:
        return []

    num_blocks = len(per_block_scores)
    labels = [l for _, l, _ in per_block_scores[0]]
    scenarios = [sc for _, _, sc in per_block_scores[0]]

    by_scenario: dict[int, list[int]] = defaultdict(list)
    for i, sc in enumerate(scenarios):
        by_scenario[sc].append(i)

    rows: list[dict] = []
    for sc, idxs in by_scenario.items():
        pos = [i for i in idxs if labels[i] == 1]
        neg = [i for i in idxs if labels[i] == 0]
        for pi in pos:
            for ni in neg:
                row: dict = {
                    "scenario_id": sc,
                    "plaus_video_idx": pi,
                    "impos_video_idx": ni,
                }
                if scenario_strs is not None:
                    row["scenario"] = scenario_strs[pi]
                if scene_dirs is not None:
                    row["plaus_scene_dir"] = scene_dirs[pi]
                    row["impos_scene_dir"] = scene_dirs[ni]
                for k in range(num_blocks):
                    pm = per_block_scores[k][pi][0]
                    nm = per_block_scores[k][ni][0]
                    row[f"block_{k:02d}"] = (
                        1.0 if pm > nm else 0.5 if pm == nm else 0.0
                    )
                rows.append(row)
    return rows


def _stem_of(scene_dir_path) -> str:
    p = Path(scene_dir_path)
    return p.parent.name if p.name.startswith("step_") else p.name


def detect_classes(scene_dirs: list, mode: str) -> list[str]:
    """Class prefixes found in the scene names (IntPhys ``O1``, ``O2``, ``O3``); empty for InfLevel."""
    classes: set[str] = set()
    for p in scene_dirs:
        stem = _stem_of(p)
        if mode == "intphys":
            cls = stem.split("_")[0]
            if cls:
                classes.add(cls)
    return sorted(classes)


# ============================================================================
# Dataset wrapper: ProbeDataset + scenario id
# ============================================================================


class PairwiseProbeDataset(Dataset):
    """A ``ProbeDataset`` whose items also carry a scenario id.

    Args:
        base:      Dataset yielding ``(features, label)``.
        scenarios: Scenario name of each item, in dataset order.
    """

    def __init__(self, base: ProbeDataset, scenarios: list[str]):
        assert len(base) == len(scenarios)
        self.base = base
        self.scenarios = scenarios
        unique = sorted(set(scenarios))
        self._scenario_to_id = {s: i for i, s in enumerate(unique)}
        self.scenario_ids = [self._scenario_to_id[s] for s in scenarios]

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        feat, label = self.base[idx]
        return feat, label, self.scenario_ids[idx]


# ============================================================================
# DataModule: build CSV → ProbeDataset → group-aware split
# ============================================================================


class PairwiseProbeDataModule(L.LightningDataModule):
    """Data for the pair-wise probe, split by scenario.

    All videos of a scenario stay on the same side of the train/validation
    split, so a plausible/impossible pair is never separated.

    Args:
        run_dir:         Run folder to index (ignored if ``csv_path`` is given).
        csv_path:        Existing probe CSV.
        step:            Inversion step to probe.
        legacy:          Features sit directly in each scene folder.
        val_split:       Fraction of the scenarios held out for validation.
        batch_size:      DataLoader batch size.
        num_workers:     DataLoader worker processes.
        scenario_mode:   ``intphys``, ``inflevel`` or ``scene``: how scenes are
                         grouped into scenarios.
        scenario_prefix: Keep only the scenes whose name starts with this.
        seed:            Seed of the split.
    """

    def __init__(
        self,
        run_dir: Optional[str | Path] = None,
        csv_path: Optional[str | Path] = None,
        step: int = 0,
        legacy: bool = False,
        val_split: float = 0.2,
        batch_size: int = 32,
        num_workers: int = 0,
        scenario_mode: str = "intphys",
        scenario_prefix: Optional[str] = None,
        seed: int = 42,
    ):
        super().__init__()
        if csv_path is None and run_dir is None:
            raise ValueError("Provide either csv_path or run_dir")
        self.run_dir = Path(run_dir) if run_dir else None
        self.csv_path = Path(csv_path) if csv_path else None
        self.step = step
        self.legacy = legacy
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.scenario_mode = scenario_mode
        self.scenario_prefix = scenario_prefix
        self.seed = seed

        self._train: Optional[Dataset] = None
        self._val: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.csv_path is not None:
            csv = self.csv_path
        elif self.legacy:
            csv = self.run_dir / "probe_scenes_legacy.csv"
            build_probe_csv(self.run_dir, csv, legacy=True)
        else:
            csv = self.run_dir / f"probe_scenes_step{self.step:04d}.csv"
            build_probe_csv(self.run_dir, csv, step=self.step)

        base = ProbeDataset(csv)

        # Optional class filter — keep only scenes whose stem starts with prefix.
        if self.scenario_prefix is not None:
            keep = [i for i, p in enumerate(base.scene_dirs)
                    if _stem_of(p).startswith(self.scenario_prefix)]
            if not keep:
                raise RuntimeError(
                    f"scenario_prefix={self.scenario_prefix!r} matched 0 scenes"
                )
            base.scene_dirs = [base.scene_dirs[i] for i in keep]
            base.labels = [base.labels[i] for i in keep]
            print(
                f"[PairwiseProbeDataModule] filter prefix={self.scenario_prefix!r}  "
                f"kept {len(keep)} scenes"
            )

        scenarios = [_scenario_from_path(p, self.scenario_mode) for p in base.scene_dirs]
        full = PairwiseProbeDataset(base, scenarios)

        # Group-aware split: shuffle unique scenarios, take val_split fraction.
        unique_scenarios = sorted(set(scenarios))
        n_val_scen = max(1, int(round(len(unique_scenarios) * self.val_split)))
        gen = torch.Generator().manual_seed(self.seed)
        order = torch.randperm(len(unique_scenarios), generator=gen).tolist()
        val_scen_set = {unique_scenarios[order[i]] for i in range(n_val_scen)}

        train_idx = [i for i, sc in enumerate(scenarios) if sc not in val_scen_set]
        val_idx = [i for i, sc in enumerate(scenarios) if sc in val_scen_set]
        self._train = Subset(full, train_idx)
        self._val = Subset(full, val_idx)

        # Stash val-set metadata in iteration order so the per-video / per-pair
        # CSVs written after validation can join feature scores back to scenes.
        self.val_scene_dirs = [base.scene_dirs[i] for i in val_idx]
        self.val_labels = [int(base.labels[i]) for i in val_idx]
        self.val_scenarios = [scenarios[i] for i in val_idx]

        labels = base.labels
        tp = sum(1 for i in train_idx if labels[i] == 1)
        tn = len(train_idx) - tp
        vp = sum(1 for i in val_idx if labels[i] == 1)
        vn = len(val_idx) - vp
        n_val_scenarios = len({scenarios[i] for i in val_idx})
        majority = max(vp, vn) / max(1, len(val_idx))
        print(
            f"[PairwiseProbeDataModule] split=group-aware (by scenario, mode={self.scenario_mode})  "
            f"train: {tp}+/{tn}-  val: {vp}+/{vn}-  val_scenarios={n_val_scenarios}  "
            f"val majority-class baseline: {majority:.4f}"
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers,
        )


# ============================================================================
# Lightning module: linear probe with pair-wise val accuracy
# ============================================================================


class PairwiseProbeModule(L.LightningModule):
    """One linear classifier per block; logs per-video and pair-wise accuracy.

    Args:
        num_blocks: Number of blocks in the saved features.
        hidden_dim: Feature dimension.
        lr:         Adam learning rate.
    """

    def __init__(self, num_blocks: int = 30, hidden_dim: int = 1536, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.probes = nn.ModuleList([nn.Linear(hidden_dim, 2) for _ in range(num_blocks)])
        self.num_blocks = num_blocks
        self._val_correct: list[int] = []
        self._val_total: int = 0
        self._val_scores: list[list] = []     # per-block lists of (margin, label, scenario_id)
        # Per-block per-video 0/1 correctness, accumulated in val-loader order.
        # Used after validation to dump bootstrap-ready per-video CSV.
        self._val_per_video_correct: list[list[int]] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, num_blocks, D)
        return torch.stack(
            [self.probes[i](x[:, i, :]) for i in range(self.num_blocks)],
            dim=1,
        )

    def training_step(self, batch, _batch_idx: int) -> torch.Tensor:
        x, y, _scenario_ids = batch
        logits = self(x)                                  # (B, num_blocks, 2)
        loss = torch.stack(
            [F.cross_entropy(logits[:, i, :], y) for i in range(self.num_blocks)]
        ).mean()
        acc = (logits.mean(dim=1).argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self._val_correct = [0] * self.num_blocks
        self._val_total = 0
        self._val_scores = [[] for _ in range(self.num_blocks)]
        self._val_per_video_correct = [[] for _ in range(self.num_blocks)]

    def validation_step(self, batch, _batch_idx: int) -> None:
        x, y, scenario_ids = batch
        logits = self(x)
        loss = torch.stack(
            [F.cross_entropy(logits[:, i, :], y) for i in range(self.num_blocks)]
        ).mean()
        self.log("val_loss", loss, prog_bar=True)

        # Per-video accuracy
        preds = logits.argmax(dim=2)
        correct = (preds == y.unsqueeze(1))
        correct_cpu = correct.detach().cpu().int()
        for i in range(self.num_blocks):
            self._val_correct[i] += int(correct[:, i].sum())
            self._val_per_video_correct[i].extend(correct_cpu[:, i].tolist())
        self._val_total += y.shape[0]

        # Per-block signed margins for pair-wise computation at epoch end
        margins = (logits[:, :, 1] - logits[:, :, 0]).detach().cpu()
        y_cpu = y.detach().cpu()
        sc_cpu = scenario_ids.detach().cpu()
        for b in range(x.shape[0]):
            for k in range(self.num_blocks):
                self._val_scores[k].append(
                    (margins[b, k].item(), int(y_cpu[b]), int(sc_cpu[b]))
                )

    def on_validation_epoch_end(self) -> None:
        if self._val_total == 0:
            return
        for i, c in enumerate(self._val_correct):
            self.log(f"val_acc_block_{i:02d}", c / self._val_total)
        for k, samples in enumerate(self._val_scores):
            pa = pairwise_accuracy(samples)
            if pa is not None:
                self.log(f"val_pair_acc_block_{k:02d}", pa)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# ============================================================================
# Wandb / run name (mirror of probe.py)
# ============================================================================


def generate_run_name(config) -> str:
    model_type = getattr(config, "model_type", "linear_probe_pairwise")
    dataset = getattr(config, "dataset", "intphys")
    job_id = os.environ.get("SLURM_JOB_ID", dt.now().strftime("%Y%m%d_%H%M%S"))
    return f"{model_type}_{dataset}_{job_id}"


def setup_wandb(config, run_name: str, logger) -> Optional[WandbLogger]:
    if not getattr(config, "wandb", True):
        logger.info("Wandb disabled, skipping wandb setup")
        return None
    project_name = (
        getattr(config, "wandb_project", None)
        or os.environ.get("WANDB_PROJECT")
        or DEFAULT_WANDB_PROJECT_NAME
    )
    logger.info(f"Setting up wandb on project {project_name} with run {run_name}")
    return WandbLogger(project=project_name, name=run_name)


# ============================================================================
# CLI
# ============================================================================


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s – %(message)s")
    _log = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Pair-wise linear probe (group-aware split, pair-wise val accuracy)."
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir", type=Path, metavar="DIR")
    src.add_argument("--csv", type=Path, metavar="CSV")

    # Model
    parser.add_argument("--num-blocks", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=1536)
    parser.add_argument("--lr", type=float, default=1e-3)

    # Data
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--scenario-mode", type=str, default="intphys",
                        choices=["intphys", "inflevel", "scene"],
                        help="How to extract scenario id from scene stem.")

    # Trainer
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num-nodes", type=int, default=1)

    # Wandb
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="intphys")

    # Per-class probing
    parser.add_argument("--scenario-prefix", type=str, default=None,
                        help="Filter scenes whose stem starts with this prefix (e.g. 'O1').")
    parser.add_argument("--per-class", action="store_true",
                        help="Detect classes from data and run probe once per class.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for the group-aware train/val split.")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated list of seeds for multi-seed "
                             "probing (e.g. '0,1,2,3,4'). When provided, "
                             "overrides --seed; the probe runs once per seed "
                             "and each seed's outputs go to a "
                             "<run-dir>/seed{S}/ subdirectory so existing "
                             "single-seed file paths and regexes keep "
                             "working untouched.")

    args = parser.parse_args()
    args.wandb = not args.no_wandb
    args.model_type = "linear_probe_pairwise"

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    # `results_dir` is reassigned per-seed inside the loop further down, but
    # `run_probe` (a closure defined below) reads it from this enclosing
    # scope so each iteration writes into the correct directory.
    results_dir_root = args.run_dir if args.run_dir else args.csv.parent
    results_dir = results_dir_root

    def run_probe(prefix: Optional[str] = None) -> None:
        suffix = f"_{prefix}" if prefix else ""
        run_name_local = generate_run_name(args) + suffix
        wandb_logger_local = setup_wandb(args, run_name_local, _log)
        loggers_to_use = [wandb_logger_local] if wandb_logger_local else []

        dm = PairwiseProbeDataModule(
            run_dir=args.run_dir,
            csv_path=args.csv,
            step=args.step,
            legacy=args.legacy,
            val_split=args.val_split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            scenario_mode=args.scenario_mode,
            scenario_prefix=prefix,
            seed=args.seed,
        )
        model = PairwiseProbeModule(
            num_blocks=args.num_blocks,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
        )

        checkpoint_cb = L.pytorch.callbacks.ModelCheckpoint(
            dirpath=results_dir,
            filename=f"probe_pairwise_step{args.step:04d}{suffix}_last",
            save_last=True,
            every_n_epochs=1,
        )

        trainer = L.Trainer(
            max_epochs=args.max_epochs,
            accelerator=accelerator,
            devices=args.devices,
            num_nodes=args.num_nodes,
            logger=loggers_to_use or True,
            log_every_n_steps=1,
            callbacks=[checkpoint_cb],
        )
        trainer.fit(model, datamodule=dm)

        _log.info(f"Running final validation (prefix={prefix!r})...")
        val_results = trainer.validate(model, datamodule=dm)

        results_file = results_dir / f"probe_pairwise_results_step{args.step:04d}{suffix}.txt"
        with open(results_file, "w") as f:
            f.write("Probe pairwise training results\n")
            f.write(f"{'=' * 50}\n")
            f.write(f"Run dir:        {args.run_dir}\n")
            f.write(f"CSV:            {args.csv}\n")
            f.write(f"Step:           {args.step}\n")
            f.write(f"Scenario mode:  {args.scenario_mode}\n")
            f.write(f"Class prefix:   {prefix}\n")
            f.write(f"Epochs:         {args.max_epochs}\n")
            f.write(f"LR:             {args.lr}\n")
            f.write(f"Batch size:     {args.batch_size}\n")
            f.write(f"Val split:      {args.val_split} (group-aware by scenario)\n")
            f.write(f"Seed:           {args.seed}\n")
            f.write(f"\n{'=' * 50}\n")
            f.write(f"Final validation\n")
            f.write(f"{'=' * 50}\n")

            metrics = val_results[0] if val_results else {}
            val_loss = metrics.get("val_loss", float("nan"))
            f.write(f"val_loss: {val_loss:.4f}\n\n")

            f.write(f"{'Block':<10} {'Per-video':>12} {'Pair-wise':>12}\n")
            f.write(f"{'-' * 36}\n")
            per_video_accs, pair_accs = [], []
            for i in range(args.num_blocks):
                pv = metrics.get(f"val_acc_block_{i:02d}", float("nan"))
                pa = metrics.get(f"val_pair_acc_block_{i:02d}", float("nan"))
                per_video_accs.append(pv)
                pair_accs.append(pa)
                f.write(f"block_{i:02d}   {pv:>12.4f} {pa:>12.4f}\n")

            f.write(f"\n{'=' * 50}\n")
            if per_video_accs:
                mean_pv = sum(0 if a != a else a for a in per_video_accs) / len(per_video_accs)
                best_pv_i = max(range(len(per_video_accs)),
                                key=lambda i: -1 if per_video_accs[i] != per_video_accs[i] else per_video_accs[i])
                f.write(f"Mean per-video accuracy: {mean_pv:.4f}\n")
                f.write(f"Best per-video block:    block_{best_pv_i:02d} ({per_video_accs[best_pv_i]:.4f})\n")
            if pair_accs:
                mean_pa = sum(0 if a != a else a for a in pair_accs) / len(pair_accs)
                best_pa_i = max(range(len(pair_accs)),
                                key=lambda i: -1 if pair_accs[i] != pair_accs[i] else pair_accs[i])
                f.write(f"Mean pair-wise accuracy: {mean_pa:.4f}\n")
                f.write(f"Best pair-wise block:    block_{best_pa_i:02d} ({pair_accs[best_pa_i]:.4f})\n")

        _log.info(f"Results written to {results_file}")

        csv_file = results_dir / f"probe_pairwise_results_step{args.step:04d}{suffix}.csv"
        pd.DataFrame({
            "block": list(range(args.num_blocks)),
            "per_video_acc": per_video_accs,
            "pair_acc": pair_accs,
        }).to_csv(csv_file, index=False)
        _log.info(f"Curve CSV written to {csv_file}")

        # ── Bootstrap-ready raw outputs ─────────────────────────────────────
        # Per-video CSV: one row per val video, three column groups per block.
        #   block_NN          — 0/1 correctness flag (figures bootstrap on this)
        #   score_block_NN    — signed margin logit_pos - logit_neg
        #   pred_block_NN     — argmax class (1 = plausible, 0 = impossible)
        # The score is what the probe actually produced before thresholding,
        # so you can recompute correctness, ROC, calibration, etc. from it.
        per_video_rows = []
        for vi, scene_dir in enumerate(dm.val_scene_dirs):
            row = {
                "scene_dir": scene_dir,
                "label": dm.val_labels[vi],
                "scenario": dm.val_scenarios[vi],
            }
            for k in range(args.num_blocks):
                margin = float(model._val_scores[k][vi][0])
                correct = int(model._val_per_video_correct[k][vi])
                row[f"block_{k:02d}"] = correct
                row[f"score_block_{k:02d}"] = margin
                row[f"pred_block_{k:02d}"] = int(margin > 0)
            per_video_rows.append(row)
        per_video_csv = (
            results_dir / f"probe_pairwise_pervideo_step{args.step:04d}{suffix}.csv"
        )
        pd.DataFrame(per_video_rows).to_csv(per_video_csv, index=False)
        _log.info(f"Per-video CSV written to {per_video_csv}  "
                  f"(score + pred + correct per block)")

        # Per-pair CSV: one row per (plausible, impossible) pair within a
        # scenario, one column per block. Cell value = 1.0 / 0.5 / 0.0.
        per_pair_rows = pairwise_correctness_per_block(
            model._val_scores,
            scene_dirs=dm.val_scene_dirs,
            scenario_strs=dm.val_scenarios,
        )
        per_pair_csv = (
            results_dir / f"probe_pairwise_perpair_step{args.step:04d}{suffix}.csv"
        )
        pd.DataFrame(per_pair_rows).to_csv(per_pair_csv, index=False)
        _log.info(f"Per-pair correctness CSV written to {per_pair_csv}")

        # Finish wandb run cleanly so the next class gets its own run.
        if wandb_logger_local is not None and wandb is not None:
            try:
                wandb.finish()
            except Exception:
                pass

    # Class detection happens once before the seed loop — it's seed-agnostic.
    if args.per_class:
        if args.csv is not None:
            csv_for_detect = args.csv
        elif args.legacy:
            csv_for_detect = args.run_dir / "probe_scenes_legacy.csv"
            build_probe_csv(args.run_dir, csv_for_detect, legacy=True)
        else:
            csv_for_detect = args.run_dir / f"probe_scenes_step{args.step:04d}.csv"
            build_probe_csv(args.run_dir, csv_for_detect, step=args.step)

        base_for_detect = ProbeDataset(csv_for_detect)
        classes = detect_classes(base_for_detect.scene_dirs, args.scenario_mode)
        if not classes:
            raise RuntimeError(
                f"--per-class detected 0 classes for mode={args.scenario_mode!r}"
            )
        _log.info(f"Per-class probing — classes: {classes}")

    # Resolve seeds. --seeds wins over --seed when provided. In single-seed
    # mode the outputs land directly in results_dir_root (no subdir), so the
    # filenames + regexes existing figure scripts use are unchanged.
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = [args.seed]
    multi_seed = len(seeds) > 1
    _log.info(
        f"Seeds: {seeds}  "
        f"({'multi-seed → seed{S}/ subdirs' if multi_seed else 'single-seed → root'})"
    )

    for seed in seeds:
        # Lightning's seed_everything covers Python random, numpy, torch
        # (CPU + CUDA), and dataloader workers — needed for the probe init
        # + minibatch order to actually differ across seeds.
        L.seed_everything(seed, workers=True)
        args.seed = seed

        if multi_seed:
            results_dir = results_dir_root / f"seed{seed}"
            results_dir.mkdir(parents=True, exist_ok=True)
            _log.info(f"=== seed {seed} — writing to {results_dir} ===")
        else:
            results_dir = results_dir_root

        if args.per_class:
            for cls in classes:
                _log.info(f"=== seed {seed}  class {cls} ===")
                run_probe(prefix=cls)
        else:
            run_probe(prefix=args.scenario_prefix)
