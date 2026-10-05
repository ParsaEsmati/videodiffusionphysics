#!/usr/bin/env python3
"""
Regression linear probe: one linear head per transformer block and target,
predicting the PhyWorld initial position ``x0`` and velocity ``v0`` from saved
features. Reports MSE, MAE and R² per block.
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime as dt
from pathlib import Path
from typing import Optional

import lightning as L
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, Subset

from data_pipeline.probe_dataset import RegressDataset, build_regress_csv


DEFAULT_WANDB_PROJECT_NAME = "videophys-probe-regress"
TARGET_COLS = ("x0", "v0")


# ============================================================================
# DataModule — random per-sample split (PhyWorld parabola has no scenario id)
# ============================================================================


class RegressionProbeDataModule(L.LightningDataModule):
    """Data for the regression probe, with a random train/validation split.

    Args:
        run_dir:     Run folder to index (ignored if ``csv_path`` is given).
        csv_path:    Existing regression CSV.
        step:        Inversion step to probe.
        legacy:      Features sit directly in each scene folder.
        val_split:   Fraction of the samples held out for validation.
        batch_size:  DataLoader batch size.
        num_workers: DataLoader worker processes.
        seed:        Seed of the split.
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
        self.seed = seed

        self._train: Optional[Subset] = None
        self._val: Optional[Subset] = None
        self._target_stats: Optional[dict] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.csv_path is not None:
            csv = self.csv_path
        elif self.legacy:
            csv = self.run_dir / "regress_scenes_legacy.csv"
            build_regress_csv(self.run_dir, csv, legacy=True, target_keys=TARGET_COLS)
        else:
            csv = self.run_dir / f"regress_scenes_step{self.step:04d}.csv"
            build_regress_csv(self.run_dir, csv, step=self.step, target_keys=TARGET_COLS)

        full = RegressDataset(csv, target_cols=TARGET_COLS)

        # Random split — no scenario grouping needed (each (x0, v0) is unique enough).
        n = len(full)
        n_val = max(1, int(round(n * self.val_split)))
        gen = torch.Generator().manual_seed(self.seed)
        order = torch.randperm(n, generator=gen).tolist()
        val_idx = order[:n_val]
        train_idx = order[n_val:]
        self._train = Subset(full, train_idx)
        self._val = Subset(full, val_idx)

        # Stash val-set scene_dirs in iteration order so the per-sample CSV
        # written after validation can be joined back to scenes.
        self.val_scene_dirs = [full.scene_dirs[i] for i in val_idx]

        # Stash per-target training-set statistics — used to centre/scale
        # predictions if you later wire in target standardisation.
        targets = torch.from_numpy(full.targets)
        train_targets = targets[train_idx]
        self._target_stats = {
            "mean": train_targets.mean(dim=0).tolist(),
            "std":  train_targets.std(dim=0).tolist(),
        }
        print(
            f"[RegressionProbeDataModule] split=random  "
            f"train={len(train_idx)}  val={n_val}  "
            f"target_mean={self._target_stats['mean']}  "
            f"target_std={self._target_stats['std']}"
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
# Lightning module — one Linear(D, 1) head per (block, target)
# ============================================================================


class RegressionProbeModule(L.LightningModule):
    """One linear head per block for ``x0`` and one for ``v0``.

    Args:
        num_blocks: Number of blocks in the saved features.
        hidden_dim: Feature dimension.
        lr:         Adam learning rate.
    """

    def __init__(self, num_blocks: int = 30, hidden_dim: int = 1536, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.x0_heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_blocks)])
        self.v0_heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_blocks)])
        self.num_blocks = num_blocks

        # Buffers for epoch-end R² aggregation.
        self._val_preds: list[torch.Tensor] = []
        self._val_targets: list[torch.Tensor] = []
        # Final concatenated tensors from the last validation epoch — used
        # after trainer.validate() to dump bootstrap-ready per-sample CSVs.
        self.last_val_preds: Optional[torch.Tensor] = None       # (N, NB, 2)
        self.last_val_targets: Optional[torch.Tensor] = None     # (N, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: ``(B, num_blocks, D)`` → preds ``(B, num_blocks, 2)`` = [x0, v0]."""
        x0 = torch.stack(
            [self.x0_heads[i](x[:, i, :]).squeeze(-1) for i in range(self.num_blocks)],
            dim=1,
        )
        v0 = torch.stack(
            [self.v0_heads[i](x[:, i, :]).squeeze(-1) for i in range(self.num_blocks)],
            dim=1,
        )
        return torch.stack([x0, v0], dim=-1)

    def training_step(self, batch, _batch_idx: int) -> torch.Tensor:
        x, y = batch                                 # x: (B, NB, D), y: (B, 2)
        preds = self(x)                              # (B, NB, 2)
        target = y.unsqueeze(1).expand_as(preds)
        loss = F.mse_loss(preds, target)
        with torch.no_grad():
            mae = (preds - target).abs().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_mae",  mae,  prog_bar=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self._val_preds = []
        self._val_targets = []

    def validation_step(self, batch, _batch_idx: int) -> None:
        x, y = batch
        preds = self(x)                                              # (B, NB, 2)
        target = y.unsqueeze(1).expand_as(preds)
        loss = F.mse_loss(preds, target)
        self.log("val_loss", loss, prog_bar=True)
        self._val_preds.append(preds.detach().cpu())
        self._val_targets.append(y.detach().cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        preds   = torch.cat(self._val_preds,   dim=0)               # (N, NB, 2)
        targets = torch.cat(self._val_targets, dim=0)               # (N, 2)
        # Stash for post-fit access; the last call (the final trainer.validate)
        # is what gets dumped.
        self.last_val_preds = preds
        self.last_val_targets = targets
        N, NB, _ = preds.shape

        for d, name in enumerate(TARGET_COLS):
            t = targets[:, d]
            t_var = ((t - t.mean()) ** 2).sum().clamp(min=1e-12)
            for k in range(NB):
                p = preds[:, k, d]
                err = p - t
                mse = (err ** 2).mean().item()
                mae = err.abs().mean().item()
                ss_res = (err ** 2).sum()
                r2 = (1.0 - ss_res / t_var).item()
                self.log(f"val_{name}_mse_block_{k:02d}", mse)
                self.log(f"val_{name}_mae_block_{k:02d}", mae)
                self.log(f"val_{name}_r2_block_{k:02d}",  r2)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# ============================================================================
# Wandb / run name (mirror of probe_pairwise.py)
# ============================================================================


def generate_run_name(config) -> str:
    model_type = getattr(config, "model_type", "linear_probe_regress")
    dataset = getattr(config, "dataset", "phyworld")
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
        description="Linear regression probe (one head per block × target)."
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
    parser.add_argument("--seed", type=int, default=42)

    # Trainer
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num-nodes", type=int, default=1)

    # Wandb
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="phyworld")

    args = parser.parse_args()
    args.wandb = not args.no_wandb
    args.model_type = "linear_probe_regress"

    run_name = generate_run_name(args)
    wandb_logger = setup_wandb(args, run_name, _log)
    loggers_to_use = [wandb_logger] if wandb_logger else []

    dm = RegressionProbeDataModule(
        run_dir=args.run_dir,
        csv_path=args.csv,
        step=args.step,
        legacy=args.legacy,
        val_split=args.val_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    model = RegressionProbeModule(
        num_blocks=args.num_blocks,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    results_dir = args.run_dir if args.run_dir else args.csv.parent

    checkpoint_cb = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=results_dir,
        filename=f"probe_regress_step{args.step:04d}_last",
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

    _log.info("Running final validation...")
    val_results = trainer.validate(model, datamodule=dm)
    metrics = val_results[0] if val_results else {}

    # Per-block, per-target table → txt + csv (mirrors probe_pairwise output)
    results_file = results_dir / f"probe_regress_results_step{args.step:04d}.txt"
    csv_file     = results_dir / f"probe_regress_results_step{args.step:04d}.csv"

    rows = []
    for i in range(args.num_blocks):
        row = {"block": i}
        for d in TARGET_COLS:
            row[f"{d}_mse"] = metrics.get(f"val_{d}_mse_block_{i:02d}", float("nan"))
            row[f"{d}_mae"] = metrics.get(f"val_{d}_mae_block_{i:02d}", float("nan"))
            row[f"{d}_r2"]  = metrics.get(f"val_{d}_r2_block_{i:02d}",  float("nan"))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(csv_file, index=False)

    with open(results_file, "w") as f:
        f.write("Probe regression training results\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Run dir:    {args.run_dir}\n")
        f.write(f"CSV:        {args.csv}\n")
        f.write(f"Step:       {args.step}\n")
        f.write(f"Targets:    {list(TARGET_COLS)}\n")
        f.write(f"Epochs:     {args.max_epochs}\n")
        f.write(f"LR:         {args.lr}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Val split:  {args.val_split} (random)\n")
        f.write(f"Seed:       {args.seed}\n")
        f.write(f"\n{'=' * 60}\n")
        f.write(f"Final validation\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"val_loss: {metrics.get('val_loss', float('nan')):.4f}\n\n")

        header = f"{'Block':<8}"
        for d in TARGET_COLS:
            header += f" {d+' MSE':>10} {d+' MAE':>10} {d+' R²':>10}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for i in range(args.num_blocks):
            line = f"block_{i:02d}"
            for d in TARGET_COLS:
                line += (
                    f" {metrics.get(f'val_{d}_mse_block_{i:02d}', float('nan')):>10.4f}"
                    f" {metrics.get(f'val_{d}_mae_block_{i:02d}', float('nan')):>10.4f}"
                    f" {metrics.get(f'val_{d}_r2_block_{i:02d}',  float('nan')):>10.4f}"
                )
            f.write(line + "\n")

        f.write(f"\n{'=' * 60}\n")
        for d in TARGET_COLS:
            r2s = [metrics.get(f"val_{d}_r2_block_{i:02d}", float("nan")) for i in range(args.num_blocks)]
            best_i = max(range(len(r2s)),
                         key=lambda i: -1e9 if r2s[i] != r2s[i] else r2s[i])
            f.write(f"Best block for {d}: block_{best_i:02d}  R²={r2s[best_i]:.4f}\n")

    _log.info(f"Results written to {results_file}")
    _log.info(f"Curve CSV written to {csv_file}")

    # ── Bootstrap-ready raw outputs ─────────────────────────────────────────
    # One row per val sample × target. Columns: scene_dir, target, true,
    # block_00_pred, ..., block_NN_pred. Figures bootstrap by resampling rows
    # within a target and recomputing R² / MAE / MSE on each resample.
    if model.last_val_preds is not None and model.last_val_targets is not None:
        preds = model.last_val_preds.numpy()                          # (N, NB, 2)
        targs = model.last_val_targets.numpy()                        # (N, 2)
        scene_dirs = dm.val_scene_dirs
        N = preds.shape[0]
        assert len(scene_dirs) == N, (
            f"val_scene_dirs ({len(scene_dirs)}) != #val samples ({N})"
        )

        rows = []
        for d, name in enumerate(TARGET_COLS):
            for n in range(N):
                row = {
                    "scene_dir": scene_dirs[n],
                    "target": name,
                    "true": float(targs[n, d]),
                }
                for k in range(args.num_blocks):
                    row[f"block_{k:02d}_pred"] = float(preds[n, k, d])
                rows.append(row)

        per_sample_csv = (
            results_dir / f"probe_regress_persample_step{args.step:04d}.csv"
        )
        pd.DataFrame(rows).to_csv(per_sample_csv, index=False)
        _log.info(f"Per-sample CSV written to {per_sample_csv}")
    else:
        _log.warning("No validation predictions to dump for per-sample CSV.")
