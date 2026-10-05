<div align="center">
<h1>The Invisible Hand of Physics: When Video Diffusion Models Know More Than They Show</h1>

**Parsa Esmati**\* · **Somjit Nath**\*
<br>
**Katja Hofmann** · **Derek Nowrouzezahrai** · **Samira Ebrahimi Kahou**† · **Majid Mirmehdi**†
<br>
**\*Equal contribution · †Equal supervision**

**Accepted at NeurIPS 2026 (Spotlight 🌟)**

<a href="https://arxiv.org/abs/2606.05328"><img src="https://img.shields.io/badge/arXiv-2606.05328-b31b1b" alt="arXiv"></a>
</div>

Official implementation for *The Invisible Hand of Physics*. The codebase provide the full pipelines and the tools required to probe video diffusion models along the latent trajectories to investigate their score on physical plausibility, and to regress physical parameters such as initial conditions and velocity. 

The code inverts real videos through Wan 2.1, CogVideoX and LTX-Video to recover those trajectories, saves the DiT features along the way, and trains linear probes on these features foe decodability. 

Read below for detailed usage and the benchmarks provided in the paper:

## Usage

There are three steps, all run from the repository root.

### 1. Build the dataset CSVs

> [!NOTE]
> Download the benchmarks from their official sources first:
> - **IntPhys** (dev set): [intphys.cognitive-ml.fr/download.html](https://intphys.cognitive-ml.fr/download.html)
> - **InfLevel** (InfLevel-Lab): [github.com/allenai/inflevel](https://github.com/allenai/inflevel)
> - **PhyWorld** (parabola evaluation set): [github.com/phyworld/phyworld](https://github.com/phyworld/phyworld)

The inversion scripts need a table that lists every video with its plausibility label. The label is copied into each video's output folder during inversion, and that is what the probe is trained on. Run once per dataset:

```bash
python create_intphys_csv.py /path/to/datasets/intphys/dev      # the folder with O1/ O2/ O3/
python create_inflevel_csv.py /path/to/datasets/inflevel_lab    # the folder with continuity/ gravity/ solidity/
```

This writes `intphys_dev.csv`, and for InfLevel one CSV per principle: `inflevel_gravity.csv`, `inflevel_solidity.csv` and `inflevel_continuity.csv`.

PhyWorld needs no CSV. Its videos and initial conditions are read directly from `parabola_eval.hdf5`:

```bash
hf download magicr/phyworld --repo-type dataset --include "*parabola*eval*" --local-dir /path/to/datasets/phyworld
```

### 2. Invert the videos

Each model has its own script. It runs every video backwards from the clean video to noise over the number of steps given by `--steps`, and saves the output of every transformer block at the steps listed in `--capture-steps`. Each run writes a new timestamped folder under `--output-dir`.

```bash
CKPT=/path/to/checkpoints
DATA=/path/to/datasets
OUT=/path/to/results
```

> [!NOTE]
> The model checkpoints are hosted on Hugging Face. Download each one into `$CKPT` by following the [Hugging Face download guide](https://huggingface.co/docs/huggingface_hub/guides/cli), for example:
>
> `hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --local-dir $CKPT/Wan2.1-T2V-1.3B-Diffusers`

#### Wan 2.1

Checkpoint: [`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers). Example commands are as follows:

```bash
# IntPhys
python inference.py \
  --ckpt-dir $CKPT/Wan2.1-T2V-1.3B-Diffusers \
  --data-dir $DATA/intphys/dev --dataset intphys --meta-csv intphys_dev.csv \
  --output-dir $OUT/wan/intphys \
  --reverse_sample --steps 100 --guidance-scale 1 \
  --capture-steps 0 10 20 30 40 50 60 70 80 90 99 \
  --height 256 --width 256 --num-frames 100 --batch-size 16

# InfLevel, one run per principle
for p in gravity solidity continuity; do
  python inference.py \
    --ckpt-dir $CKPT/Wan2.1-T2V-1.3B-Diffusers \
    --data-dir $DATA/inflevel_lab --dataset inflevel --meta-csv inflevel_$p.csv \
    --output-dir $OUT/wan/inflevel_$p \
    --reverse_sample --steps 100 --guidance-scale 1 \
    --capture-steps 0 10 20 30 40 50 60 70 80 90 \
    --height 256 --width 256 --num-frames 100 --batch-size 16
done

# PhyWorld
python inference.py \
  --ckpt-dir $CKPT/Wan2.1-T2V-1.3B-Diffusers \
  --data-dir $DATA/phyworld/id_ood_data/parabola_eval.hdf5 --dataset phyworld --phyworld-split 00000 \
  --output-dir $OUT/wan/phyworld \
  --reverse_sample --steps 100 --guidance-scale 1 \
  --capture-steps 0 10 20 30 40 50 60 70 80 90 99 \
  --height 256 --width 256 --num-frames 100 --batch-size 16 --num-workers 16
```

#### CogVideoX-2b

Checkpoint: [`zai-org/CogVideoX-2b`](https://huggingface.co/zai-org/CogVideoX-2b). Example commands are as follows:

```bash
# IntPhys
python cogvideox_inference.py \
  --ckpt-dir $CKPT/CogVideoX-2b \
  --data-dir $DATA/intphys/dev --dataset intphys --meta-csv intphys_dev.csv \
  --output-dir $OUT/cogvideox/intphys \
  --reverse_sample --steps 100 --guidance-scale 1 \
  --capture-steps 0 10 20 30 40 50 60 70 80 90 99 \
  --height 256 --width 256 --num-frames 100 --batch-size 16

# InfLevel, one run per principle
for p in gravity solidity continuity; do
  python cogvideox_inference.py \
    --ckpt-dir $CKPT/CogVideoX-2b \
    --data-dir $DATA/inflevel_lab --dataset inflevel --meta-csv inflevel_$p.csv \
    --output-dir $OUT/cogvideox/inflevel_$p \
    --reverse_sample --steps 100 --guidance-scale 1 \
    --capture-steps 0 10 20 30 40 50 60 70 80 90 \
    --height 256 --width 256 --num-frames 100 --batch-size 16
done

# PhyWorld
python cogvideox_inference.py \
  --ckpt-dir $CKPT/CogVideoX-2b \
  --data-dir $DATA/phyworld/id_ood_data/parabola_eval.hdf5 --dataset phyworld --phyworld-split 00000 \
  --output-dir $OUT/cogvideox/phyworld \
  --reverse_sample --steps 100 --guidance-scale 1 \
  --capture-steps 0 10 20 30 40 50 60 70 80 90 99 \
  --height 256 --width 256 --num-frames 100 --batch-size 16 --num-workers 16
```

#### LTX-Video

Checkpoint: [`Lightricks/LTX-Video`](https://huggingface.co/Lightricks/LTX-Video). Example commands are as follows:

```bash
# IntPhys
python ltx_inference.py \
  --ckpt-dir $CKPT/LTX-Video \
  --data-dir $DATA/intphys/dev --dataset intphys --meta-csv intphys_dev.csv \
  --output-dir $OUT/ltx/intphys \
  --reverse_sample --steps 100 --guidance-scale 1 \
  --capture-steps 0 10 20 30 40 50 60 70 80 90 99 \
  --height 256 --width 256 --num-frames 100 --batch-size 16

# InfLevel, one run per principle
for p in gravity solidity continuity; do
  python ltx_inference.py \
    --ckpt-dir $CKPT/LTX-Video \
    --data-dir $DATA/inflevel_lab --dataset inflevel --meta-csv inflevel_$p.csv \
    --output-dir $OUT/ltx/inflevel_$p \
    --reverse_sample --steps 100 --guidance-scale 1 \
    --capture-steps 0 10 20 30 40 50 60 70 80 90 99 \
    --height 256 --width 256 --num-frames 100 --batch-size 16
done

# PhyWorld
python ltx_inference.py \
  --ckpt-dir $CKPT/LTX-Video \
  --data-dir $DATA/phyworld/id_ood_data/parabola_eval.hdf5 --dataset phyworld --phyworld-split 00000 \
  --output-dir $OUT/ltx/phyworld \
  --reverse_sample --steps 100 --guidance-scale 1 \
  --capture-steps 0 10 20 30 40 50 60 70 80 90 99 \
  --height 256 --width 256 --num-frames 100 --batch-size 16 --num-workers 16
```

#### Baseline encoders

V-JEPA 2 and VideoMAE-Large are run with `encode_video_encoder.py`. It takes the same dataset arguments and writes the same format, at step 0 and directly into `--output-dir`. An example command for V-JEPA 2 is as follows:

```bash
python encode_video_encoder.py --ckpt facebook/vjepa2-vitl-fpc64-256 \
  --data-dir $DATA/intphys/dev --dataset intphys --meta-csv intphys_dev.csv \
  --output-dir $OUT/vjepa2/intphys \
  --num-frames 64 --image-size 256 --input-layout BTCHW
```

For VideoMAE-Large use `--ckpt MCG-NJU/videomae-large --num-frames 16 --image-size 224`.

### 3. Train the probe

`probe_pairwise.py` fits one linear classifier per transformer block on the features of one inversion step, chosen with `--step`, and reports per-video and pair-wise accuracy on held-out videos. Example commands are as follows:

```bash
# IntPhys: --per-class trains a separate probe for O1, O2 and O3
RUN=$OUT/wan/intphys/2026-04-16_11-53-37    # the timestamped folder written in step 2
python probe_pairwise.py --run-dir $RUN \
  --num-blocks 30 --hidden-dim 1536 \
  --scenario-mode intphys --step 50 --per-class --seeds 0,1,2,3,4

# InfLevel: run once per principle folder, without --per-class
RUN=$OUT/wan/inflevel_gravity/2026-04-21_17-20-33
python probe_pairwise.py --run-dir $RUN \
  --num-blocks 30 --hidden-dim 1536 \
  --scenario-mode inflevel --step 50 --seeds 0,1,2,3,4
```

For PhyWorld, `probe_regress.py` fits a linear regressor per block for the initial position `x0` and the initial velocity `v0`, and reports MSE, MAE and R². An example command is as follows:

```bash
RUN=$OUT/wan/phyworld/2026-04-27_19-45-49
python probe_regress.py --run-dir $RUN \
  --num-blocks 30 --hidden-dim 1536 --step 50
```

For the other models, set the two sizes to match:

| Model | `--num-blocks` | `--hidden-dim` |
|---|---|---|
| Wan 2.1 | 30 | 1536 |
| CogVideoX-2b | 30 | 1920 |
| LTX-Video | 28 | 2048 |
| V-JEPA 2, VideoMAE-Large | 25 | 1024 |

For the baseline encoders, also use `--step 0` and pass their `--output-dir` as `--run-dir`.

The results are written into the run folder as `probe_pairwise_results_step<NNNN>*.csv` (one `seed<N>/` sub-folder per seed) and `probe_regress_results_step<NNNN>.csv`, where `<NNNN>` is the step passed to `--step`. Logging to Weights & Biases is on by default; add `--no-wandb` to turn it off.
