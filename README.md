# ARCHITECTURE ALIGNMENT WITH SPARSE PRIORS IN TABULAR FOUNDATION MODELS

Controlled experiments with row-token (`1d`) and alternating-axis (`2d`) models.

Final weights and generation configurations are included in [public_checkpoints/](public_checkpoints/).

Use Python 3.10+ and run commands from the repository root:

```bash
python -m pip install -r requirements.txt
```

Generated data and results go in `paper/`. Skip training when using the supplied weights. Use new output directories for each run; every Python entry point supports `--help`.

## Released TFMs respond differently to irrelevant features

These experiments evaluate whether pretrained tabular foundation models remain robust when inputs are expanded with irrelevant features. The same paired tasks are used at every dose, so changes are measured within task rather than between different datasets.

### Experimental design

The released configuration evaluates TabDPT v1.2 with retrieval disabled, TabPFN v2, TabPFN v2.5, TabPFN v3, and TabICL v2. It covers 42 OpenML regression datasets and Friedman synthetic tasks. The irrelevant-to-relevant feature ratios are `0, 0.2, 0.4, 1, 2, 4, 7`.

Two complementary measurements are produced:

1. **Predictive robustness:** save raw predictions at every irrelevant-feature dose, then report paired changes in R² relative to the zero-dose task.
2. **Functional suppression:** independently permute relevant and irrelevant query features and compute partial-dependence variation to measure which features control the fitted prediction function.

OpenML nuisance columns are whole-row permuted copies. Friedman nuisance columns are independent uniform draws. Dataset preparation is cached and shared by prediction and functional analysis.

The experiment definitions are in [default_config.json](experiments_arch/experiment_0_pretrained/default_config.json) and [friedman_sweeps_config.json](experiments_arch/experiment_0_pretrained/friedman_sweeps_config.json).

### Checkpoints and model packages

Install the pinned packages from `requirements.txt`, then place the official model weights in `checkpoints/` using these filenames:

```text
tabdpt1_2.safetensors
tabpfn-v2-regressor.ckpt
tabpfn-v2.5-regressor-v2.5_default.ckpt
tabpfn-v3-regressor-v3_default.ckpt
tabicl-regressor-v2-20260212.ckpt
```

The checkpoint files are third-party assets and are not included in this repository. Pass a different project-relative directory through `CHECKPOINT_DIR` when needed.

### Running the released-TFM experiments

The following commands run the OpenML predictive experiment, the Friedman sample-size/noise sweeps, and the OpenML functional experiment:

```bash
bash experiments_arch/experiment_0_pretrained/run_part_i1.sh
bash experiments_arch/experiment_0_pretrained/run_friedman_sweeps.sh
bash experiments_arch/experiment_0_pretrained/run_part_i2.sh
```

Run the matching Friedman functional analysis with:

```bash
python -m experiments_arch.experiment_0_pretrained run-part-i2-friedman-sweeps \
  --config experiments_arch/experiment_0_pretrained/friedman_sweeps_config.json \
  --out-dir artifacts/experiment_0_pretrained \
  --checkpoint-dir checkpoints --device cuda
```

All stages are cache-aware and write raw data, predictions, evaluations, and failure records under `artifacts/experiment_0_pretrained/`. To regenerate the paper and appendix figure bundle from the completed evaluations, run:

```bash
python -m experiments_arch.experiment_0_pretrained plot-publication \
  --artifact-root artifacts/experiment_0_pretrained
```

Use `--models`, `--datasets`, or `--ratios` on the Python entry points for a smaller diagnostic run. The shell wrappers also accept extra arguments and forward them to their corresponding Python commands.

The appendix also compares TabDPT v1.0, v1.1, and v1.3 with retrieval disabled, reusing the v1.2 results above. These additional versions require separate environments.

<details>
<summary>Commands for additional TabDPT versions</summary>

Add `tabdpt_76M.ckpt` (v1.0), `tabdpt1_1.safetensors` (v1.1), and `tabdpt1_3.safetensors` (v1.3) to `checkpoints/`.

Set up the environments and use the main experiment’s cached dataset indexes:

```bash
bash experiments_arch/experiment_0_pretrained/setup_tabdpt_release_envs.sh

python -m experiments_arch.experiment_0_pretrained prepare-tabdpt-version-indexes \
  --friedman-index /path/to/friedman/index.csv \
  --part-i1-index /path/to/openml/index.csv \
  --out-dir artifacts/experiment_0_tabdpt_versions/data_indexes
```

Run both analyses for each version:

```bash
for version in v1_0 v1_1 v1_3; do
  bash experiments_arch/experiment_0_pretrained/run_tabdpt_release.sh "$version"
done
```

The setup and run scripts share `ENV_ROOT`, which defaults to the ignored
`.tabdpt-version-envs/` directory in the repository. Set `BASE_PYTHON` to the
Python executable used to create the environments, or set `ENV_ROOT` in both
commands to store them elsewhere.

Combine with the v1.2 results and plot:

```bash
python -m experiments_arch.experiment_0_pretrained evaluate-tabdpt-versions \
  --friedman-v1-0 artifacts/experiment_0_tabdpt_versions/v1_0/friedman/predictions \
  --openml-v1-0 artifacts/experiment_0_tabdpt_versions/v1_0/openml/predictions \
  --friedman-v1-1 artifacts/experiment_0_tabdpt_versions/v1_1/friedman/predictions \
  --openml-v1-1 artifacts/experiment_0_tabdpt_versions/v1_1/openml/predictions \
  --friedman-v1-3 artifacts/experiment_0_tabdpt_versions/v1_3/friedman/predictions \
  --openml-v1-3 artifacts/experiment_0_tabdpt_versions/v1_3/openml/predictions \
  --v1-2-effects /path/to/v1_2_friedman/paired_effects.csv \
  --v1-2-effects /path/to/v1_2_openml/paired_effects.csv \
  --out-dir artifacts/experiment_0_tabdpt_versions/evaluations

python -m experiments_arch.experiment_0_pretrained plot-tabdpt-versions \
  --versions-root artifacts/experiment_0_tabdpt_versions
```

</details>

## Controlled experiments

### 1. Data generation

Training and test data use data seeds 0 (search) and 1 (final). The released `public_checkpoints/configs/` manifests record all parameters. Example for final `d=10,k=1` data:

```bash
python -m experiments_arch.experiment_1_controlled_bayes.data_generation.generate \
  --output-dir paper/data/d10_k1_seed1 --d 10 --k 1 --seed 1 \
  --n-context-min 50 --n-context-max 100 --n-total 150 \
  --train-tasks 320000 --validation-tasks 3200 --test-tasks 32000 \
  --signal-energy 1 --noise-std 1.5 --generator-batch-size 32
```

For `d=20`, use 640,000 training tasks; change `k` for each condition. For search data, use `--seed 0` and a new output directory. Keep the generator batch size at 32.

To generate all six final conditions from the supplied configurations:

```bash
python -m experiments_arch.experiment_1_controlled_bayes.reproduce --stage data
```

### 2. Hyperparameter selection and training

The hyperparameter search uses only Schedule-free AdamW, with learning rates in the range **0.0005–0.004** (grid: `0.0005, 0.001, 0.002, 0.004`). Run both architectures with training seeds `0,1,2` for `d=10, k∈{1,5,10}` and `d=20, k∈{1,10,20}`. Use 10,000 maximum steps for `d=10` and 20,000 for `d=20`.

Training defaults match the paper: width 96, four heads, MLP width 192, batch size 32, and five (`1d`) or three (`2d`) layers. Validation uses 3,200 tasks every 500 steps; stop after five checks without improvement and restore the best checkpoint.

**Search:** train each learning rate with three training seeds on **data seed 0**, then score its 32,000 selection tasks (`test.h5`). Example for one candidate:

```bash
python -m experiments_arch.experiment_1_controlled_bayes.train \
  --dataset-manifest paper/data/d10_k1_seed0/manifest.json \
  --architecture 1d --learning-rate 0.001 \
  --max-steps 10000 --seed 0 --device cuda \
  --output-dir paper/search/d10_k1/1d/sf_0.001/seed_0

python -m experiments_arch.experiment_1_controlled_bayes.evaluate \
  --dataset-manifest paper/data/d10_k1_seed0/manifest.json \
  --checkpoint paper/search/d10_k1/1d/sf_0.001/seed_0/checkpoint.pt \
  --output-dir paper/search/d10_k1/1d/sf_0.001/seed_0/evaluation --device cuda
```

Repeat for all four learning rates and seeds `0,1,2`, using separate directories. For each `(d,k,architecture)`, select the lowest mean MSE across seeds:

```bash
python -m experiments_arch.experiment_1_controlled_bayes.select \
  --metrics paper/search/d10_k1/1d/*/seed_*/evaluation/metrics.json \
  --metric mse --mode min --output paper/search/d10_k1/1d/selected.json
```

**Final training:** use the selected learning rate (also supplied with the release), retrain from scratch on **data seed 1**, and repeat for both architectures, all six conditions, and seeds `0,1,2`:

```bash
python -m experiments_arch.experiment_1_controlled_bayes.train \
  --dataset-manifest paper/data/d10_k1_seed1/manifest.json \
  --architecture 1d --learning-rate 0.001 \
  --max-steps 10000 --seed 0 --device cuda \
  --output-dir public_checkpoints/retrained/d10_k1/arch_1d/seed_0
```

This example uses the selected configuration for `d=10,k=1,1d` and saves new weights in `public_checkpoints/retrained/`; released weights are in `public_checkpoints/final/`. For final MSE, run `evaluate` as above with the seed-1 manifest and final checkpoint. The final test set is never used for hyperparameter selection.

### 3. Experiment materials and measurements

Using the generated data and supplied Nano checkpoints, run:

```bash
python -m experiments_arch.experiment_1_controlled_bayes.reproduce \
  --stage measurements --device cuda \
  --tabpfn-checkpoint /path/to/tabpfn-v2-regressor.ckpt
```

Supply the frozen **TabPFN v2 regression checkpoint** for the pretrained comparison. This command generates 1,000 projection queries per task (seed 0), evaluates all 36 Nano checkpoints on the 32,000 test tasks, and runs source-message interventions for Nano `2d` seed 0 and TabPFN v2. It fits affine coefficients and saves per-task measurements under `paper/results/`.

Interventions use the same 128 tasks for both models: data seed 1, selection seed 0, and the first 128 indices of `numpy.random.default_rng(0).permutation(32000)`. All feature-attention layers and doses `0.05, 0.1, 0.2` are evaluated; row scopes are `both`, `context`, and `query` for `k=1`, and `both` otherwise. To generate this cohort alone:

```bash
python -m experiments_arch.experiment_1_controlled_bayes.prepare_materials cohort \
  --manifest public_checkpoints/configs/d10_k1_seed1.json --count 128 --seed 0 \
  --output-dir paper/cohorts/d10_k1
```

For separate stages, replace `--stage measurements` with `materials`, `baselines`, or `interventions`. Completed matching measurements are reused when rerunning a stage.

### 4. Summaries and figures

Compute the plotting inputs directly from the generated data and saved measurements, then render the figures:

```bash
python -m experiments_arch.experiment_1_controlled_bayes.paper_figures.summarize \
  --results-dir paper/results --output-dir paper/plot_data

python -m experiments_arch.experiment_1_controlled_bayes.paper_figures \
  --data-dir paper/plot_data --output-dir paper/figures --formats pdf png
```

The summaries include baseline errors, architecture gaps, OLS comparisons, intervention statistics, signed sensitivities, and response matrices. Intervention intervals use 2,000 generator-batch bootstrap draws (seed 20260919), shared across models, row scopes, layers, doses, and roles within each condition. No plotting-summary download is needed.

Use `--only baselines`, `--only interventions`, or `--only ols` on the plotting command for a subset. To run data generation, measurements, summaries, and plotting together, use `reproduce --stage all --device cuda --tabpfn-checkpoint /path/to/tabpfn-v2-regressor.ckpt`.

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).
