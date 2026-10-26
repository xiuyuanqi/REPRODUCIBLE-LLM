# BlockLock

`plot/` contains the complete pipeline for the current 2-by-3 experimental figure: CSV loading, aggregation, independent validation, plotting, and the paper caption.

## Setup and execution

Use Python 3.10 or later. From the directory containing this README, install the dependencies and rebuild the figures:

```bash
python3 -m venv plot/.venv
plot/.venv/bin/python -m pip install -r plot/requirements.txt
plot/.venv/bin/python plot/rebuild.py
```

The pipeline uses a headless plotting backend and runs on a CPU with the summary CSVs. CUDA, model weights, and raw logits are unnecessary. Internal script and output paths are resolved relative to the scripts, so the pipeline can also be invoked from another working directory using its full path. The validated environment uses Python 3.12, Matplotlib 3.11.2, and NumPy 2.5.4, with Matplotlib's bundled DejaVu Sans font.

For style changes using the existing derived data, redraw without refreshing the source CSVs:

```bash
plot/.venv/bin/python plot/draw_figure.py
```

## CSV sources

Configure the data root and primary CSV files in `plot/source_config.json`. The default `root` is `..`, resolved relative to that configuration file. This selects the repository directory containing this README, regardless of the current working directory.

The repository layout is:

```text
github/
|-- README.md
|-- A100/
|   `-- cell_summary.csv
|-- A6000/
|   |-- cell_summary01.csv
|   `-- cell_summary02.csv
|-- H100/
|   |-- cell_summary01.csv
|   `-- cell_summary02.csv
`-- plot/
    |-- source_config.json
    |-- rebuild.py
    `-- ...
```

The pipeline reads these five primary summaries, with paths relative to the repository directory:

| Hardware | CSV |
| --- | --- |
| A6000 | A6000/cell_summary01.csv |
| A6000 | A6000/cell_summary02.csv |
| A100 | A100/cell_summary.csv |
| H100 | H100/cell_summary02.csv |
| H100 | H100/cell_summary01.csv |

Keep this layout when moving or cloning the repository; the default configuration remains valid. To store the CSVs elsewhere, set `root` to an absolute path or a path relative to `plot/source_config.json`, then update the corresponding `files` entries and run `rebuild.py`. Source CSVs are read in place and remain unchanged. A missing configured input stops the rebuild. Each primary summary is read once; per-question exports and notebook checkpoints are excluded from the input list.

The plotting package uses the CSVs in this repository and the packages in `requirements.txt`.

## Files and responsibilities

| File | Purpose |
| --- | --- |
| rebuild.py | Run CSV loading, aggregation, validation, and plotting |
| load_sources.py | Read the five primary CSVs, check duplicate records, and record provenance and SHA-256 hashes |
| build_core_data.py | Select matched conditions and compute heatmaps, timings, and prefill throughput |
| build_data.py | Start data processing, add dataset-level A6000 results, and export panel (d) data |
| export_panel_d.py | Export panel (d) question-level medians, aggregate timings, and comparison factors |
| verify.py | Independently reread source CSVs and validate selected records, heatmaps, and dataset-level results |
| verify_panel_d.py | Independently validate panel (d)'s 336 question-level records, 12 summary rows, and nine factors |
| draw_figure.py | Draw the full 2-by-3 figure and standalone panel (d) |
| source_config.json | Configure the CSV root and source file list |
| requirements.txt | List Python dependencies |
| figure_caption.tex | Provide the paper caption and metric definitions |

## Outputs

The pipeline writes outputs to `plot/`:

- `blocklock_overview_2x3.png`, `.svg`, and `.pdf`: full six-panel figure.
- `panel_d_with_factors.png`, `.svg`, and `.pdf`: standalone panel (d).
- `figure_a6000_dataset_values.csv`: dataset-level A6000 agreement and decode timings.
- `panel_d_question_level.csv`: panel (d) question-level summary inputs, retaining the original millisecond units and source CSV paths.
- `panel_d_summary.csv`: panel (d) absolute timings in seconds and normalized timings.
- `panel_d_corun_factors.csv`: panel (d) comparison factors and their definitions.
- `source_data/`: generated source audits, selected records, and intermediate heatmap, timing, and throughput tables.
- `verification.json` and `panel_d_verification.json`: numerical validation results.

Running the pipeline again overwrites these generated files.

## Selection and aggregation

The analysis selects BF16, prompt mode, questions q1-q4, `cuDNN_disabled`, question-level timing, ten completed repetitions, and 45 run pairs per question. Each hardware/model/dataset/question/batch-size condition must contain all four methods. Missing matched conditions appear in gray. Repeatability is measured across repetitions at each fixed batch size B; cross-B equality requires a separate comparison.