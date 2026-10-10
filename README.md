# critterframe

```text
  .   .
   \ /
   ⌐■-■
 ~/ ! \~   Image frames of critters into dataframes of traits
~| o:o |~
 | o:o |
/ \_:_/ \
```

critterframe is a flexible Python package built for large-scale organismal image processing, including easy-to-understand tools to:

1. **Ingest** occurrence data and images into the framework
```python
cf.ingest_occurrences("C:/my_project", "specimens.csv", id_col="specimen_id", image_url_col="image_url")
cf.download_images("C:/my_project")
```

2. **Segment** each organism out of its occurrence image
```python
cf.run_segments("C:/my_project", steps=[cf.segment(cf.groundedsam2())])
```

3. **Extract** and export traits from segmented organisms
```python
cf.run_metrics("C:/my_project", run_name="traits",
                transforms=[cf.remove_appendages()], metrics=[cf.mean_lightness()])
cf.export_metrics("C:/my_project", "traits.csv")
```

#### Plus core support for:
- Validation of segments & traits
- Advanced filtering
- Human annotation
- Training dataset export
- Importing segmentation models
- Managing organismal parts (i.e. head, thorax)
- Traits derived from stored traits: per occurrence (i.e. ratios) or per group (i.e. cluster assignments)
- Scale & color calibration

#### And support in extensions for: 
- Training a segmentation model using Segmentation Models Pytorch
- Training a bioencoder model
- Advanced customizable ingests from a GBIF DarwinCore archive

See [scripts/](https://github.com/jidec/critterframe/tree/main/scripts) for full example pipelines, described in the [Guide](https://jidec.github.io/critterframe/guide/)

## The framework

1. **One focal organism per image**

   > **Why:** A single organism is the natural unit for organismal image analysis and maps cleanly onto an occurrence. We see this as a worthy simplification that removes a complex layer of bookkeeping. Tools will eventually be provided to help convert multi-organism images into one-organism images for import.

2. **One canonical mask per organism or organism-part**

   > **Why:** Virtually all analyses need the best available representation of a biological part, not a growing collection of competing masks. Alternative segmentation approaches can be evaluated against reference masks before deciding which should become canonical.

3. **All derived values are metrics - whether traits, QC scores, or annotations**

   > **Why:** A common metric model lets the same machinery support biological measurements, quality control, validation, clustering results, embeddings, and lots more.

4. **Filtering is selection, not deletion**

   > **Why:** Filtering criteria are analytical decisions that may change as a project develops. We apply filters during export or post-critterframe analysis rather than removing data from the processing pipeline.

## The pipeline

1. **Small operations compose flexibly into complex pipelines**

   > **Why:** The same operations support both simple pipelines (i.e. foundational segmentation & thresholded color extraction) and complex ones (i.e. with custom part segmentation, filtering via outlier detection, metric learning embeddings, etc.).

2. **Idempotent, resumable, & evolvable**

   > **Why:** Large image datasets are expensive to process and continually evolve. Running the same recipe twice does no extra work and produces no extra data. Interrupted runs can continue. Incorporating new data is as simple as importing a new snapshot with the new data included.

3. **Full derivation provenance**

   > **Why:** Every mask and metric is linked to how it was produced and what inputs it depended on. Analyses are traceable & reproducible by default.

## Other key features

- Virtually every pipeline step leaves a visual report by default (`visualize=True`), making things easy to scrutinize
- Project folders are portable records with data & provenance ready for archiving alongside a publication
- Persistent named subsets make it easy to pass data around for validation, training, or subset-specific processing
- Metrics exports designed for easy analysis post-critterframe
- Multithreading/sharding for image downloading, segmentation, and metric runs

## Where to go next

| Page | Answers |
|---|---|
| [Concepts](https://jidec.github.io/critterframe/concepts/) | What a project folder holds, and what each term means. |
| [Guide](https://jidec.github.io/critterframe/guide/) | Which example pipeline to start from, how to validate one, and what the extensions add. |
| [Examples](https://jidec.github.io/critterframe/examples/simplest_full_pipeline/) | The pipeline scripts in full: the simplest one, the reusable workflows, and a whole project step by step. |
| [API Reference](https://jidec.github.io/critterframe/api/) | Every function a pipeline calls, and every module behind them. |
| [Internals](https://jidec.github.io/critterframe/internals/) | How the package is laid out, with import and call graphs, for reading or extending the code. |

## Installation
```
pip install "critterframe[torch] @ git+https://github.com/jidec/critterframe.git"
```

`[torch]` pulls in the ready-to-go SAM2/GroundedSAM2 torch models essential for many projects (as well as some deep-learning-based metrics), installing PyPI's default torch build for your platform if you don't yet have torch installed.

Installing [CUDA](https://developer.nvidia.com/cuda-downloads) beforehand to enable processing on GPU is recommended (GroundedSAM2 can fall back to CPU but becomes very slow).

If you need a torch version that fits a certain CUDA version (such as the version that exists on a computing cluster)
install [that torch version](https://pytorch.org/get-started/locally/) before installing `critterframe[torch]` (critterframe's `torch` dependency carries no version pin so it won't be overridden).

Projects segmenting with a custom model or working from existing masks (and not using deep-learning based metrics) don't need `torch`:
```
pip install "critterframe @ git+https://github.com/jidec/critterframe.git"
```

Claude Code was used to contribute code, documentation, & tests to this project (with every line examined by a human).

## License

[GPL-3.0](https://github.com/jidec/critterframe/blob/main/LICENSE)
