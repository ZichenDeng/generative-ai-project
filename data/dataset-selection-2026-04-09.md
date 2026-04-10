# Dataset Selection Memo

## Decision

The recommended primary dataset for this project is `FLAb`, with a task definition chosen from the benchmark subsets already used in recent antibody model evaluation. The recommended fallback strategy is to use a narrower benchmark derived from publicly labeled antibody tasks already used by MINT or related work, rather than attempting to build a custom HER2-centered dataset from scratch.

## Why FLAb Should Be the Primary Benchmark

FLAb is currently the cleanest match to the project we should actually execute. The public repository describes FLAb as the largest publicly available therapeutic antibody dataset, with more than three million antibody assay data points and explicit partitions by therapeutic property, including expression, thermostability, immunogenicity, aggregation, polyreactivity, binding affinity, and pharmacokinetics. This is exactly the kind of dataset structure we need for a benchmarked downstream prediction project rather than a vague discovery narrative.

More importantly, the recent MINT paper already uses FLAb for antibody property prediction. In the antibody-task section, the paper evaluates MINT against antibody-specialized baselines on four FLAb benchmark tasks and reports exact sample sizes for those tasks. This matters because it gives us a directly defensible benchmark setup instead of forcing us to invent one ourselves.

## Recommended Primary Task

The strongest primary task is a supervised prediction benchmark on one or two FLAb subsets, ideally chosen to balance biological relevance and implementation simplicity.

The safest first option is to use one `binding` task and one `expression` or other developability-related task from FLAb. This gives the project a clear story:

- one task tied more closely to functional interaction quality;
- one task tied more closely to practical developability.

That is a much better narrative than trying to claim end-to-end antibody design.

## Concrete FLAb Benchmark Evidence

The MINT paper states that its FLAb antibody benchmark includes three binding-energy datasets and one expression dataset, and reports the following sample sizes for independent antibody-antigen pairs:

- Binding: `n = 422`
- Binding: `n = 2048`
- Binding: `n = 4275`
- Expression: `n = 4275`

The paper further states that it followed a tenfold cross-validation setup with nested inner cross-validation for ridge regression on embeddings. That means the benchmark procedure is concrete enough for us to reproduce or adapt.

## What PLAbDab Is Good For

PLAbDab is valuable, but it should not be misrepresented as the primary labeled benchmark. The Oxford Protein Informatics Group describes PLAbDab as a self-updating repository containing over `150,000` paired antibody sequences derived from patents and academic papers, with links to primary sources and downloadable paired and unpaired sequence collections.

This makes PLAbDab useful for:

- antigen-specific sequence retrieval;
- literature-linked antibody lookup;
- building exploratory target-focused subsets;
- checking whether a target such as HER2 has enough sequence coverage to justify a case study.

It does **not** automatically solve the label problem for supervised prediction. The existence of paired sequences does not mean we have clean numeric labels for developability, binding, or function at the scale we need.

## What SAbDab Is Good For

SAbDab contains antibody structures from the PDB with consistent annotation, including heavy-light pairings, curated affinity data, and sequence annotations. This makes it useful for:

- structural lookup;
- structural sanity checks for a target-focused case study;
- deriving structure-aware context if we later add a small structural analysis.

SAbDab should be treated as a structural support resource, not the default primary benchmark.

## What Thera-SAbDab Is Good For

Thera-SAbDab tracks WHO-recognized antibody and nanobody therapeutics and links them to close or exact structural representatives in SAbDab. It is best used for:

- therapeutic target metadata;
- identifying clinically relevant antibody targets;
- checking whether known therapeutics exist for an antigen of interest;
- adding context to the discussion section or a focused case study.

It is not the primary benchmark for our prediction task.

## Why HER2 Should Not Be the Primary Dataset Right Now

HER2 is biologically plausible and easy to motivate, but it is not yet a safe choice for the main benchmark. Right now we do not have evidence that a sufficiently large, clean, public, labeled HER2-specific dataset exists for the exact supervised task we want. If we commit to HER2 too early, we risk turning the project into a data-collection exercise rather than a modeling study.

The correct role for HER2 is therefore:

- optional target-focused extension;
- discussion case study;
- candidate-ranking analysis only if data availability checks succeed.

It should not be the center of the primary benchmark until we verify label quality and sample size.

## Recommended Dataset Stack

### Primary

`FLAb`

Use this as the main supervised benchmark. Start with one binding-related subset and one developability-related subset.

### Fallback A

`MINT-linked antibody benchmark tasks outside FLAb`

If FLAb ingestion is slower than expected, fall back to one smaller benchmark already used in recent antibody representation papers, such as the SARS-CoV-2 antibody mutant affinity task discussed in the MINT antibody section.

### Fallback B

`PLAbDab + Thera-SAbDab + SAbDab`

Use this only for retrieval, target-specific annotation, exploratory subset construction, or qualitative analysis. Do not treat this combination as a ready-made supervised benchmark.

## Immediate Data Actions

1. Inspect the FLAb repository and choose exactly two candidate subsets for the first experiments.
2. Record the exact column schema, label type, and split policy for those subsets.
3. Define the primary metric family before any model training begins.
4. Use PLAbDab only to test whether a HER2 or cancer-target extension is realistic.
5. Keep structural databases as support resources unless the project explicitly becomes structure-aware.

## Bottom Line

The data layer should be built around `FLAb` as the primary benchmark, with `PLAbDab`, `SAbDab`, and `Thera-SAbDab` used as supporting resources rather than substitutes for labeled benchmark data.
