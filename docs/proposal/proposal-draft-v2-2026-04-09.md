# Proposal Draft v2

## Title

Benchmarking antibody-aware protein representations for antibody-antigen interaction and developability prediction

## Team

Tina Zhang, Zichen Deng, Edward Li, and Peter Lin combine backgrounds in computational biology, computer science, and cancer therapeutics, which positions the team well to connect modern protein foundation models with biologically meaningful antibody evaluation tasks.

## Project Idea

Recent generative and representation-learning methods have significantly advanced computational antibody modeling. In particular, RFdiffusion-style methods demonstrate that antibody design can be approached with modern generative modeling, while interaction-aware language models such as MINT suggest that explicitly modeling interacting protein context can improve downstream prediction tasks. However, reliable computational prioritization of useful antibody candidates remains difficult, especially when both interaction quality and developability-related properties must be considered.

Our project will study whether pretrained antibody-aware or interaction-aware protein representations improve prediction quality on a biologically relevant antibody benchmark compared with simpler sequence-based baselines. Rather than attempting a full end-to-end therapeutic discovery pipeline, we will focus on a tractable supervised benchmark centered on antibody-antigen interaction or developability prediction, with a possible HER2-focused case study if the data support it.

## Execution Plan

We will first curate a benchmark dataset from public antibody resources and select one primary task with well-defined labels, such as binding-related prediction or a developability-associated prediction task. We will compare a small set of models: a simple baseline, a general protein language model embedding baseline, and one stronger antibody-aware or interaction-aware representation-learning approach if feasible. We will use standard train-validation-test splits and evaluate performance with task-appropriate metrics such as AUROC, AUPRC, Pearson or Spearman correlation, depending on the benchmark.

If the benchmark experiments progress well, we will extend the project with a limited target-specific candidate-ranking analysis rather than a full de novo generation pipeline. This will allow us to keep the main project coherent and feasible while still exploring the practical value of modern generative-AI-inspired protein representations for computational biology.

## Expected Outcome

We expect that antibody-aware or interaction-aware pretrained representations will outperform simpler sequence baselines on at least one biologically meaningful downstream antibody prediction task. The main contribution of the project will be a careful empirical comparison, an analysis of where current models succeed or fail, and a discussion of what this implies for future computational antibody design workflows.

## References

- Bennett et al. Atomically accurate de novo design of antibodies with RFdiffusion. Nature. 2025.
- Learning the language of protein-protein interactions. Nature Communications. 2025.
- Learning the language of antibody hypervariability. PNAS. 2024.
