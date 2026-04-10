# Team Roles and Milestone Plan

## Goal

This plan is designed for self-selection. It does not assign named owners. Instead, it breaks the milestone into balanced work packages that team members can claim.

The main objective for the milestone is to produce a clear two-page document with:

- a narrowed project statement
- a concrete FLAb-based benchmark setup
- detailed methods
- preliminary results
- a short discussion of next experiments

## Milestone Requirements

From the course requirements, the milestone should include:

1. a descriptive title
2. a brief abstract
3. an introduction
4. methods in greater detail than the proposal
5. preliminary results
6. tentative interpretation and discussion

Figures and tables are encouraged.

## Workload Principle

The team should split work by balanced deliverable packages, not by named people and not by isolated paper sections. Each package below is intended to be similar in effort if everyone contributes responsibly.

## Milestone Work Packages

### Package A: Project Framing and Writing Core

Scope:

- finalize the milestone title
- write the abstract
- write the introduction
- ensure that the project claim is realistic and narrow

Expected output:

- title
- abstract
- introduction draft

Workload level:

- moderate

Notes:

- this package should keep the framing aligned with the actual benchmark work
- avoid overclaiming full antibody discovery or generation

### Package B: Dataset and Methods

Scope:

- document the FLAb task choices
- summarize the dataset schema
- describe preprocessing and fold generation
- define the task type and evaluation metrics

Expected output:

- methods draft for data and evaluation
- one clear description of the Koenig binding and expression tasks

Workload level:

- moderate to high

Notes:

- this package should stay concrete
- every dataset mentioned should correspond to a real file and label

### Package C: Baseline Experiments and Results

Scope:

- run the first baseline experiments
- aggregate results
- prepare one milestone-ready results table
- help draft the results section

Expected output:

- preliminary metrics
- one results table
- short results text

Workload level:

- high

Notes:

- this is the heaviest technical package
- if needed, two people can share this package while another package is slightly merged

### Package D: Interpretation, Polish, and Assembly

Scope:

- write the discussion
- describe remaining experiments
- make sure the document fits the required template
- unify formatting, captions, and notation

Expected output:

- discussion draft
- future work paragraph
- final assembled milestone PDF draft

Workload level:

- moderate

Notes:

- this package is lighter technically but important for coherence
- the person or pair taking this package should be detail-oriented

## Recommended Self-Selection Rule

If there are four team members, the cleanest arrangement is:

- one person claims Package A
- one person claims Package B
- one person claims Package C
- one person claims Package D

If the team feels that Package C is too heavy, use this adjustment:

- one person claims Package A
- one person claims Package B
- two people split Packages C and D, with one focused on experiments and one focused on interpretation plus assembly

That version is usually the most balanced in practice.

## Concrete Milestone Deliverables

Before writing freeze, the team should have:

- exact task names selected
- exact metrics selected
- fold generation confirmed
- at least one baseline result
- ideally one stronger comparison result

Before final assembly, the team should have:

- one results table
- one short methods section
- one short discussion section
- one person responsible for merging everything into the template

## What the Milestone Should Emphasize

The milestone should emphasize:

- benchmark design
- dataset choice
- model comparison
- early evidence

It should not emphasize:

- HER2 as the project center
- full generative design claims
- large future ambitions without results

## After the Milestone

After the milestone, the team should keep using balanced work packages.

### Post-Milestone Package 1: Benchmark Expansion

- add the next FLAb benchmark task
- verify robustness across tasks

### Post-Milestone Package 2: Stronger Model Integration

- add one better representation model
- compare against the baseline cleanly

### Post-Milestone Package 3: Analysis and Interpretation

- perform error analysis
- identify which biological or methodological patterns explain performance

### Post-Milestone Package 4: Final Paper and Presentation

- draft figures and tables
- organize paper writing
- prepare the presentation storyline

## Post-Milestone Priority Order

1. stabilize the Koenig binding and expression results
2. add `warszawski2019_d44_Kd`
3. compare one stronger model against the initial baseline
4. do error analysis
5. only then consider a target-specific extension

## Bottom Line

For the milestone, the team should self-assign four balanced packages: framing, methods, experiments, and assembly. The experiments package is the heaviest, so if workload balance becomes an issue, split that package first.
