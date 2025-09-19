---
title: Intro to DP with OpenDP
author: Michael Shoemate, Privacy Architect for the OpenDP Library
---

## Outline

- Introduction
  - What is Differential Privacy?
- DP Wizard
  - Live demo
- OpenDP Library
  - Context API
  - Tabular Data
  - Framework API
  - Programming Framework
  - Plugins

## What is Differential Privacy?

_Differential privacy_ is a definition of privacy that mathematically quantifies 
how much a data release can reveal about any one person in the data.

## What is Differential Privacy?

_Differential privacy_ is a definition of privacy that mathematically quantifies 
how much a data release can reveal about any one person in the data.

These data releases may be:

- descriptive statistics like counts, sums, means, quantiles
- descriptive statistics under grouping, like contingency tables, cross-tabulations
- machine learning models
- synthetic data
- statistical models like regression, decision trees, PCA, etc

## What is Differential Privacy?

_Differential privacy_ is a definition of privacy that mathematically quantifies 
how much a data release can reveal about any one person in the data.

"Differential" in the sense that the technology obscures the difference between data releases on data sets that differ by one individual.

## What is Differential Privacy?

_Differential privacy_ is a definition of privacy that mathematically quantifies 
how much a data release can reveal about any one person in the data.

Why we use it:

- Meaningful
  - Protects every individual
  - Robust against any adversary
  - Immune to auxiliary information
- Quantifiable
  - Bounded "privacy loss"
  - Mathematically rigorous
  - Allows for interactivity

## What is Differential Privacy?

_Differential privacy_ is a definition of privacy that mathematically quantifies 
how much a data release can reveal about any one person in the data.

How it is applied:

- add a "small" amount of random noise into statistical computations
- careful application of data pre-processing when appropriate
- and occasionally, completely replace with unexpected algorithms

When implemented correctly, an estimator/algorithm/mechanism can be shown to "satisfy" differential privacy.

## What is Differential Privacy?

_Differential privacy_ is a definition of privacy that mathematically quantifies 
how much a data release can reveal about any one person in the data.

Example count algorithm:

1. compute number of records in dataset
2. add noise from the laplace distribution

The "privacy loss" of sharing the outcome of this algorithm 
depends on how many records an individual can contribute to the data,
and how much noise is added.

## DP Wizard

https://mccalluc-dp-wizard.share.connect.posit.cloud/

## OpenDP Library Setup

First install and import the required dependencies:

```shell
%pip install opendp[mbi]
```

```python
>>> import opendp.prelude as dp
>>> import polars as pl
>>>
>>> # The OpenDP team is working to vet algorithms used in these slides.
>>> # Until that is complete we need to opt-in to use these features.
>>> dp.enable_features("contrib")

```

## Example Dataset
[Labour Force Survey microdata](https://ec.europa.eu/eurostat/web/microdata/public-microdata/labour-force-survey) released by Eurostat
surveys working hours of individuals in the European Union collected on a quarterly cadence

```python
>>> lfs_data = pl.scan_csv(
...     dp.examples.get_france_lfs_path(), 
...     ignore_errors=True)

```

Relevant columns:

- SEX: 1 | 2
- AGE: integer
- ILOSTAT: Labour status during the reference week
- HWUSUAL: Number of Hours Per Week Usually Worked
- QUARTER
- YEAR


## Context API

```python
>>> context = dp.Context.compositor(
...    data=lfs_data,
...    privacy_unit=dp.unit_of(contributions=36),
...    privacy_loss=dp.loss_of(epsilon=1.0),
... )

```

- privacy unit: the greatest influence an individual may have on your dataset
    - what the "protected change" is
- privacy loss: the greatest loss in privacy any one individual may suffer in your dataset
    - how much advantage an adversary can gain from the release

## Context API

```python
>>> context = dp.Context.compositor(
...    data=lfs_data,
...    privacy_unit=dp.unit_of(contributions=36),
...    privacy_loss=dp.loss_of(epsilon=1.0),
... )

```

- privacy unit: pessimistic upper bound based on the number of quarters in the data
    - 4 quarters across 9 years
- privacy loss: for central-DP with aggregate statistics, `epsilon = 1` is the recommended default

## Context API

```python
>>> context = dp.Context.compositor(
...    data=lfs_data,
...    privacy_unit=dp.unit_of(contributions=36),
...    privacy_loss=dp.loss_of(epsilon=1.0),
... )

```

Intuition: Differential privacy is a specialized constraint on the stability of a function.

Functional stability is a property that, given a bounded change in the input to a function, 
you know the output has a bounded change.

Here, we set a bound on how much the input can change (the privacy unit),
and a bound on how much the output can change (the privacy loss).

## Context API

```python
>>> context = dp.Context.compositor(
...    data=lfs_data,
...    privacy_unit=dp.unit_of(contributions=36),
...    privacy_loss=dp.loss_of(epsilon=1.0),
... )

```

From here on, aim to create APIs similar to existing data science libraries...

## DP Count

A simple count of the number of individuals in the data.

```python
>>> query_num_responses = context.query(epsilon=1 / 5).select(dp.len())

```

## DP Count

A simple count of the number of individuals in the data.

```python
>>> query_num_responses = context.query(epsilon=1 / 5).select(dp.len())

```

With statistical significance `alpha = 0.05`, or equivalently with 95% confidence...

```python
>>> query_num_responses.summarize(alpha=0.05)
shape: (1, 5)
┌────────┬──────────────┬─────────────────┬───────┬────────────┐
│ column ┆ aggregate    ┆ distribution    ┆ scale ┆ accuracy   │
│ ---    ┆ ---          ┆ ---             ┆ ---   ┆ ---        │
│ str    ┆ str          ┆ str             ┆ f64   ┆ f64        │
╞════════╪══════════════╪═════════════════╪═══════╪════════════╡
│ len    ┆ Frame Length ┆ Integer Laplace ┆ 180.0 ┆ 539.731115 │
└────────┴──────────────┴─────────────────┴───────┴────────────┘

```

...the true estimate will differ from the noisy estimate by at most 540.

## DP Count

A simple count of the number of individuals in the data.

```python
>>> query_num_responses = context.query(epsilon=1 / 5).select(dp.len())

```

Assuming the privacy-utility tradeoff is acceptable, submit the query:
```python
>>> query_num_responses.release().collect() # doctest: +SKIP
shape: (1, 1)
┌─────────┐
│ len     │
│ ---     │
│ u32     │
╞═════════╡
│ 3811915 │
└─────────┘

```

This consumes a portion of the privacy budget.


## DP Sum

OpenDP properly accounts for all sources of numerical imprecision and potential overflow 
in both the implementation of the function, as well as in the privacy analysis.

```python
>>> context = dp.Context.compositor(
...    data=lfs_data,
...    privacy_unit=dp.unit_of(contributions=36),
...    privacy_loss=dp.loss_of(epsilon=1.0),
...    margins=[
...         dp.polars.Margin(
...             # the length of the data is no greater than
...             #    average quarterly survey size (public)
...             #    * number of quarters (public)
...             max_length=150_000 * 36
...             # Remember to only use public information
...         ),
...     ],
... )

```

- Privacy guarantee is valid only if the data has at most `max_length` records.
- Recommended best practice is to only ever create one Context


## DP Sum

```python
>>> query_work_hours = (
...     context.query(epsilon=1 / 5)
...     .filter(pl.col.HWUSUAL != 99.0)
...     .select(pl.col.HWUSUAL.cast(int).dp.sum(bounds=(0, 80)))
... )

```

Preprocessing:

- filter
- cast
- imputation
- clipping

## DP Sum

```python
>>> query_work_hours = (
...     context.query(epsilon=1 / 5)
...     .filter(pl.col.HWUSUAL != 99.0)
...     .select(pl.col.HWUSUAL.cast(int).dp.sum(bounds=(0, 80)))
... )

```

With 95% confidence, the DP sum release will differ from the exact sum by at most 43,000.
```python
>>> query_work_hours.summarize(alpha=0.05)
shape: (1, 5)
┌─────────┬───────────┬─────────────────┬─────────┬─────────────┐
│ column  ┆ aggregate ┆ distribution    ┆ scale   ┆ accuracy    │
│ ---     ┆ ---       ┆ ---             ┆ ---     ┆ ---         │
│ str     ┆ str       ┆ str             ┆ f64     ┆ f64         │
╞═════════╪═══════════╪═════════════════╪═════════╪═════════════╡
│ HWUSUAL ┆ Sum       ┆ Integer Laplace ┆ 14400.0 ┆ 43139.04473 │
└─────────┴───────────┴─────────────────┴─────────┴─────────────┘

```

## DP Mean

The mean independently estimates a sum and a count:

```python
>>> query_work_hours = (
...     context.query(epsilon=1 / 5)
...     .filter(pl.col.HWUSUAL != 99.0)
...     .select(pl.col.HWUSUAL.cast(int).dp.mean(bounds=(0, 80)))
... )

>>> query_work_hours.summarize(alpha=0.05)
shape: (2, 5)
┌─────────┬───────────┬─────────────────┬─────────┬──────────────┐
│ column  ┆ aggregate ┆ distribution    ┆ scale   ┆ accuracy     │
│ ---     ┆ ---       ┆ ---             ┆ ---     ┆ ---          │
│ str     ┆ str       ┆ str             ┆ f64     ┆ f64          │
╞═════════╪═══════════╪═════════════════╪═════════╪══════════════╡
│ HWUSUAL ┆ Sum       ┆ Integer Laplace ┆ 28800.0 ┆ 86277.589474 │
│ HWUSUAL ┆ Length    ┆ Integer Laplace ┆ 360.0   ┆ 1078.963271  │
└─────────┴───────────┴─────────────────┴─────────┴──────────────┘

```

The privacy budget allocated to this query is partitioned amongst the two statistics.

## DP Mean

You can equivalently estimate the sum and count separately:

```python
>>> query_work_hours = (
...     context.query(epsilon=1 / 5)
...     .filter(pl.col.HWUSUAL != 99.0)
...     .select(
...        pl.col.HWUSUAL.cast(int).dp.sum(bounds=(0, 80)), 
...        dp.len()
...     )
... )

>>> query_work_hours.summarize(alpha=0.05)
shape: (2, 5)
┌─────────┬──────────────┬─────────────────┬─────────┬──────────────┐
│ column  ┆ aggregate    ┆ distribution    ┆ scale   ┆ accuracy     │
│ ---     ┆ ---          ┆ ---             ┆ ---     ┆ ---          │
│ str     ┆ str          ┆ str             ┆ f64     ┆ f64          │
╞═════════╪══════════════╪═════════════════╪═════════╪══════════════╡
│ HWUSUAL ┆ Sum          ┆ Integer Laplace ┆ 28800.0 ┆ 86277.589474 │
│ len     ┆ Frame Length ┆ Integer Laplace ┆ 360.0   ┆ 1078.963271  │
└─────────┴──────────────┴─────────────────┴─────────┴──────────────┘

```

## DP Mean (Bounded-DP)

Under "Bounded-DP", the number of records in the dataset is not considered public information.

```python
>>> # apply domain descriptors (margins) to preprocessed data
>>> context_bounded_dp = dp.Context.compositor(
...     # apply some preprocessing outside of OpenDP
...      # drops "Not applicable" values
...     data=lfs_data.filter(pl.col.HWUSUAL != 99),
...     privacy_unit=dp.unit_of(contributions=36),
...     privacy_loss=dp.loss_of(epsilon=1.0),
...     margins=[
...         dp.polars.Margin(
...             max_length=150_000 * 36,
...             # ADDITIONAL CODE STARTING HERE
...             # don't protect the total number of records (bounded-DP)
...             invariant="lengths",
...         ),
...     ],
... )

>>> query_mean_work_hours = (
...     context_bounded_dp.query(epsilon=1 / 5)
...     .select(pl.col.HWUSUAL.cast(int).dp.mean(bounds=(0, 80)))
... )

>>> query_mean_work_hours.summarize(alpha=0.05)
shape: (2, 5)
┌─────────┬───────────┬─────────────────┬────────┬──────────────┐
│ column  ┆ aggregate ┆ distribution    ┆ scale  ┆ accuracy     │
│ ---     ┆ ---       ┆ ---             ┆ ---    ┆ ---          │
│ str     ┆ str       ┆ str             ┆ f64    ┆ f64          │
╞═════════╪═══════════╪═════════════════╪════════╪══════════════╡
│ HWUSUAL ┆ Sum       ┆ Integer Laplace ┆ 7200.0 ┆ 21569.772352 │
│ HWUSUAL ┆ Length    ┆ Integer Laplace ┆ 0.0    ┆ NaN          │
└─────────┴───────────┴─────────────────┴────────┴──────────────┘

>>> query_mean_work_hours.release().collect() # doctest: +SKIP
shape: (1, 1)
┌───────────┐
│ HWUSUAL   │
│ ---       │
│ f64       │
╞═══════════╡
│ 37.645122 │
└───────────┘

```

## DP Quantile

```python
>>> candidates = list(range(20, 60))
>>> query_multi_quantiles = (
...     context.query(epsilon=1 / 5)
...     .filter(pl.col.HWUSUAL != 99.0)
...     .select(
...         pl.col.HWUSUAL.dp.quantile(a, candidates).alias(f"{a}-Quantile")
...         for a in [0.25, 0.5, 0.75]
...     )
... )
>>> query_multi_quantiles.summarize()
shape: (3, 4)
┌───────────────┬───────────────┬────────────────┬────────┐
│ column        ┆ aggregate     ┆ distribution   ┆ scale  │
│ ---           ┆ ---           ┆ ---            ┆ ---    │
│ str           ┆ str           ┆ str            ┆ f64    │
╞═══════════════╪═══════════════╪════════════════╪════════╡
│ 0.25-Quantile ┆ 0.25-Quantile ┆ ExponentialMin ┆ 3240.0 │
│ 0.5-Quantile  ┆ 0.5-Quantile  ┆ ExponentialMin ┆ 1080.0 │
│ 0.75-Quantile ┆ 0.75-Quantile ┆ ExponentialMin ┆ 3240.0 │
└───────────────┴───────────────┴────────────────┴────────┘

```

## DP Grouping

- Stable Keys
    - Privacy: Costs an extra "delta" privacy parameter
    - Utility: Must release counts, discards keys with small counts
- Explicit Keys
    - Privacy: No extra "delta" privacy parameter
    - Utility: Joins against the explicit key set, imputes missing keys
- Invariant Keys
    - Privacy: Weakens the integrity of the privacy guarantee
    - Utility: Releases all keys in the clear

## DP Grouping with Stable Keys

```python
>>> context = dp.Context.compositor(
...     data=lfs_data,
...     privacy_unit=dp.unit_of(contributions=36),
...     privacy_loss=dp.loss_of(epsilon=1 / 5, delta=1e-7),
...     margins=[dp.polars.Margin(max_length=150_000 * 36)],
... )

>>> query_age_ilostat = (
...     context.query(epsilon=1 / 5, delta=1e-7)
...     .group_by("AGE", "ILOSTAT")
...     .agg(dp.len())
... )

>>> query_age_ilostat.summarize()
shape: (1, 5)
┌────────┬──────────────┬─────────────────┬───────┬───────────┐
│ column ┆ aggregate    ┆ distribution    ┆ scale ┆ threshold │
│ ---    ┆ ---          ┆ ---             ┆ ---   ┆ ---       │
│ str    ┆ str          ┆ str             ┆ f64   ┆ u32       │
╞════════╪══════════════╪═════════════════╪═══════╪═══════════╡
│ len    ┆ Frame Length ┆ Integer Laplace ┆ 180.0 ┆ 3458      │
└────────┴──────────────┴─────────────────┴───────┴───────────┘

>>> df = query_age_ilostat.release().collect()

```
<img src="images/stable_keys.png" alt="Stable Keys" width="800"/>


## DP Grouping with Explicit Keys

Reusing the key-set released in the previous query:

```python
>>> query_age_ilostat = (
...     context.query(epsilon=1 / 5)
...     .group_by("AGE", "ILOSTAT")
...     .agg(pl.col.HWUSUAL.dp.sum((0, 80)))
...     .with_keys(df["AGE", "ILOSTAT"])
... )

>>> query_age_ilostat.summarize()
shape: (1, 4)
┌─────────┬───────────┬───────────────┬──────────────┐
│ column  ┆ aggregate ┆ distribution  ┆ scale        │
│ ---     ┆ ---       ┆ ---           ┆ ---          │
│ str     ┆ str       ┆ str           ┆ f64          │
╞═════════╪═══════════╪═══════════════╪══════════════╡
│ HWUSUAL ┆ Sum       ┆ Float Laplace ┆ 14472.517992 │
└─────────┴───────────┴───────────────┴──────────────┘

```


## Pre-Processing

Stable transformations in `with_columns`:
```python
>>> query_hwusual_binned = (
...     context.query(epsilon=1 / 5)
...     # shadows the usual work hours "HWUSUAL" column with binned data
...     .with_columns(pl.col.HWUSUAL.cut(breaks=[0, 20, 40, 60, 80, 98]))
...     .group_by(pl.col.HWUSUAL)
...     .agg(dp.len())
... )

```

## Pre-Processing

Stable transformations in `group_by`:
```python
>>> query_hwusual_binned = (
...     context.query(epsilon=1 / 5)
...     .group_by(pl.col.HWUSUAL.cut(breaks=[0, 20, 40, 60, 80, 98]))
...     .agg(dp.len())
... )

```


## Pre-Processing

- Data Types: int, float, string, bool, categorical, enum, datetime, time
- Boolean logic, binary operations
- Cast, clip, cut, imputation, expression filtering, recoding
- Date parsing and temporal logic


## User Identifiers

- individuals may make an unbounded number of contributions
- all contributions from each user share the same identifier

```python
>>> # the PIDENT column contains individual identifiers
>>> # an individual may contribute data under at most 1 PIDENT identifier
>>> privacy_unit = dp.unit_of(
...     contributions=1, 
...     identifier=pl.col("PIDENT")
... )

>>> context = dp.Context.compositor(
...     data=lfs_data,
...     privacy_unit=privacy_unit,
...     privacy_loss=dp.loss_of(epsilon=1.0, delta=1e-8),
...     margins=[dp.polars.Margin(max_length=150_000 * 36)],
... )

>>> query = (
...     context.query(epsilon=1 / 5, delta=1e-8 / 5)
...     .filter(pl.col.HWUSUAL != 99)
...     .truncate_per_group(10)
...     .select(pl.col.HWUSUAL.cast(int).dp.mean((0, 80)))
... )
>>> query.summarize()
shape: (2, 4)
┌─────────┬───────────┬─────────────────┬────────┐
│ column  ┆ aggregate ┆ distribution    ┆ scale  │
│ ---     ┆ ---       ┆ ---             ┆ ---    │
│ str     ┆ str       ┆ str             ┆ f64    │
╞═════════╪═══════════╪═════════════════╪════════╡
│ HWUSUAL ┆ Sum       ┆ Integer Laplace ┆ 8000.0 │
│ HWUSUAL ┆ Length    ┆ Integer Laplace ┆ 100.0  │
└─────────┴───────────┴─────────────────┴────────┘

```

## Truncating Per-Group

```python
>>> query = (
...     context.query(epsilon=1 / 5, delta=1e-8 / 5)
...     .filter(pl.col.HWUSUAL != 99)
...     .group_by(pl.col.PIDENT) # truncation begins here
...     .agg(pl.col.HWUSUAL.mean()) # arbitrary expressions can be used here
...     .select(pl.col.HWUSUAL.cast(int).dp.mean((0, 80)))
... )
>>> query.summarize()
shape: (2, 4)
┌─────────┬───────────┬─────────────────┬───────┐
│ column  ┆ aggregate ┆ distribution    ┆ scale │
│ ---     ┆ ---       ┆ ---             ┆ ---   │
│ str     ┆ str       ┆ str             ┆ f64   │
╞═════════╪═══════════╪═════════════════╪═══════╡
│ HWUSUAL ┆ Sum       ┆ Integer Laplace ┆ 800.0 │
│ HWUSUAL ┆ Length    ┆ Integer Laplace ┆ 10.0  │
└─────────┴───────────┴─────────────────┴───────┘

```

## Truncating Contributed Groups
```python
>>> quarterly = [pl.col.QUARTER, pl.col.YEAR]
>>> query = (
...     context.query(epsilon=1 / 5, delta=1e-8 / 5)
...     .filter(pl.col.HWUSUAL != 99)
...     .truncate_per_group(1, by=quarterly)
...     .truncate_num_groups(10, by=quarterly)
...     .group_by(quarterly)
...     .agg(dp.len(), pl.col.HWUSUAL.cast(int).dp.sum((0, 80)))
... )
>>> query.summarize()
shape: (2, 5)
┌─────────┬──────────────┬─────────────────┬────────┬───────────┐
│ column  ┆ aggregate    ┆ distribution    ┆ scale  ┆ threshold │
│ ---     ┆ ---          ┆ ---             ┆ ---    ┆ ---       │
│ str     ┆ str          ┆ str             ┆ f64    ┆ u32       │
╞═════════╪══════════════╪═════════════════╪════════╪═══════════╡
│ len     ┆ Frame Length ┆ Integer Laplace ┆ 100.0  ┆ 2165      │
│ HWUSUAL ┆ Sum          ┆ Integer Laplace ┆ 8000.0 ┆ null      │
└─────────┴──────────────┴─────────────────┴────────┴───────────┘

```

## Zero-Concentrated DP (zCDP)

- a weaker definition of privacy than pure-DP, but stronger than approximate-DP
- using zCDP causes OpenDP to change the noise distribution

```python
>>> privacy_loss = dp.loss_of(rho=0.2, delta=1e-8)

>>> context = dp.Context.compositor(
...     data=lfs_data,
...     privacy_unit=dp.unit_of(contributions=1, identifier=pl.col("PIDENT")),
...     privacy_loss=privacy_loss,
...     margins=[dp.polars.Margin(max_length=150_000 * 36)],
... )

```

## Zero-Concentrated DP (zCDP)

Re-running the previous query, but this time under zCDP:

```python
>>> quarterly = [pl.col.QUARTER, pl.col.YEAR]
>>> query = (
...     context.query(rho=0.2 / 4, delta=1e-8 / 4)
...     .filter(pl.col.HWUSUAL != 99)
...     .truncate_per_group(1, by=quarterly)
...     .truncate_num_groups(10, by=quarterly)
...     .group_by(quarterly)
...     .agg(dp.len(), pl.col.HWUSUAL.cast(int).dp.sum((0, 80)))
... )
>>> query.summarize()
shape: (2, 5)
┌─────────┬──────────────┬──────────────────┬────────────┬───────────┐
│ column  ┆ aggregate    ┆ distribution     ┆ scale      ┆ threshold │
│ ---     ┆ ---          ┆ ---              ┆ ---        ┆ ---       │
│ str     ┆ str          ┆ str              ┆ f64        ┆ u32       │
╞═════════╪══════════════╪══════════════════╪════════════╪═══════════╡
│ len     ┆ Frame Length ┆ Integer Gaussian ┆ 14.142136  ┆ 89        │
│ HWUSUAL ┆ Sum          ┆ Integer Gaussian ┆ 1131.37085 ┆ null      │
└─────────┴──────────────┴──────────────────┴────────────┴───────────┘

```

## Laplace vs. Gaussian Noise

<img src="images/laplace-vs-gaussian.png" alt="Laplace vs Gaussian" width="80%"/>

- Gaussian noise preserves the normality assumption
- Gaussian noise cannot satisfy pure differential privacy
    - satisfies the weaker definition of approximate differential privacy or zCDP
- Gaussian noise affords greater utility (adds less overall noise) for a similar privacy guarantee when answering many queries
- Sensitivity calibration
    - Laplace noise scale is proportional to the $L_1$ distance
    - Gaussian noise scale is proportional to the $L_2$ distance

## Direct Use of Mechanisms

- The OpenDP Library is designed as a set of modular building blocks
- APIs to directly invoke lower-level mechanisms are available
- Can be used for research purposes or plugged into existing systems


## Laplace Mechanism: Context API

```python
>>> context = dp.Context.compositor(
...     data=[1., 4., 7.],
...     privacy_unit=dp.unit_of(l1=1.),
...     privacy_loss=dp.loss_of(epsilon=1.0),
...     domain=dp.vector_domain(dp.atom_domain(T=float, nan=False)),
... )

>>> query_lap = context.query(epsilon=1).laplace()
>>> query_lap.param()
1.0

>>> query_lap.resolve()
Measurement(
    input_domain   = VectorDomain(AtomDomain(T=f64)),
    input_metric   = L1Distance(f64),
    output_measure = MaxDivergence)

>>> query_lap.release() # doctest: +SKIP
[0.7077237243471377, 4.827747780105709, 5.908376498290111]
```

## Laplace Mechanism: Framework API

```python
>>> m_lap = dp.m.make_laplace(
...     input_domain=dp.vector_domain(dp.atom_domain(T=float, nan=False)),
...     input_metric=dp.l1_distance(T=float),
...     scale=1.0
... )

>>> m_lap
Measurement(
    input_domain   = VectorDomain(AtomDomain(T=f64)),
    input_metric   = L1Distance(f64),
    output_measure = MaxDivergence)

>>> m_lap.map(d_in=1.0)
1.0

>>> m_lap([1., 4., 7.]) # doctest: +SKIP
[1.8894052083047814, 5.543928329362439, 6.961532708302391]

```

## Randomized Response: Context API

```python
>>> context = dp.Context.compositor(
...     data=True,
...     privacy_unit=dp.unit_of(local=True),
...     privacy_loss=dp.loss_of(epsilon=1.0),
... )

>>> query_rr = context.query(epsilon=1.0).randomized_response_bool()
>>> query_rr.param()
0.7310585786300048

>>> query_rr.resolve()
Measurement(
    input_domain   = AtomDomain(T=bool),
    input_metric   = DiscreteDistance(),
    output_measure = MaxDivergence)

>>> query_rr.release() # doctest: +SKIP
[0.7077237243471377, 4.827747780105709, 5.908376498290111]
```

## Randomized Response: Framework API

```python
>>> m_rr = dp.m.make_randomized_response_bool(
...     prob=0.7310585786300048
... )

>>> m_rr
Measurement(
    input_domain   = AtomDomain(T=bool),
    input_metric   = DiscreteDistance(),
    output_measure = MaxDivergence)

>>> m_rr.map(d_in=1)
0.9999999999999997

>>> m_rr(True) # doctest: +SKIP
True

```

## Randomized Response: Proof

- Proofs demonstrate that the privacy guarantees hold.
- [Proof document](https://docs.opendp.org/en/v0.13.0/proofs/rust/src/measurements/randomized_response/make_randomized_response_bool.pdf)

## Extensibility: Programming Framework

- [A Framework to Understand DP](https://docs.opendp.org/en/stable/theory/a-framework-to-understand-dp.html)

| transformation | measurement    | odometer       |
|----------------|----------------|----------------|
| input_domain   | input_domain   | input_domain   |
| input_metric   | input_metric   | input_metric   |
| function       | function       | function       |
| output_domain  |                |                |
| output_metric  | output_measure | output_measure |
| stability_map  | privacy_map    |                |

## Extensibility: Plugins

- Extend the library with your own algorithms implemented in Python or Rust
- Python and Rust algorithms can be mixed together
- Examples:
    - [Constant Mechanism](https://docs.opendp.org/en/stable/api/user-guide/plugins/measurement.html)
    - [Theil-Sen Regression](https://docs.opendp.org/en/stable/api/user-guide/plugins/theil-sen-regression.html)
    - Hands-on Differential Privacy

<img src="images/Hands-on-Differential-Privacy.png" alt="Book Cover" width="20%"/>

## Extensibility: Supported Languages

| language | functionality                                          |
|----------|--------------------------------------------------------|
| Rust     | core mechanisms, Polars, native transformations        |
| Python   | Rust functionality, Context API, Plugins, scikit-learn |
| R        | Rust functionality (except Polars)                     |
| C        | Rust functionality                                     |

## Conclusion

- Questions?
- Join our [Slack](https://join.slack.com/t/opendp/shared_invite/zt-1aca9bm7k-hG7olKz6CiGm8htI2lxE8w)
- [opendp.org](opendp.org)

<img src="images/opendp-website.png" alt="OpenDP Website" width="100%"/>