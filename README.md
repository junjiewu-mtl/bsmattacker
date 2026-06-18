# BSMAttacker

**A reproducible benchmark and attack-injection library for V2X Basic Safety Message (BSM) misbehavior detection.**

<!-- At release: add the Zenodo DOI badge here, e.g.
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) -->

BSMAttacker is the research artifact for the paper *"BSMAttacker: Simulation-to-Field
Evaluation of Safety-Application-Directed V2X Misbehavior Detection"* (IEEE Transactions
on Intelligent Transportation Systems, under review). It injects eleven
safety-application attacks into benign V2X trajectories, scores them with nine
misbehavior detectors trained on simulation and tested on real-world data, and replays
the surviving attacks against real safety-application warning events.

## Headline result

Detectors trained on the VeReMi Extension simulation and tested on five real-world V2X
datasets show a simulation-to-field transfer gap (mean AUROC 0.728 to 0.659) that is
concentrated in the unsupervised detectors. At a 95% recall operating point, only the two
supervised detectors keep their false-positive rate below 24% (Gradient Boosting 20.2%,
LSTM 23.6%), while every unsupervised detector exceeds 79%. On NYC CV Pilot replay, a
Constant Position Offset attack leaves forward-collision warning unraised for 84.6% of
events and intersection-movement assist for 57.4%.

## What is in this repository

| Path | Contents |
|---|---|
| [`bsm_attacker/`](bsm_attacker/) | The attack-injection library: a `BaseAttacker` interface, an `AttackPipeline`, the VeReMi Extension sensor-noise model, and the attack implementations. |
| [`benchmark/`](benchmark/) | The evaluation harness: the injection/feature engine ([`injection_engine.py`](benchmark/injection_engine.py)), the nine-detector zoo ([`detectors.py`](benchmark/detectors.py)), the plausibility check ([`plausibility_check.py`](benchmark/plausibility_check.py)), and the benchmark driver ([`run_benchmark.py`](benchmark/run_benchmark.py)). |
| [`benchmark/configs/benchmark.yaml`](benchmark/configs/benchmark.yaml) | Detector hyperparameters, the eleven-attack set, feature definitions, and corpus paths. |
| [`safety_replay/`](safety_replay/) | Forward-collision (FCW), intersection-movement (IMA), and emergency-brake (EEBL) replay on real warning events. |
| [`examples/`](examples/) | A smoke test and a small example trace. |
| [`tests/`](tests/) | A minimal functional test that injects an attack and checks the labels. |

## Installation

```bash
git clone https://github.com/junjiewu-mtl/bsmattacker.git
cd bsmattacker
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # pinned versions used for the published results
pip install -e .                  # the bsm_attacker, benchmark, and safety_replay packages
```

Python 3.10 or newer is required. The `requirements.txt` pins the exact versions used for
the published results; relax them to `>=` if you only need the attack library. Deep SVDD
additionally needs `deepod` (`pip install -e ".[deep-svdd]"`).

## Quick start

Verify the install and the attack roster:

```bash
python examples/smoke_test.py
```

Inject one attack on a labeled trace of your own:

```python
import pandas as pd
from bsm_attacker import AttackPipeline

df = pd.read_csv("examples/sample_trace.csv")          # your BSM trace
pipeline = AttackPipeline(random_seed=42)
attacked = pipeline.inject_single_attack(df, "constant_speed")
print(attacked["Is_Attack"].sum(), "rows flagged as attack")
```

The input frame must expose the kinematic BSM fields (position, speed, heading,
longitudinal/lateral acceleration, yaw rate), a `timestamp`, a `device_id`, and a
`Scenario_Label` column for the four context-aware attacks. See
[`bsm_attacker/README.md`](bsm_attacker/README.md) for the exact schema.

## The eleven safety-application attacks

Seven attacks are inherited from prior libraries and four context-aware attacks are
introduced in the paper. The context-aware attacks fire only inside the driving scenario
they target.

| Attack | Module / config key | Source | Scenario gate | Safety app |
|---|---|---|---|---|
| Constant Position Offset | `constant_position_offset` | VeReMi Ext. | none | FCW |
| Constant Speed | `constant_speed` | VeReMi Ext. | none | FCW |
| Eventual Stop | `eventual_stop` | VeReMi Ext. | none | FCW |
| Ghost Vehicle | `ghost_vehicle` | VASP | none | IMA |
| Opposite-Direction Heading | `opposite_heading` | VASP | none | IMA |
| Cross-Path Heading Spoof | `perpendicular_heading` | VASP | none | IMA |
| Constant Acceleration | `constant_acceleration` | VASP | none | EEBL |
| Slow Position Drift | `slow_position_drift` | this paper | Cruising | FCW |
| False Hard-Brake in Cruising | `false_deceleration_cruising` | this paper | Cruising | EEBL |
| Phantom Forward-Roll at Stop | `position_pullthrough_at_stop` | this paper | Stationary (waiting) | IMA |
| Frozen Heading at Turn | `heading_lock` | this paper | Turning | IMA |

The library also implements additional VeReMi- and VASP-family attacks; the eleven above
are the set evaluated in the paper.

## The nine detectors

| Detector | Key | Family |
|---|---|---|
| Gradient Boosting | `gb` | supervised |
| LSTM (one-layer, unidirectional) | `lstm` | supervised |
| Isolation Forest | `iforest` | isolation |
| Deep Isolation Forest | `dif` | isolation |
| Local Outlier Factor | `lof` | density |
| One-Class SVM | `ocsvm` | boundary |
| Deep SVDD | `deep_svdd` | boundary |
| Anomaly Transformer | `anomaly_transformer` | reconstruction |
| Graph Deviation Network | `gdn` | graph |

Detectors are defined in [`benchmark/detectors.py`](benchmark/detectors.py). The sequence
detectors (LSTM, Anomaly Transformer, GDN) use length-10 windows; the other six score one
message at a time. All nine train only on the synthetic corpus and are tested on the
real-world corpora without retraining.

## Reproducing the paper

```bash
# 1. Single-cell sanity check (one detector, one attack, synthetic corpus, CPU).
python -m benchmark.run_benchmark --corpus synthetic --models iforest --attacks constant_speed

# 2. Full synthetic matrix (nine detectors x eleven attacks).
python -m benchmark.run_benchmark --corpus synthetic

# 3. Cross-deployment evaluation on the real-world corpora.
python -m benchmark.run_benchmark --cross-eval

# 4. Safety-application replay (FCW / IMA / EEBL) on real warning events.
python -m safety_replay.replay --config safety_replay/configs/replay.yaml
```

Edit the `corpus:` paths in [`benchmark/configs/benchmark.yaml`](benchmark/configs/benchmark.yaml)
to point at your local copies of the datasets. The training split is vehicle-disjoint and
the random seed is fixed at 42.

## Data availability

The code in this repository reproduces the benchmark; the datasets are obtained separately.

| Dataset | Role | Access |
|---|---|---|
| VeReMi Extension | synthetic training corpus | Public (cite van der Heijden et al. 2018; Kamel et al. 2020). |
| Montreal | real-world (collected by the authors) | Released by the authors via a Zenodo DOI (de-identified; pending institutional ethics confirmation). |
| SPMD Ann Arbor | real-world | Public; obtain from the original source. |
| AMCD Tysons | real-world | Public; obtain from USDOT ITS DataHub. |
| V2AIX Aachen | real-world | Public; obtain from the original source. |
| BME Budapest | real-world | Public; obtain from the original source. |
| NYC CV Pilot events | safety-app replay | Public; obtain from USDOT ITS DataHub. |

Datasets are not redistributed here and are excluded by `.gitignore`. The F2MD rule-based
plausibility baseline (ETSI TR 103 460) is available from the authors on request.

## Citation

If you use BSMAttacker, please cite the paper and the software (see `CITATION.cff`):

```bibtex
@article{wu2026bsmattacker,
  author  = {Wu, Junjie and Fung, Benjamin C. M. and Yu, Hanbo and Stakhanova, Natalia},
  title   = {{BSMAttacker}: Simulation-to-Field Evaluation of Safety-Application-Directed
             {V2X} Misbehavior Detection},
  journal = {{IEEE} Transactions on Intelligent Transportation Systems},
  year    = {2026},
  note    = {Under review},
}
```

The Zenodo DOI for the exact software version is added to this section at release.

## Acknowledgments

This research is supported by the National Cybersecurity Consortium (NCC 2023-R8),
NSERC Discovery Grants (RGPIN-2024-04087), and the Canada Research Chairs Program
(CRC-2019-00041).

## License

Apache License 2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). The public datasets
retain their original licenses and are not redistributed here.

## Contact

Junjie Wu, McGill University (junjie.wu@mail.mcgill.ca). Issues and attack-suite
contributions are welcome through the GitHub issue tracker.
