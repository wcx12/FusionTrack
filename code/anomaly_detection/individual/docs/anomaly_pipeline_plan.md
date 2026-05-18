# MTF-BA Anomaly Pipeline Plan

## Goal

Build the project in three layers:

1. Individual trajectory anomaly detection
2. Group anomaly detection
3. Fusion of individual and group scores

The first implementation target is the individual module based on the baseline
trajectory anomaly workflow, adapted to VT-Tiny-MOT. The group module is not
implemented yet, but its interfaces are defined here so later integration does
not require refactoring the data layer.

## Data Assumptions

Current project data comes from VT-Tiny-MOT and is already extracted into:

- `outputs/vt_tiny_mot_trajectories/observations_<split>.csv`
- `outputs/vt_tiny_mot_trajectories/trajectories_<split>.jsonl`

The baseline expects one sample to be one trajectory. For this project:

- one individual sample = one `(sequence, track_id)` trajectory
- one group sample = one `(sequence, frame window)` scene slice with all objects

## Shared ID Convention

Every object-level record uses:

- `sample_id = "{sequence}:{track_id}"`

This ID is the contract across:

- feature generation
- detector training/inference
- individual score export
- future group score export
- fusion

## Module Boundaries

### 1. Data Layer

Responsible for converting raw extracted observations into two views:

- Individual view:
  - keyed by `sample_id`
  - contains the full per-object trajectory
- Group view:
  - keyed by `(sequence, window_start, window_end)` or similar
  - contains all visible objects in the window

### 2. Individual Feature Layer

Transforms a single-object trajectory into several representations inspired by
the baseline:

- `route`
- `speed`
- `shape`

Recommended first version:

- `route`: relative center trajectory `(cx_t - cx_0, cy_t - cy_0)`
- `speed`: scalar per-step motion magnitude
- `shape`: normalized motion increments `(dx_norm, dy_norm)`

First milestone uses RGB only. Thermal and cross-modal signals are phase 2
extensions.

### 3. Individual Detector Layer

Each representation is trained as an independent detector. The baseline LSTM
autoencoder can be reused here with adapted input pipelines.

Each detector outputs one anomaly score per `sample_id`.

### 4. Individual Ensemble Layer

Combines detector scores into one `individual_score`.

Planned order:

1. simple mean/max fusion
2. baseline-style complementary ensemble

### 5. Group Interface Layer

No model yet. The interface should expose enough data for later group anomaly
reasoning:

- sequence id
- frame ids
- object ids / sample ids
- category ids / names
- positions
- velocities
- bounding boxes
- modality visibility masks

### 6. Fusion Layer

Future fusion combines:

- `individual_score`
- `group_score`

Fusion should operate on object-aligned records via `sample_id`.

## Recommended Implementation Phases

### Phase 1: Protocol and shared schema

- Define sample IDs and output schemas
- Keep the current extraction scripts as raw-data entrypoints
- Add common metadata/score structures for reuse

### Phase 2: Individual trajectory view

- Convert `observations_<split>.csv` into object-centric trajectories
- Preserve metadata:
  - `sequence`
  - `track_id`
  - `category_id`
  - `category_name`
  - `fps`
  - visibility per modality

Deliverable:

- object-centric trajectory dataset keyed by `sample_id`

### Phase 3: Individual feature generation

- Build `route`, `speed`, and `shape` representations from object trajectories
- Split into `train/val/test`
- Export in a format that the adapted baseline training code can consume

Deliverable:

- representation datasets for each split and detector

### Phase 4: Individual detector adaptation

- Reuse baseline model/training logic
- Replace baseline-specific preprocessing assumptions
- Train one detector per representation
- Export detector-level scores

Deliverable:

- detector score files keyed by `sample_id`

### Phase 5: Individual ensemble

- Fuse detector scores
- Export final individual anomaly score

Deliverable:

- `individual_scores` file with component scores and fused score

### Phase 6: Group interface stabilization

- Keep scene/window view in a reusable format
- Define future group score output contract

Deliverable:

- ready-to-consume group input/output interfaces

## Output Schema

Each object-level score record should look like:

```json
{
  "sample_id": "DJI_0022_1:4700000",
  "sequence": "DJI_0022_1",
  "track_id": "4700000",
  "source": "individual",
  "score": 0.83,
  "component_scores": {
    "route": 0.71,
    "speed": 0.65,
    "shape": 0.90
  }
}
```

Future group-level object-aligned score records should use the same fields, but
set `source = "group"`.

## Immediate Next Step

Implement the individual trajectory view:

1. read `observations_<split>.csv`
2. group by `(sequence, track_id)`
3. produce standardized object-centric trajectory records keyed by `sample_id`
4. keep the existing sequence/window loader untouched for future group modeling
