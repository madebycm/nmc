# Astar Island — Complete Technical Specification

> **Purpose**: Self-contained reference for building the Astar Island solver. Read this file to understand everything needed without reading HTML docs.

## 1. Task Overview

Observe a black-box Norse civilisation simulator through limited viewports and predict the probability distribution of terrain types across the entire 40x40 map after 50 simulated years.

- **Task type**: Observation + probabilistic prediction
- **API base**: `https://api.ainm.no/astar-island`
- **Auth**: Bearer token header (`Authorization: Bearer <JWT>`) or cookie (`access_token=<JWT>`)
- **JWT source**: Log in at app.ainm.no → inspect cookies → grab `access_token`

## 2. Round Lifecycle

1. Admin creates a round → status `pending`
2. Round starts → status `active` (prediction window opens, ~165 minutes / 2h45m)
3. Window closes → status `scoring`
4. Scores computed → status `completed`

**Rounds are created at unknown times.** Must poll `GET /rounds` to detect active rounds.

### Scoring
- **Leaderboard score** = `max(round_score)` across all rounds (best single round)
- **Hot streak** = average of last 3 rounds (also tracked)
- **Later rounds may have higher `round_weight`** → weighted_score = round_score × round_weight

## 3. Per-Round Budget

| Resource | Limit |
|----------|-------|
| Simulation queries | **50 per round** (shared across all 5 seeds) |
| Submissions per seed | **Unlimited** (last one counts, overwrites previous) |
| Rate: simulate | 5 req/sec |
| Rate: submit | 2 req/sec |

## 4. Map & Terrain

40×40 grid. 8 internal terrain codes mapping to **6 prediction classes**:

| Internal Code | Terrain | Prediction Class Index | Notes |
|--------------|---------|----------------------|-------|
| 10 | Ocean | 0 (Empty) | Impassable, borders map. **STATIC** |
| 11 | Plains | 0 (Empty) | Flat buildable land |
| 0 | Empty | 0 (Empty) | Generic empty |
| 1 | Settlement | 1 | Active Norse settlement |
| 2 | Port | 2 | Coastal settlement with harbour |
| 3 | Ruin | 3 | Collapsed settlement |
| 4 | Forest | 4 | Provides food, mostly static but can reclaim ruins |
| 5 | Mountain | 5 | Impassable. **ALWAYS STATIC** |

**Key insight**: Ocean and Mountain NEVER change. Forest is mostly static. The interesting/scored cells are those that can transition between Empty/Settlement/Port/Ruin/Forest.

## 5. Simulation Mechanics (Hidden Rules)

Each of 50 years cycles through phases IN ORDER:

### Phase 1: Growth
- Settlements produce food from adjacent terrain
- High population → expand by founding new settlements on nearby land
- Coastal settlements develop ports
- Build longships for naval operations

### Phase 2: Conflict
- Settlements raid each other
- Longships extend raiding range
- Low-food settlements raid more aggressively
- Successful raids: loot resources, damage defender
- Conquered settlements can change faction (owner_id)

### Phase 3: Trade
- Ports within range trade if not at war
- Trade generates wealth + food for both
- Technology diffuses between trading partners

### Phase 4: Winter
- Variable severity each year
- All settlements lose food
- Settlements collapse from: starvation, sustained raids, harsh winters
- Collapsed settlement → Ruin, population disperses to nearby friendly settlements

### Phase 5: Environment
- Ruins can be reclaimed by nearby thriving settlements (new outpost inherits resources)
- Coastal ruins can be restored as ports
- Unclaimed ruins: overtaken by forest growth or fade to plains

### Settlement Properties (visible in simulate response)
- position (x, y)
- population, food, wealth, defense (float values)
- tech level (not directly shown?)
- has_port (bool)
- alive (bool)
- owner_id (faction allegiance)

**Initial states expose ONLY**: position, has_port, alive. Internal stats only visible through simulation queries.

## 6. API Endpoints — Complete Reference

### GET /astar-island/rounds (Public)
Returns all rounds. Find active ones with `status == "active"`.
```json
[{"id": "uuid", "round_number": 1, "status": "active", "map_width": 40, "map_height": 40,
  "prediction_window_minutes": 165, "started_at": "ISO8601", "closes_at": "ISO8601",
  "round_weight": 1, "created_at": "ISO8601"}]
```

### GET /astar-island/rounds/{round_id} (Public)
Returns initial map states for all 5 seeds.
```json
{"id": "uuid", "round_number": 1, "status": "active", "map_width": 40, "map_height": 40,
 "seeds_count": 5, "initial_states": [
   {"grid": [[10,10,...], ...], "settlements": [{"x": 5, "y": 12, "has_port": true, "alive": true}]}
 ]}
```
Grid values: 0=Empty, 1=Settlement, 2=Port, 3=Ruin, 4=Forest, 5=Mountain, 10=Ocean, 11=Plains

### GET /astar-island/budget (Team auth)
```json
{"round_id": "uuid", "queries_used": 23, "queries_max": 50, "active": true}
```

### POST /astar-island/simulate (Team auth) — costs 1 query
Request:
```json
{"round_id": "uuid", "seed_index": 0, "viewport_x": 10, "viewport_y": 5, "viewport_w": 15, "viewport_h": 15}
```
- seed_index: 0–4
- viewport_w/h: 5–15 (max 15×15)
- Each call uses DIFFERENT random sim_seed → different stochastic outcome

Response:
```json
{"grid": [[4, 11, 1, ...], ...],
 "settlements": [{"x":12,"y":7,"population":2.8,"food":0.4,"wealth":0.7,"defense":0.6,"has_port":true,"alive":true,"owner_id":3}],
 "viewport": {"x":10,"y":5,"w":15,"h":15}, "width": 40, "height": 40,
 "queries_used": 24, "queries_max": 50}
```
Grid is viewport_h × viewport_w only. Settlements only within viewport.

### POST /astar-island/submit (Team auth) — unlimited resubmissions
Request:
```json
{"round_id": "uuid", "seed_index": 0, "prediction": [[[p0,p1,p2,p3,p4,p5], ...], ...]}
```
- prediction shape: H×W×6 (`prediction[y][x][class]`)
- 6 classes: [Empty, Settlement, Port, Ruin, Forest, Mountain]
- Each cell's 6 probs must sum to 1.0 (±0.01 tolerance)
- All probs must be non-negative
- **CRITICAL**: Never use 0.0 probability — use floor of 0.01, then renormalize

Response: `{"status": "accepted", "round_id": "uuid", "seed_index": 0}`
Resubmitting overwrites previous. Only last submission counts.

### GET /astar-island/my-rounds (Team auth)
Like /rounds but with your scores, rank, budget.
Includes: round_score, seed_scores[], seeds_submitted, rank, total_teams, queries_used, initial_grid

### GET /astar-island/my-predictions/{round_id} (Team auth)
Your predictions with argmax_grid, confidence_grid, score, submitted_at.

### GET /astar-island/analysis/{round_id}/{seed_index} (Team auth)
Post-round only. Returns your prediction + ground_truth (both H×W×6 tensors) + score.
**Gold mine for learning** — use completed rounds to tune your model.

### GET /astar-island/leaderboard (Public)
```json
[{"team_name": "str", "weighted_score": 72.5, "rounds_participated": 3,
  "hot_streak_score": 78.1, "rank": 1, "is_verified": true}]
```

## 7. Scoring Formula

**Entropy-weighted KL divergence**:
```
KL(p || q) = Σ pᵢ × log(pᵢ / qᵢ)       per cell
entropy(cell) = -Σ pᵢ × log(pᵢ)          per cell

weighted_kl = Σ entropy(cell) × KL(truth[cell], pred[cell]) / Σ entropy(cell)
score = max(0, min(100, 100 × exp(-3 × weighted_kl)))
```

- 100 = perfect match, 0 = terrible
- Only **dynamic cells** (non-zero entropy) contribute — static ocean/mountain are excluded
- Higher entropy cells (more uncertain) weigh more
- **If ground truth has p>0 but your q=0 → KL = infinity → score destroyed**
- **ALWAYS floor at 0.01 and renormalize**

Per round: average of 5 seed scores. Missing seed = 0.

## 8. Query Strategy (50 queries for 40×40 map with 5 seeds)

Map is 40×40. Max viewport is 15×15. To cover full map need:
- 40/15 = 2.67 → 3 columns, 3 rows = **9 viewports per full map scan**
- With 50 queries across 5 seeds:
  - 10 queries per seed → 1 full map scan + 1 extra query per seed
  - OR focus queries on interesting areas (where settlements are)
  - OR fewer seeds with more observations per seed (risky — missing seed = 0)

**Recommended allocation**: 10 queries per seed
- 9 for full map coverage (3×3 grid of 15×15 + overlap on edges)
- 1 extra for re-observing high-uncertainty areas

**Better strategy**: Since simulation is stochastic, multiple observations of the SAME area give you empirical probability distributions. But 10 queries per seed barely covers the map once.

**Optimal viewport tiling for 40×40**:
- Viewports at (0,0), (13,0), (25,0) → covers columns 0-14, 13-27, 25-39
- Rows: (x,0), (x,13), (x,25) → covers rows 0-14, 13-27, 25-39
- = 9 viewports with overlap in columns 13-14 and 25-27

## 9. Prediction Strategy

### Baseline (uniform): scores ~1-5
Every cell gets [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]

### Better baseline (initial-state-informed):
- Ocean cells → [0.97, 0.005, 0.005, 0.005, 0.005, 0.01] (almost certainly stays ocean=Empty)
- Mountain cells → [0.01, 0.01, 0.01, 0.01, 0.01, 0.95] (stays mountain)
- Forest cells → [0.05, 0.02, 0.01, 0.02, 0.88, 0.02] (mostly stays forest)
- Settlement cells → spread probability across Settlement/Port/Ruin/Empty
- Plains cells → can become anything, but mostly stay Empty

### Observation-informed:
- Use simulation queries to see actual outcomes after 50 years
- With multiple observations of same region: empirical distribution
- With 10 queries per seed (1 observation per region): single sample → use as strong prior

### Advanced:
- Learn hidden parameters from observations across seeds (same params for all 5)
- Build a simulator approximation
- Use analysis endpoint on completed rounds to calibrate model
- Monte Carlo predictions using learned model

## 10. Infrastructure Plan

### Where to run: H100 Server (86.38.238.86)
- `/clade/astar/` — workspace (currently empty)
- Venv: `/clade/venv/` has requests, numpy, scipy, httpx
- Always-on, good for polling
- Python 3.12.3

### Architecture
```
/clade/astar/
├── solver.py          # Main solver: poll → observe → predict → submit
├── api.py             # API client wrapper
├── strategy.py        # Prediction logic
├── config.py          # Auth token, settings
└── run_loop.sh        # Cron/loop runner
```

### Auth Token
**NEEDED**: JWT token from app.ainm.no. Must be provided by user.

## 11. Critical Pitfalls

1. **Never probability 0.0** — use floor 0.01, renormalize
2. **Submit ALL 5 seeds** — missing seed = 0 score for that seed
3. **50 queries shared** — don't waste on one seed
4. **Stochastic outcomes** — same seed+viewport gives different results each call
5. **Prediction format**: `prediction[y][x][class]` — row-major (H×W×6)
6. **Class mapping**: Internal codes (0,1,2,3,4,5,10,11) → 6 prediction classes (0-5). Ocean(10) and Plains(11) both map to class 0.
7. **Rate limits**: 5 req/s simulate, 2 req/s submit
8. **Window timing**: ~165 min, must submit before `closes_at`
