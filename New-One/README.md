# CCTV Pathway Monitor

Detect static objects blocking defined pathway areas from an RTSP CCTV stream. Uses YOLO for detection, ByteTrack for tracking, and creates **one observation per obstruction** (no duplicate alerts for the same object in the same place).

## Pipeline

```
RTSP CCTV → Frame Capture → YOLO Detection → ByteTrack → Static Filter
    → Pathway Polygon Check → Create/Update Observation → Capture Image
```

Pathway check uses **bounding-box overlap**, not just the center point. If part of an object (e.g. chair legs) crosses into the pathway, it counts as blocking.

## Setup

```bash
cd cctv-pathway-monitor
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Edit `config.yaml` with your RTSP URL. Detection uses **YOLO-World** by default so boxes, bags, and machines on the pathway are found (standard COCO model has no "box" class). See `docs/DETECTABLE_OBJECTS.md` for the full list.

```yaml
rtsp:
  url: "rtsp://username:password@192.168.1.100:554/stream1"
```

## Define Pathway Areas

Draw two zone types on a CCTV frame:

| Type | Color | Purpose |
|------|-------|---------|
| `pathway` | Yellow | Main walkway — mobile banned outside magenta sub-zone |
| `mobile_usage` | Magenta | **Mobile allowed** sub-area (phone OK inside) |

```bash
python tools/define_pathway.py --source rtsp://user:pass@192.168.1.100/stream --type pathway
python tools/define_pathway.py --source snapshot.jpg --type mobile_usage
```

- **Click** polygon corners on the frame
- **Enter** to save
- **r** to reset points
- **q** to quit

Pathways are saved to `data/pathways.json`.

## Mobile Usage Violations

| Location | Phone use |
|----------|-----------|
| **Inside magenta zone** (`mobile_usage`) | Allowed — no violation |
| **On yellow pathway, outside magenta** | **Violation** |
| **Off pathway** (cubicle area etc.) | Ignored |

Logic:
1. Person must be on the **yellow pathway**
2. Person must be **outside** the **magenta mobile-allowed zone**
3. Person must be **using a phone** (phone detected near person)

Magenta zones are still checked for **object blocking** (box, chair, etc.) like the rest of the pathway.

Restart after changes: `python main.py --preview`

## Run Monitor

```bash
python main.py --preview
```

## Web Dashboard (Login + Drilldown)

Start the vision portal (reads from `vision_observation` table):

```bash
pip install -r requirements.txt
python web_app.py
```

Open **http://localhost:8080**

Default login (change in `config.yaml` → `web.users`):

| User | Password |
|------|----------|
| admin | admin123 |
| operator | operator123 |

### Portal navigation

1. **Login** → secure session
2. **Dashboard** → totals, severity breakdown, recent observations
3. **Violation menus** (sidebar):
   - **Pathway Block** → list of obstruction observations
   - **Mobile Usage** → list of phone violation observations
4. **Click any row** → full observation detail + captured image

Filter lists by **Active** / **Resolved** status on each violation page.

## How Deduplication Works

| Scenario | Behavior |
|----------|----------|
| New static object enters pathway | Creates observation + captures image |
| Same object stays 5+ minutes | Updates duration & summary on **same** observation |
| Object leaves pathway | Marks observation as `resolved` |
| ByteTrack reassigns track ID | Spatial matching (80px radius) prevents duplicate records |

Summaries change by duration:
- **< 1 min** — brief detection message
- **1–5 min** — ongoing obstruction
- **> 5 min** — persistent violation

## Output

All violations are stored in SQLite table **`vision_observation`**:

| Column | Description |
|--------|-------------|
| `observation_id` | Unique UUID |
| `camera_id` | Camera identifier from config |
| `zone_id` | Pathway zone ID |
| `zone_name` | Pathway zone name |
| `track_id` | ByteTrack ID (objects / persons) |
| `object_class` | e.g. chair, box, mobile_usage |
| `first_seen` | Violation start time (UTC) |
| `last_seen` | Last update time (UTC) |
| `duration_seconds` | How long violation has lasted |
| `severity` | `low` / `medium` / `high` (based on duration) |
| `status` | `active` or `resolved` |
| `summary` | AI-style text summary |
| `image_path` | Captured image on first detection |
| `violation_type` | `pathway_block` or `mobile_usage` |
| `confidence_score` | YOLO detection confidence |

Set camera ID in `config.yaml`:

```yaml
camera:
  id: "camera_6"
```

Query records:

```bash
python tools/list_observations.py --status active
python tools/list_observations.py --status resolved --limit 50
```

- **SQLite DB**: `data/observations.db`
- **Captured images**: `captures/`

## Config Highlights

| Setting | Default | Purpose |
|---------|---------|---------|
| `blocking_exclude_classes` | `person` | Excluded from blocking only; still used for mobile detection |
| `max_displacement_px` | 15 | Object must stay nearly still |
| `spatial_match_radius_px` | 80 | Same-place deduplication radius |
| `pathway.min_overlap_ratio` | 0.12 | Min % of bbox inside pathway (with min area) |
| `pathway.min_overlap_area_px` | 500 | Min pixels inside pathway (with min ratio) |
| `pathway.use_foot_point` | true | Bottom-center of bbox on pathway counts |
| `moderate_threshold_sec` | 300 | Summary text changes after 5 min |
