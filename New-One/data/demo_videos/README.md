# Gate demo sample videos

| File | Role |
|------|------|
| `front_gate_demo.mp4` | Front gate entry (`gate_in`) |
| `back_gate_demo.mp4` | Back gate exit (`gate_out`) |

# Safety demo sample videos

| File | Violation type |
|------|----------------|
| `fire_demo.mp4` | Fire / smoke detection |
| `object_fall_demo.mp4` | Object fall detection |
| `near_miss_demo.mp4` | Near miss detection |

## Generate gate samples

```bash
python scripts/prepare_demo_videos.py
```

## Use in the web app

1. Start the web app: `python web_app.py`
2. **Safety Demo** (`/demo`) — fire, object fall, near miss
3. **Gate Demo** (`/vehicles/demo`) — front/back gate ANPR
4. Upload videos or place files in this folder, then click **Run Demo Detection**

Results appear on the **Dashboard** and violation pages.
