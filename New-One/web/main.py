from __future__ import annotations

import sys
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import AppConfig
from src.detection_settings import (
    SETTING_LABELS,
    DetectionSettings,
    DetectionSettingsStore,
)
from src.ppe_items import (
    PPE_ITEM_SETTING_DESCRIPTIONS,
    PPE_ITEM_SETTING_LABELS,
    decode_missing_ppe,
    format_missing_ppe_list,
)
from web.auth import get_session_user
from web.db import OBJECT_CLASS_LABELS, VIOLATION_LABELS, VisionDatabase
from web.camera_zone_demo import CameraZoneDemoProcessor
from web.gate_demo import GateDemoProcessor
from web.live_preview import LivePreviewManager
from web.safety_demo import SafetyDemoProcessor
from web.vehicle_db import VehicleDatabase

WEB_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))

SETTING_DESCRIPTIONS = {
    "pathway_block": "Detect static objects blocking pathway zones.",
    "mobile_usage": "Detect phone use anywhere in the camera view except allowed mobile zones.",
    "ppe_violation": "Detect persons missing required PPE anywhere in the camera view.",
    "fire_detection": "Detect and classify fire types (structural fire, welding, electrical spark).",
    "smoke_detection": "Detect smoke anywhere in the camera view.",
    "object_fall": "Detect falling objects or persons from downward motion anywhere in the camera view.",
    "near_miss": "Detect close proximity between a person and a moving hazard (forklift, vehicle, etc.).",
    "oil_spillage": "Detect oil spills and leaks anywhere in the camera view.",
    "wet_floor": "Detect wet floors, puddles, and slip hazards anywhere in the camera view.",
    "low_visibility": "Detect fog, haze, glare, dust, and poor scene clarity across the camera view.",
}


def create_app(config_path: str | None = None) -> FastAPI:
    config = AppConfig.load(config_path)
    db = VisionDatabase(config)
    vehicle_db = VehicleDatabase(config)
    settings_store = DetectionSettingsStore(config)
    settings_store.ensure_exists(config)
    live_preview = LivePreviewManager()
    gate_demo = GateDemoProcessor(config)
    safety_demo = SafetyDemoProcessor(config)
    camera_zone_demo = CameraZoneDemoProcessor(config)
    drawer_html_path = ROOT / "tools" / "pathway_drawer.html"
    app = FastAPI(title="NeoEHS AI Vision")

    PUBLIC_PATHS = {"/login", "/favicon.ico"}

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        path = request.url.path
        if path.startswith("/static") or path in PUBLIC_PATHS:
            return await call_next(request)
        if path == "/login" and request.method == "POST":
            return await call_next(request)
        if not get_session_user(request):
            return RedirectResponse("/login", status_code=303)
        return await call_next(request)

    # Must be added AFTER auth middleware so SessionMiddleware runs first on each request
    app.add_middleware(SessionMiddleware, secret_key=config.web_secret_key)
    app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")

    def require_login(request: Request) -> str:
        return get_session_user(request) or ""

    def fmt_duration(seconds: float) -> str:
        seconds = int(seconds)
        minutes, sec = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours}h {minutes}m {sec}s"
        if minutes:
            return f"{minutes}m {sec}s"
        return f"{sec}s"

    def fmt_object_class(object_class: str) -> str:
        if "," in object_class:
            return format_missing_ppe_list(decode_missing_ppe(object_class))
        return OBJECT_CLASS_LABELS.get(object_class, object_class)

    templates.env.filters["duration"] = fmt_duration
    templates.env.filters["object_class"] = fmt_object_class
    templates.env.globals["violation_labels"] = VIOLATION_LABELS
    templates.env.globals["object_class_labels"] = OBJECT_CLASS_LABELS

    @app.get("/", response_class=HTMLResponse)
    async def root(request: Request):
        if get_session_user(request):
            return RedirectResponse("/dashboard", status_code=303)
        return RedirectResponse("/login", status_code=303)

    @app.get("/login", response_class=HTMLResponse)
    async def login_page(request: Request, error: str | None = None):
        if get_session_user(request):
            return RedirectResponse("/dashboard", status_code=303)
        return templates.TemplateResponse(
            request,
            "login.html",
            {"error": error},
        )

    @app.post("/login")
    async def login_submit(
        request: Request,
        username: str = Form(...),
        password: str = Form(...),
    ):
        if username in config.web_users and config.web_users[username] == password:
            request.session["user"] = username
            return RedirectResponse("/dashboard", status_code=303)
        return templates.TemplateResponse(
            request,
            "login.html",
            {"error": "Invalid username or password"},
            status_code=401,
        )

    @app.get("/logout")
    async def logout(request: Request):
        request.session.clear()
        return RedirectResponse("/login", status_code=303)

    @app.get("/settings", response_class=HTMLResponse)
    async def settings_page(
        request: Request,
        saved: int | None = None,
        cleared: int | None = None,
        user: str = Depends(require_login),
    ):
        settings = settings_store.load(config)
        settings_dict = {
            "pathway_block": settings.pathway_block,
            "mobile_usage": settings.mobile_usage,
            "ppe_violation": settings.ppe_violation,
            "fire_detection": settings.fire_detection,
            "smoke_detection": settings.smoke_detection,
            "object_fall": settings.object_fall,
            "near_miss": settings.near_miss,
            "oil_spillage": settings.oil_spillage,
            "wet_floor": settings.wet_floor,
            "low_visibility": settings.low_visibility,
        }
        ppe_item_settings = {
            "ppe_helmet": settings.ppe_helmet,
            "ppe_vest": settings.ppe_vest,
            "ppe_gloves": settings.ppe_gloves,
            "ppe_boots": settings.ppe_boots,
        }
        return templates.TemplateResponse(
            request,
            "settings.html",
            {
                "user": user,
                "settings": settings_dict,
                "ppe_item_settings": ppe_item_settings,
                "setting_labels": SETTING_LABELS,
                "ppe_item_labels": PPE_ITEM_SETTING_LABELS,
                "setting_descriptions": SETTING_DESCRIPTIONS,
                "ppe_item_descriptions": PPE_ITEM_SETTING_DESCRIPTIONS,
                "saved": bool(saved),
                "cleared": bool(cleared),
                "clear_message": request.query_params.get("clear_message", ""),
                "active_page": "settings",
            },
        )

    @app.post("/settings/clear-data")
    async def settings_clear_data(
        request: Request,
        user: str = Depends(require_login),
    ):
        form = await request.form()
        if form.get("confirm") != "true":
            raise HTTPException(status_code=400, detail="Confirmation is required")

        if safety_demo.is_running() or camera_zone_demo.is_running() or gate_demo.is_running():
            raise HTTPException(
                status_code=409,
                detail="Stop demo processing before clearing data",
            )

        from urllib.parse import quote

        from src.data_reset import reset_demo_data

        keep_zones = form.get("keep_zones") == "true"
        result = reset_demo_data(config.project_root, keep_zones=keep_zones)
        message = quote(result.summary)
        return RedirectResponse(
            f"/settings?cleared=1&clear_message={message}",
            status_code=303,
        )

    @app.post("/settings")
    async def settings_submit(request: Request, user: str = Depends(require_login)):
        form = await request.form()
        settings = DetectionSettings(
            pathway_block=form.get("pathway_block") == "true",
            mobile_usage=form.get("mobile_usage") == "true",
            ppe_violation=form.get("ppe_violation") == "true",
            fire_detection=form.get("fire_detection") == "true",
            smoke_detection=form.get("smoke_detection") == "true",
            object_fall=form.get("object_fall") == "true",
            near_miss=form.get("near_miss") == "true",
            oil_spillage=form.get("oil_spillage") == "true",
            wet_floor=form.get("wet_floor") == "true",
            low_visibility=form.get("low_visibility") == "true",
            ppe_helmet=form.get("ppe_helmet") == "true",
            ppe_vest=form.get("ppe_vest") == "true",
            ppe_gloves=form.get("ppe_gloves") == "true",
            ppe_boots=form.get("ppe_boots") == "true",
        )
        settings_store.save(settings)
        return RedirectResponse("/settings?saved=1", status_code=303)

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard(request: Request, user: str = Depends(require_login)):
        stats = db.dashboard_stats()
        recent = db.list_observations(limit=10)
        cameras = live_preview.list_cameras(config)
        default_camera = config.camera_id
        if cameras and not any(camera["id"] == default_camera for camera in cameras):
            default_camera = cameras[0]["id"]
        return templates.TemplateResponse(
            request,
            "dashboard.html",
            {
                "user": user,
                "stats": stats,
                "recent": recent,
                "active_page": "dashboard",
                "live_preview_enabled": config.web_live_preview_enabled,
                "cameras": cameras,
                "default_camera_id": default_camera,
            },
        )

    @app.get("/api/cameras")
    async def api_cameras(user: str = Depends(require_login)):
        return {"cameras": live_preview.list_cameras(config)}

    @app.get("/api/live/status")
    async def api_live_status(
        camera_id: str | None = None,
        user: str = Depends(require_login),
    ):
        selected_camera = camera_id or config.camera_id
        return live_preview.get_status(selected_camera)

    @app.get("/api/live/stream")
    async def api_live_stream(
        request: Request,
        camera_id: str | None = None,
        user: str = Depends(require_login),
    ):
        if not config.web_live_preview_enabled:
            raise HTTPException(status_code=404, detail="Live preview is disabled")

        selected_camera = camera_id or config.camera_id
        try:
            config.get_camera(selected_camera)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        return StreamingResponse(
            live_preview.mjpeg_stream(config, selected_camera),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/observations", response_class=HTMLResponse)
    async def observations_list(
        request: Request,
        status: str | None = None,
        severity: str | None = None,
        violation_type: str | None = None,
        user: str = Depends(require_login),
    ):
        if violation_type and violation_type not in VIOLATION_LABELS:
            raise HTTPException(status_code=404, detail="Unknown violation type")
        if severity and severity not in {"low", "medium", "high"}:
            raise HTTPException(status_code=400, detail="Invalid severity")

        observations = db.list_observations(
            violation_type=violation_type,
            status=status,
            severity=severity,
            limit=200,
        )
        total = db.count_observations(
            violation_type=violation_type,
            status=status,
            severity=severity,
        )

        title_parts = ["All Observations"]
        if status:
            title_parts = [status.capitalize() + " Observations"]
        if severity:
            title_parts = [severity.capitalize() + " Severity"]
        if violation_type:
            title_parts.append(VIOLATION_LABELS[violation_type])

        return templates.TemplateResponse(
            request,
            "observations_list.html",
            {
                "user": user,
                "observations": observations,
                "total": total,
                "status_filter": status,
                "severity_filter": severity,
                "violation_type_filter": violation_type,
                "page_title": " · ".join(title_parts),
                "active_page": "observations",
            },
        )

    @app.get("/violations/{violation_type}", response_class=HTMLResponse)
    async def violations_list(
        request: Request,
        violation_type: str,
        status: str | None = None,
        user: str = Depends(require_login),
    ):
        if violation_type not in VIOLATION_LABELS:
            raise HTTPException(status_code=404, detail="Unknown violation type")

        observations = db.list_observations(
            violation_type=violation_type,
            status=status,
            limit=200,
        )
        total = db.count_observations(violation_type=violation_type)
        active = db.count_observations(violation_type=violation_type, status="active")
        resolved = db.count_observations(violation_type=violation_type, status="resolved")

        return templates.TemplateResponse(
            request,
            "violations.html",
            {
                "user": user,
                "violation_type": violation_type,
                "violation_label": VIOLATION_LABELS[violation_type],
                "observations": observations,
                "status_filter": status,
                "total": total,
                "active": active,
                "resolved": resolved,
                "active_page": violation_type,
            },
        )

    @app.get("/vehicles", response_class=HTMLResponse)
    async def vehicles_dashboard(request: Request, user: str = Depends(require_login)):
        stats = vehicle_db.stats()
        inside_vehicles = vehicle_db.list_inside()
        recent_events = vehicle_db.list_events(limit=25)
        return templates.TemplateResponse(
            request,
            "vehicles.html",
            {
                "user": user,
                "stats": stats,
                "inside_vehicles": inside_vehicles,
                "recent_events": recent_events,
                "gate_cameras": config.gate_cameras(),
                "active_page": "vehicles",
            },
        )

    @app.get("/vehicles/history", response_class=HTMLResponse)
    async def vehicles_history(request: Request, user: str = Depends(require_login)):
        events = vehicle_db.list_events(limit=500)
        return templates.TemplateResponse(
            request,
            "vehicle_history.html",
            {
                "user": user,
                "events": events,
                "active_page": "vehicles",
            },
        )

    @app.get("/vehicles/demo", response_class=HTMLResponse)
    async def vehicles_demo(request: Request, user: str = Depends(require_login)):
        gate_demo.refresh_uploads()
        status = gate_demo.status()
        return templates.TemplateResponse(
            request,
            "vehicles_demo.html",
            {
                "user": user,
                "demo_status": status,
                "active_page": "vehicles_demo",
            },
        )

    @app.post("/vehicles/demo/upload")
    async def vehicles_demo_upload(
        front_video: UploadFile | None = File(None),
        back_video: UploadFile | None = File(None),
        user: str = Depends(require_login),
    ):
        if not front_video and not back_video:
            raise HTTPException(status_code=400, detail="Select at least one video to upload")

        try:
            if front_video and front_video.filename:
                data = await front_video.read()
                if data:
                    gate_demo.save_upload("gate_in", front_video.filename, data)
            if back_video and back_video.filename:
                data = await back_video.read()
                if data:
                    gate_demo.save_upload("gate_out", back_video.filename, data)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return RedirectResponse("/vehicles/demo?uploaded=1", status_code=303)

    @app.post("/vehicles/demo/run")
    async def vehicles_demo_run(user: str = Depends(require_login)):
        if not config.gate_vehicle_enabled:
            raise HTTPException(status_code=400, detail="Gate vehicle detection is disabled in config")

        try:
            gate_demo.start()
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return RedirectResponse("/vehicles/demo?started=1", status_code=303)

    @app.post("/vehicles/demo/clear")
    async def vehicles_demo_clear(user: str = Depends(require_login)):
        try:
            gate_demo.clear_videos()
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return RedirectResponse("/vehicles/demo?cleared=1", status_code=303)

    @app.get("/api/vehicles/demo/status")
    async def vehicles_demo_status(user: str = Depends(require_login)):
        return gate_demo.status()

    @app.get("/demo", response_class=HTMLResponse)
    async def safety_demo_page(request: Request, user: str = Depends(require_login)):
        safety_demo.refresh_uploads()
        status = safety_demo.status()
        return templates.TemplateResponse(
            request,
            "safety_demo.html",
            {
                "user": user,
                "demo_status": status,
                "active_page": "safety_demo",
            },
        )

    @app.post("/demo/upload")
    async def safety_demo_upload(
        fire_video: UploadFile | None = File(None),
        object_fall_video: UploadFile | None = File(None),
        near_miss_video: UploadFile | None = File(None),
        user: str = Depends(require_login),
    ):
        if not fire_video and not object_fall_video and not near_miss_video:
            raise HTTPException(status_code=400, detail="Select at least one video to upload")

        try:
            if fire_video and fire_video.filename:
                data = await fire_video.read()
                if data:
                    safety_demo.save_upload("fire_detection", fire_video.filename, data)
            if object_fall_video and object_fall_video.filename:
                data = await object_fall_video.read()
                if data:
                    safety_demo.save_upload("object_fall", object_fall_video.filename, data)
            if near_miss_video and near_miss_video.filename:
                data = await near_miss_video.read()
                if data:
                    safety_demo.save_upload("near_miss", near_miss_video.filename, data)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return RedirectResponse("/demo?uploaded=1", status_code=303)

    @app.post("/demo/run")
    async def safety_demo_run(user: str = Depends(require_login)):
        try:
            safety_demo.start()
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return RedirectResponse("/demo?started=1", status_code=303)

    @app.post("/demo/clear")
    async def safety_demo_clear(user: str = Depends(require_login)):
        try:
            safety_demo.clear_videos()
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return RedirectResponse("/demo?cleared=1", status_code=303)

    @app.get("/api/demo/status")
    async def safety_demo_status(user: str = Depends(require_login)):
        return safety_demo.status()

    @app.get("/camera-demo", response_class=HTMLResponse)
    async def camera_zone_demo_page(request: Request, user: str = Depends(require_login)):
        camera_zone_demo.refresh()
        status = camera_zone_demo.status()
        return templates.TemplateResponse(
            request,
            "camera_zone_demo.html",
            {
                "user": user,
                "demo_status": status,
                "violation_labels": VIOLATION_LABELS,
                "active_page": "camera_zone_demo",
            },
        )

    @app.post("/camera-demo/{camera_id}/frame")
    async def camera_zone_demo_upload_frame(
        camera_id: str,
        frame_image: UploadFile = File(...),
        user: str = Depends(require_login),
    ):
        try:
            data = await frame_image.read()
            camera_zone_demo.save_frame(camera_id, frame_image.filename or "frame.jpg", data)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return RedirectResponse("/camera-demo?uploaded=1", status_code=303)

    @app.get("/camera-demo/frame/{camera_id}")
    async def camera_zone_demo_frame(camera_id: str, user: str = Depends(require_login)):
        path = camera_zone_demo.store.frame_path(camera_id)
        if not path:
            raise HTTPException(status_code=404, detail="Reference frame not found")
        return FileResponse(path)

    @app.get("/camera-demo/drawer/{camera_id}", response_class=HTMLResponse)
    async def camera_zone_demo_drawer(
        camera_id: str,
        request: Request,
        user: str = Depends(require_login),
    ):
        frame_path = camera_zone_demo.store.frame_path(camera_id)
        if not frame_path:
            raise HTTPException(
                status_code=400,
                detail="Upload a video or capture a drawing frame before defining zones",
            )
        if not drawer_html_path.is_file():
            raise HTTPException(status_code=500, detail="Pathway drawer template missing")

        zone_type = request.query_params.get("type", "pathway")
        frame_url = f"/camera-demo/frame/{camera_id}?v={int(frame_path.stat().st_mtime)}"
        save_url = f"/camera-demo/zones/{camera_id}/save"
        html = drawer_html_path.read_text(encoding="utf-8")
        html = html.replace(
            'const frameUrl = params.get("frame") || "frame.jpg";',
            f'const frameUrl = {frame_url!r};',
        ).replace(
            'const saveUrl = params.get("saveUrl") || "/save";',
            f'const saveUrl = {save_url!r};',
        ).replace(
            'const zoneType = params.get("type") || "pathway";',
            f'const zoneType = {zone_type!r};',
        )
        return HTMLResponse(html)

    @app.post("/camera-demo/zones/{camera_id}/save")
    async def camera_zone_demo_save_zone(
        camera_id: str,
        request: Request,
        user: str = Depends(require_login),
    ):
        payload = await request.json()
        try:
            result = camera_zone_demo.save_zone(camera_id, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return result

    @app.post("/camera-demo/{camera_id}/video")
    async def camera_zone_demo_upload_video(
        camera_id: str,
        demo_video: UploadFile = File(...),
        user: str = Depends(require_login),
    ):
        try:
            data = await demo_video.read()
            result = camera_zone_demo.save_video(
                camera_id, demo_video.filename or "demo.mp4", data
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        query = "uploaded=1&frame_from_video=1"
        if result.get("zones_cleared"):
            query += "&zones_cleared=1"
        return RedirectResponse(f"/camera-demo?{query}", status_code=303)

    @app.post("/camera-demo/{camera_id}/video-frame")
    async def camera_zone_demo_capture_video_frame(
        camera_id: str,
        time_sec: float = Form(0.0),
        user: str = Depends(require_login),
    ):
        try:
            result = camera_zone_demo.capture_video_frame(camera_id, time_sec=time_sec)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        query = "frame_from_video=1"
        if result.get("zones_cleared"):
            query += "&zones_cleared=1"
        return RedirectResponse(f"/camera-demo?{query}", status_code=303)

    @app.post("/camera-demo/{camera_id}/clear")
    async def camera_zone_demo_clear_camera(
        camera_id: str,
        user: str = Depends(require_login),
    ):
        try:
            camera_zone_demo.clear_camera(camera_id)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return RedirectResponse("/camera-demo?cleared=1", status_code=303)

    @app.post("/camera-demo/run")
    async def camera_zone_demo_run(user: str = Depends(require_login)):
        try:
            camera_zone_demo.start()
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return RedirectResponse("/camera-demo?started=1", status_code=303)

    @app.post("/camera-demo/stop")
    async def camera_zone_demo_stop(user: str = Depends(require_login)):
        camera_zone_demo.stop()
        return RedirectResponse("/camera-demo", status_code=303)

    @app.get("/api/camera-demo/status")
    async def camera_zone_demo_status(user: str = Depends(require_login)):
        return camera_zone_demo.status()

    @app.get("/camera-demo/review/stream")
    async def camera_zone_demo_review_stream(user: str = Depends(require_login)):
        return StreamingResponse(
            camera_zone_demo.review_mjpeg_stream(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/vehicles/{plate_number}", response_class=HTMLResponse)
    async def vehicle_detail(
        request: Request,
        plate_number: str,
        user: str = Depends(require_login),
    ):
        events = vehicle_db.list_events(limit=200, plate_number=plate_number.upper())
        inside = vehicle_db.list_inside()
        current_visit = next((v for v in inside if v.plate_number == plate_number.upper()), None)
        completed_durations = [
            event.duration_seconds
            for event in events
            if event.direction == "out" and event.duration_seconds is not None
        ]
        avg_duration = (
            sum(completed_durations) / len(completed_durations) if completed_durations else 0.0
        )
        total_entries = sum(1 for event in events if event.direction == "in")
        return templates.TemplateResponse(
            request,
            "vehicle_detail.html",
            {
                "user": user,
                "plate_number": plate_number.upper(),
                "events": events,
                "is_inside": current_visit is not None,
                "current_visit": current_visit,
                "total_entries": total_entries,
                "avg_duration_sec": avg_duration,
                "active_page": "vehicles",
            },
        )

    @app.get("/vehicle-images/{event_id}")
    async def vehicle_event_image(event_id: str, user: str = Depends(require_login)):
        event = vehicle_db.get_event(event_id)
        if not event:
            raise HTTPException(status_code=404, detail="Vehicle event not found")
        path = vehicle_db.resolve_image_path(event.image_path)
        if not path:
            raise HTTPException(status_code=404, detail="Vehicle image not found")
        return FileResponse(path)

    @app.get("/observations/{observation_id}", response_class=HTMLResponse)
    async def observation_detail(
        request: Request,
        observation_id: str,
        user: str = Depends(require_login),
    ):
        observation = db.get_observation(observation_id)
        if not observation:
            raise HTTPException(status_code=404, detail="Observation not found")

        image_url = None
        if observation.image_path:
            image_url = f"/images/{observation.observation_id}"

        return templates.TemplateResponse(
            request,
            "observation_detail.html",
            {
                "user": user,
                "obs": observation,
                "violation_label": VIOLATION_LABELS.get(
                    observation.violation_type, observation.violation_type
                ),
                "image_url": image_url,
                "active_page": observation.violation_type,
            },
        )

    @app.get("/images/{observation_id}")
    async def observation_image(observation_id: str):
        observation = db.get_observation(observation_id)
        if not observation or not observation.image_path:
            raise HTTPException(status_code=404, detail="Image not found")

        path = db.resolve_image_path(observation.image_path)
        if not path:
            raise HTTPException(status_code=404, detail="Image file missing on disk")
        return FileResponse(path)

    @app.on_event("startup")
    async def startup_live_preview() -> None:
        if config.web_live_preview_enabled:
            live_preview.warmup(config)

    return app


def run_server(config_path: str | None = None, host: str | None = None, port: int | None = None) -> None:
    import argparse
    import logging

    import uvicorn

    from src.config import AppConfig

    parser = argparse.ArgumentParser(description="CCTV AI Vision Dashboard")
    parser.add_argument("--config", default=config_path, help="Path to config.yaml")
    parser.add_argument("--host", default=host)
    parser.add_argument("--port", type=int, default=port)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    app = create_app(args.config)
    config = AppConfig.load(args.config)
    bind_host = args.host or config.web_host
    bind_port = args.port or config.web_port

    print(f"NeoEHS AI Vision dashboard: http://{bind_host}:{bind_port}/")
    uvicorn.run(app, host=bind_host, port=bind_port)


if __name__ == "__main__":
    run_server()
