from __future__ import annotations

from fastapi import HTTPException, Request, status


def get_session_user(request: Request) -> str | None:
    if "session" not in request.scope:
        return None
    return request.session.get("user")


def require_user(request: Request) -> str:
    user = get_session_user(request)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_303_SEE_OTHER,
            headers={"Location": "/login"},
        )
    return user
