"""Access control for private brokerage data (single local user, shared-secret header)."""

from __future__ import annotations

import hmac
from typing import Annotated

from fastapi import Header, HTTPException, status

from marketops.config import get_settings

MIN_TOKEN_LEN = 16
TOKEN_HEADER = "X-MarketOps-Token"


def require_private_token(token: str | None) -> None:
    """Constant-time check of the local token; denies when none is configured."""
    expected = get_settings().agent_api_token
    if len(expected) < MIN_TOKEN_LEN:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "Private brokerage access needs MARKETOPS_AGENT_API_TOKEN (16+ characters).",
        )
    if not token or not hmac.compare_digest(token.encode(), expected.encode()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid access token.")


def private_access(
    x_marketops_token: Annotated[str | None, Header()] = None,
) -> None:
    """FastAPI dependency guarding every route that can return real account data."""
    require_private_token(x_marketops_token)
