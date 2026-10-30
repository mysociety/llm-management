"""
HTTP contracts, routers and access to the application serving each request.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from ..bootstrap import Application


async def get_application(request: Request) -> Application:
    """
    Return the runtime assembled for this HTTP app without constructing resources.

    Routes select their dependencies from this typed container. Tests can override
    this getter to substitute an application; feature functions still receive
    their specific dependencies explicitly.
    """
    return request.app.state.application


DependencyApplication = Annotated[Application, Depends(get_application)]
