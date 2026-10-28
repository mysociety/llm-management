"""Resolve shared runtime owners from the application serving this request."""

from typing import Annotated

from fastapi import Depends, Request

from ..automatic.pipeline import AutomaticClassifier
from ..bootstrap import Application
from ..deployments.service import DeploymentService
from ..foi.backends import FOIResources
from ..foi.email_finder.service import FOIFinder
from ..sar.pipeline import SARClassifier


async def get_application(request: Request) -> Application:
    return request.app.state.application


DependencyApplication = Annotated[Application, Depends(get_application)]


async def get_service(application: DependencyApplication) -> DeploymentService:
    return application.deployments


async def get_foi_resources(application: DependencyApplication) -> FOIResources:
    return application.foi


async def get_foi_finder(application: DependencyApplication) -> FOIFinder:
    return application.foi_finder


async def get_sar_classifier(application: DependencyApplication) -> SARClassifier:
    return application.sar


async def get_automatic_classifier(
    application: DependencyApplication,
) -> AutomaticClassifier:
    return application.automatic


DependencyService = Annotated[DeploymentService, Depends(get_service)]
DependencyFOIResources = Annotated[FOIResources, Depends(get_foi_resources)]
DependencyFOIFinder = Annotated[FOIFinder, Depends(get_foi_finder)]
DependencySARClassifier = Annotated[SARClassifier, Depends(get_sar_classifier)]
DependencyAutomaticClassifier = Annotated[
    AutomaticClassifier, Depends(get_automatic_classifier)
]
