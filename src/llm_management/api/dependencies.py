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


ApplicationDependency = Annotated[Application, Depends(get_application)]


async def get_service(application: ApplicationDependency) -> DeploymentService:
    return application.deployments


async def get_foi_resources(application: ApplicationDependency) -> FOIResources:
    return application.foi


async def get_foi_finder(application: ApplicationDependency) -> FOIFinder:
    return application.foi_finder


async def get_sar_classifier(application: ApplicationDependency) -> SARClassifier:
    return application.sar


async def get_automatic_classifier(
    application: ApplicationDependency,
) -> AutomaticClassifier:
    return application.automatic


ServiceDependency = Annotated[DeploymentService, Depends(get_service)]
FOIResourcesDependency = Annotated[FOIResources, Depends(get_foi_resources)]
DependencyFOIFinder = Annotated[FOIFinder, Depends(get_foi_finder)]
SARClassifierDependency = Annotated[SARClassifier, Depends(get_sar_classifier)]
AutomaticClassifierDependency = Annotated[
    AutomaticClassifier, Depends(get_automatic_classifier)
]
