"""
The application's closed set of local resource types.

Each declaration pairs a configuration schema with a lazy, statically checked
factory/owner binding. Catalog validation uses only schemas; resolving a binding
imports Python owners but never loads weights. There is no plugin discovery or
configuration-controlled Python import path.

The same declarations generate Pydantic's tagged validation and serialization
schema. Callers keep concrete configuration and owner types through ResourceType
values; only the combined registration index needs heterogeneous storage.

Each uppercase value, such as CATEGORY_MATCHER, is a ResourceType registration.
Its adjacent binding function imports the factory and owner class when needed.
Calling registration.create() constructs a cold controller; its load() or
inference methods perform model loading later. A type describes an approach,
while deployment slugs identify independently owned instances of that approach.

See docs/resource-registration.md for the resource families and extension steps.
"""

from __future__ import annotations

from collections.abc import Callable
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Generic,
    NamedTuple,
    Protocol,
    TypeVar,
    get_args,
)

from pydantic import GetCoreSchemaHandler, GetPydanticSchema
from pydantic_core import core_schema

from ..inference.resources import LocalResource, ResourceRegistry
from .local_config import (
    AutoClassifierDeployment,
    AutoLogisticDeployment,
    CategoryMatcherDeployment,
    LocalDeployment,
    PresidioDeployment,
    QuestionSliceHeadDeployment,
    ResponseTokenizerDeployment,
    SARClassifierDeployment,
    SARLogisticDeployment,
    SequenceClassifierDeployment,
    TopicTokenizerDeployment,
)

if TYPE_CHECKING:
    from ..automatic.resources import LogisticDetector as AutomaticLogisticDetector
    from ..foi.backends import ModernBertClassificationHead
    from ..foi.category_classification import CategoryMatcher
    from ..inference.classifiers import (
        AuthenticatedSequenceClassifier,
        LocalSequenceClassifier,
    )
    from ..sanitization.presidio import PresidioSanitizer
    from ..sar.resources import LogisticDetector as SARLogisticDetector
    from .catalog import DeploymentCatalog


class ResourceController(Protocol):
    """
    Describe a model controller with a LocalResource lifecycle guard.

    An owner such as CategoryMatcher holds loaded model state and inference
    methods. Its resource attribute is the guard: a LocalResource that calls
    load/unload callbacks and prevents idle unloading while a use() lease is
    active. Providers need that lifecycle interface; feature code uses the
    concrete owner's inference methods.
    """

    @property
    def resource(self) -> LocalResource[Any]:
        """
        Return the guard shared by inference leases, warm-up and idle unloading.
        """
        ...


ResourceOwner = ResourceController | LocalResource[Any]
Config = TypeVar("Config", bound=LocalDeployment)
Owner = TypeVar("Owner", bound=ResourceOwner)
ConfigContra = TypeVar("ConfigContra", bound=LocalDeployment, contravariant=True)
OwnerCo = TypeVar("OwnerCo", bound=ResourceOwner, covariant=True)


class ResourceFactory(Protocol[ConfigContra, OwnerCo]):
    """
    Describe a cold factory accepting configuration and application dependencies.

    Link its accepted schema and returned owner for static checking. Factories
    may register a lifecycle guard but must defer artifacts and model loading.
    """

    def __call__(
        self,
        config: ConfigContra,
        *,
        catalog: DeploymentCatalog,
        registry: ResourceRegistry,
    ) -> OwnerCo:
        """
        Construct an owner using the supplied catalog and application registry.
        """
        ...


class ResourceBinding(NamedTuple, Generic[Config, Owner]):
    """
    Pair a cold factory with the concrete owner class used to check its result.

    Lazy binding functions return this immutable record after importing their
    feature module. Keeping both fields typed checks factory compatibility
    without passing anonymous tuples or dictionaries between modules.
    """

    factory: ResourceFactory[Config, Owner]
    owner_type: type[Owner]


class ResourceType(NamedTuple, Generic[Config, Owner]):
    """
    Associate a data-only deployment schema with a deferred factory/owner binding.

    The schema is available during catalog validation. Calling binding imports
    the runtime factory and owner class only when construction is requested.
    Exported instances also act as typed tokens for ResourceProviders.provider().
    """

    config_type: type[Config]
    binding: Callable[[], ResourceBinding[Config, Owner]]

    @property
    def loader(self) -> str:
        """
        Use the schema's discriminator as the sole loader identity.
        """
        values = get_args(self.config_type.model_fields["loader"].annotation)
        if len(values) != 1 or not isinstance(values[0], str):
            raise TypeError("Resource schemas require one literal loader tag")
        return values[0]

    def create(
        self,
        config: LocalDeployment,
        *,
        catalog: DeploymentCatalog,
        registry: ResourceRegistry,
    ) -> Owner:
        """
        Construct a cold owner and check both sides of the registration contract.

        Reject an incompatible configuration before importing the binding, then
        check the factory's returned owner against its declared class. Providers
        separately check and retain the owner's lifecycle guard. Construction
        exceptions propagate without being translated into feature-level results.
        """
        if not isinstance(config, self.config_type):
            raise TypeError(
                f"Resource {config.slug!r} requires {self.config_type.__name__}"
            )
        binding = self.binding()
        owner = binding.factory(config, catalog=catalog, registry=registry)
        if not isinstance(owner, binding.owner_type):
            raise TypeError(
                f"Resource {config.slug!r} did not produce {binding.owner_type.__name__}"
            )
        return owner


_declared: list[ResourceType[Any, Any]] = []


def _register_resource_type(
    config_type: type[Config],
    binding: Callable[[], ResourceBinding[Config, Owner]],
) -> ResourceType[Config, Owner]:
    """
    Add one built-in declaration and return its typed registration value.

    Record the schema and lazy binding without calling the binding. Reject
    duplicate loader tags before building the catalog's discriminated schema.
    This helper is private because registrations must be declared in this module
    before RESOURCE_TYPES is frozen; runtime code uses the named values below.
    """
    registration = ResourceType(config_type=config_type, binding=binding)
    if any(item.loader == registration.loader for item in _declared):
        raise ValueError(f"Duplicate resource loader: {registration.loader}")
    _declared.append(registration)
    return registration


def category_matcher_binding() -> ResourceBinding[
    CategoryMatcherDeployment, CategoryMatcher
]:
    """
    Resolve the pinned category matcher factory and its application-owned controller.
    """
    from ..foi.category_classification import CategoryMatcher, category_matcher

    return ResourceBinding(factory=category_matcher, owner_type=CategoryMatcher)


CATEGORY_MATCHER = _register_resource_type(
    CategoryMatcherDeployment, category_matcher_binding
)


def question_classifier_binding() -> ResourceBinding[
    SequenceClassifierDeployment, LocalSequenceClassifier
]:
    """
    Resolve the local QuestionSlice classifier factory and its sequence-classifier owner.
    """
    from ..foi.backends import create_question_classifier
    from ..inference.classifiers import LocalSequenceClassifier

    return ResourceBinding(
        factory=create_question_classifier, owner_type=LocalSequenceClassifier
    )


QUESTION_CLASSIFIER = _register_resource_type(
    SequenceClassifierDeployment, question_classifier_binding
)


def question_head_binding() -> ResourceBinding[
    QuestionSliceHeadDeployment, ModernBertClassificationHead
]:
    """
    Resolve the QuestionSlice head factory used with remote encoder embeddings.
    """
    from ..foi.backends import (
        ModernBertClassificationHead,
        create_question_classification_head,
    )

    return ResourceBinding(
        factory=create_question_classification_head,
        owner_type=ModernBertClassificationHead,
    )


QUESTION_HEAD = _register_resource_type(
    QuestionSliceHeadDeployment, question_head_binding
)


def topic_tokenizer_binding() -> ResourceBinding[
    TopicTokenizerDeployment, LocalResource[Any]
]:
    """
    Resolve the topic tokenizer factory, whose owner is its lifecycle guard.
    """
    from ..foi.topic_classification import tokenizer_resource

    return ResourceBinding(factory=tokenizer_resource, owner_type=LocalResource)


TOPIC_TOKENIZER = _register_resource_type(
    TopicTokenizerDeployment, topic_tokenizer_binding
)


def response_tokenizer_binding() -> ResourceBinding[
    ResponseTokenizerDeployment, LocalResource[Any]
]:
    """
    Resolve the status tokenizer factory, whose owner is its lifecycle guard.
    """
    from ..foi.response_model import tokenizer_resource

    return ResourceBinding(factory=tokenizer_resource, owner_type=LocalResource)


RESPONSE_TOKENIZER = _register_resource_type(
    ResponseTokenizerDeployment, response_tokenizer_binding
)


def sar_logistic_binding() -> ResourceBinding[
    SARLogisticDeployment, SARLogisticDetector
]:
    """
    Resolve the SAR logistic factory and its feature-specific inference owner.
    """
    from ..sar.resources import LogisticDetector

    return ResourceBinding(factory=LogisticDetector, owner_type=LogisticDetector)


SAR_LOGISTIC = _register_resource_type(SARLogisticDeployment, sar_logistic_binding)


def sar_classifier_binding() -> ResourceBinding[
    SARClassifierDeployment, AuthenticatedSequenceClassifier
]:
    """
    Resolve the SAR confirmation factory and its authenticated classifier owner.
    """
    from ..inference.classifiers import AuthenticatedSequenceClassifier
    from ..sar.resources import create_deberta_detector

    return ResourceBinding(
        factory=create_deberta_detector, owner_type=AuthenticatedSequenceClassifier
    )


SAR_CLASSIFIER = _register_resource_type(
    SARClassifierDeployment, sar_classifier_binding
)


def automatic_logistic_binding() -> ResourceBinding[
    AutoLogisticDeployment, AutomaticLogisticDetector
]:
    """
    Resolve the automatic-response logistic factory and its inference owner.
    """
    from ..automatic.resources import LogisticDetector

    return ResourceBinding(factory=LogisticDetector, owner_type=LogisticDetector)


AUTOMATIC_LOGISTIC = _register_resource_type(
    AutoLogisticDeployment, automatic_logistic_binding
)


def automatic_classifier_binding() -> ResourceBinding[
    AutoClassifierDeployment, AuthenticatedSequenceClassifier
]:
    """
    Resolve the concrete automatic-response encoder factory and classifier owner.
    """
    from ..automatic.resources import create_encoder_detector
    from ..inference.classifiers import AuthenticatedSequenceClassifier

    return ResourceBinding(
        factory=create_encoder_detector, owner_type=AuthenticatedSequenceClassifier
    )


AUTOMATIC_CLASSIFIER = _register_resource_type(
    AutoClassifierDeployment, automatic_classifier_binding
)


def presidio_binding() -> ResourceBinding[PresidioDeployment, PresidioSanitizer]:
    """
    Resolve the sanitizer factory and its application-owned Presidio controller.
    """
    from ..sanitization.presidio import PresidioSanitizer, configured_presidio

    return ResourceBinding(factory=configured_presidio, owner_type=PresidioSanitizer)


PRESIDIO = _register_resource_type(PresidioDeployment, presidio_binding)


RESOURCE_TYPES: MappingProxyType[type[LocalDeployment], ResourceType[Any, Any]] = (
    MappingProxyType({item.config_type: item for item in _declared})
)


def local_deployment_schema(
    source: Any, handler: GetCoreSchemaHandler
) -> core_schema.CoreSchema:
    """
    Build catalog validation and serialization from the registered loader schemas.

    GetPydanticSchema supplies the annotated source type and schema handler.
    Replace that source's base schema with a loader-discriminated union of the
    registered concrete schemas. Pydantic can then validate each resource's
    fields, serialize those fields and expose the same choices in JSON schema.

    Static consumers use LocalDeployment as the common base and narrow through
    catalog.local.get() or a typed ResourceType token. No handwritten union needs
    to be updated alongside the registrations.
    """
    return core_schema.tagged_union_schema(
        {
            item.loader: handler.generate_schema(item.config_type)
            for item in RESOURCE_TYPES.values()
        },
        discriminator="loader",
    )


LocalDeploymentConfig = Annotated[
    LocalDeployment, GetPydanticSchema(local_deployment_schema)
]
