"""Versioned plugin discovery and capability resolution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from importlib.metadata import EntryPoint, entry_points
from types import MappingProxyType
from typing import Protocol


class CapabilityKind(StrEnum):
    """Small, independently versioned extension roles."""

    DOMAIN_FACTORY = "domain_factory"
    MODEL = "model"
    DERIVED_FIELD = "derived_field"
    VALIDATOR = "validator"
    EXPORTER = "exporter"
    VIEW_RECIPE = "view_recipe"


@dataclass(frozen=True, slots=True)
class Capability:
    """A named implementation provided by a plugin."""

    capability_id: str
    kind: CapabilityKind
    version: str
    provider: object

    def __post_init__(self) -> None:
        if not self.capability_id or not self.version:
            raise ValueError("capability ID and version are required")


@dataclass(frozen=True, slots=True)
class PluginManifest:
    """Import-light metadata used for compatibility resolution."""

    plugin_id: str
    version: str
    core_api: str
    license_expression: str
    description: str
    capabilities: tuple[tuple[str, CapabilityKind, str], ...]
    field_namespaces: tuple[str, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not all(
            (self.plugin_id, self.version, self.core_api, self.license_expression)
        ):
            raise ValueError("plugin identity, version, API, and license are required")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class PhenomenonPlugin(Protocol):
    """Structural contract implemented by phenomenon plugins."""

    @property
    def manifest(self) -> PluginManifest: ...

    def capabilities(self) -> tuple[Capability, ...]: ...


class PluginRegistry:
    """Discover plugins without core imports of phenomenon packages."""

    ENTRY_POINT_GROUP = "openphenomena.plugins"

    def __init__(self) -> None:
        self._plugins: dict[str, PhenomenonPlugin] = {}
        self._capabilities: dict[str, Capability] = {}

    @classmethod
    def discover(cls) -> PluginRegistry:
        registry = cls()
        selected = entry_points(group=cls.ENTRY_POINT_GROUP)
        for entry_point in sorted(selected, key=lambda item: item.name):
            registry.load_entry_point(entry_point)
        return registry

    def load_entry_point(self, entry_point: EntryPoint) -> None:
        loaded = entry_point.load()
        plugin_object = loaded() if isinstance(loaded, type) else loaded
        self.register(plugin_object)

    def register(self, plugin: PhenomenonPlugin) -> None:
        manifest = plugin.manifest
        if manifest.plugin_id in self._plugins:
            raise ValueError(f"duplicate plugin ID: {manifest.plugin_id}")
        provided = plugin.capabilities()
        declared = {
            (capability_id, kind, version)
            for capability_id, kind, version in manifest.capabilities
        }
        actual = {(item.capability_id, item.kind, item.version) for item in provided}
        if declared != actual:
            raise ValueError("plugin manifest capabilities do not match providers")
        collisions = set(self._capabilities).intersection(
            item.capability_id for item in provided
        )
        if collisions:
            raise ValueError(f"duplicate capability IDs: {sorted(collisions)}")
        self._plugins[manifest.plugin_id] = plugin
        self._capabilities.update({item.capability_id: item for item in provided})

    @property
    def plugin_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._plugins))

    def plugin(self, plugin_id: str) -> PhenomenonPlugin:
        return self._plugins[plugin_id]

    def capability(self, capability_id: str) -> Capability:
        return self._capabilities[capability_id]

    def capabilities_of_kind(self, kind: CapabilityKind) -> tuple[Capability, ...]:
        return tuple(
            sorted(
                (item for item in self._capabilities.values() if item.kind is kind),
                key=lambda item: item.capability_id,
            )
        )
