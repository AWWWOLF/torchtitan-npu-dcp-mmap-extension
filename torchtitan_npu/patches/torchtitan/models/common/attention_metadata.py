# Backport upstream PR: https://github.com/pytorch/torchtitan/pull/4897
# Backport upstream PR: https://github.com/pytorch/torchtitan/pull/4617
# Remove this module (and its activation import in
# ``torchtitan_npu/patches/torchtitan/__init__.py``) once the pinned
# ``torchtitan`` dependency carries both PRs natively.

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Backport of upstream's backend-owned attention-metadata contract.

Upstream #4897 moved the ownership of attention metadata to the attention
backends: ``InnerAttention.Config.build_attention_metadata`` builds the
per-batch metadata, ``InnerAttention.attention_metadata_key`` names the entry
the ``Decoder`` builds once per microbatch and hands to the layer that owns
it.  #4617 added the context-parallel half: each CP backend prepares its own
rank-local metadata (``prepare_cp_metadata``) once the load-balancer
permutation is known.

The pinned TorchTitan (0.3.0) has neither the ``InnerAttention`` base class
nor a metadata map: ``Decoder.get_attention_masks`` builds one object of one
kind for the whole stack.  This module carries the upstream contract, with the
upstream names, signatures and semantics, on the classes 0.3.0 does have —
every inner attention is a ``Module``, so the contract lives on
``Module``/``Module.Config`` instead of on ``InnerAttention``:

===============================  ==============================
upstream (#4897)                 here (0.3.0)
===============================  ==============================
``InnerAttention.Config``        ``Module.Config``
``InnerAttention``               ``Module``
``self.attention_metadata_key``  attribute set by the stamped ``build()``
set in ``InnerAttention.__init__``
===============================  ==============================

Two deliberate deviations, both to be proposed upstream when the DSV4 stack is
ported onto this seam:

- **the map key is a config-declared hashable**, not ``type[InnerAttention]``.
  Upstream keys by backend class, which #4897 itself documents as limiting
  ("if the same backend class is configured differently across layers,
  metadata is currently built once using the first encountered
  configuration"): the DeepSeek stacks run several geometries of one backend
  class.  The default key is still the backend class, so a stack that does not
  override it behaves exactly like upstream.
- **the CP transform is read off the metadata object** rather than being a
  static method on the attention class (upstream #4617), because here the map
  key is not necessarily the class.  ``prepare_cp_attention_metadata`` is the
  single call site; the backend-side name stays ``prepare_cp_metadata``.
"""

from collections.abc import Hashable, Mapping
from typing import Any

import torch
import torchtitan.models.common.attention
from torch.nn.attention.flex_attention import BlockMask
from torchtitan.models.common.attention import VarlenMetadata
from torchtitan.protocols.module import Module

__all__ = [
    "AttentionMetadata",
    "AttentionMetadataMap",
    "VarlenAttentionMetadata",
    "prepare_cp_attention_metadata",
]

# ``#4897`` renamed ``VarlenMetadata`` to ``VarlenAttentionMetadata``.  The
# alias lets this repository's code use the name it will keep after the pin
# bump; the pinned version still exports the old one.
VarlenAttentionMetadata = VarlenMetadata

# 0.3.0's two metadata kinds (upstream's union additionally carries the KDA
# and CP-local variants that arrived with #4639/#4617).
AttentionMetadata = BlockMask | VarlenMetadata

# Upstream: ``Mapping[type[InnerAttention], AttentionMetadata]``.
AttentionMetadataMap = Mapping[Hashable, AttentionMetadata]


def build_attention_metadata(
    self,
    positions: torch.Tensor,
    *,
    padding_mask: torch.Tensor | None = None,
    max_num_documents: int | None = None,
    max_context_length: int | None = None,
) -> AttentionMetadata | None:
    """Build the metadata this attention backend consumes, if any.

    Upstream ``InnerAttention.Config.build_attention_metadata``; backends that
    need no metadata inherit this default and contribute no map entry.
    """
    del positions, padding_mask, max_num_documents, max_context_length
    return None


def attention_metadata_key(self) -> Hashable:
    """The map key of this attention config (upstream's backend key by default).

    A config that runs several geometries of one backend class overrides this
    (a geometry dataclass is the intended shape); the module-side key must
    agree with it, which is what the stamped ``Module.Config.build`` below
    guarantees.
    """
    owner = type(self)._owner
    return type(self) if owner is None else owner


def _module_attention_metadata_key(self) -> Hashable:
    """The map key of an attention module (upstream sets this in ``__init__``)."""
    return self.__dict__.get("_attention_metadata_key", type(self))


def _set_module_attention_metadata_key(self, value: Hashable) -> None:
    self.__dict__["_attention_metadata_key"] = value


def _attention_wrapper_metadata_key(self) -> Hashable:
    """The key an attention wrapper dispatches by (upstream ``BaseAttention``).

    The layer reads ``layer.attention.attention_metadata_key``, so the wrapper
    forwards to the inner attention that actually built the metadata.
    """
    return self.inner_attention.attention_metadata_key


def prepare_cp_attention_metadata(
    attention_metadata: AttentionMetadataMap,
    *,
    permutation: torch.Tensor | None,
    cp_mesh: Any = None,
    load_balancer_type: str | None = None,
    positions: torch.Tensor | None = None,
) -> dict[Hashable, Any]:
    """Rank-local metadata for context parallel (upstream #4617's CP half).

    Every entry whose metadata defines ``prepare_cp_metadata`` is transformed;
    the others pass through unchanged (the Ulysses case, where metadata stays
    global).  Upstream calls the backend class' static method per map entry
    with the permutation alone, because the mesh is ambient there; 0.3.0 hands
    the mesh, the load-balancer type and this rank's positions along as well,
    and an implementation reads whichever of them it needs.
    """
    prepared: dict[Hashable, Any] = {}
    for key, metadata in attention_metadata.items():
        prepare = getattr(metadata, "prepare_cp_metadata", None)
        prepared[key] = (
            metadata
            if prepare is None
            else prepare(
                permutation=permutation,
                cp_mesh=cp_mesh,
                load_balancer_type=load_balancer_type,
                positions=positions,
            )
        )
    return prepared


def apply() -> None:
    attention = torchtitan.models.common.attention
    # The names the pinned version lacks, on the module the call sites import
    # them from: #4897's rename and map aliases, plus the CP transform the
    # repository's CP path (upstream #4617's call site) applies to the map.
    attention.VarlenAttentionMetadata = VarlenAttentionMetadata  # pyrefly: ignore [missing-attribute]
    attention.AttentionMetadata = AttentionMetadata  # pyrefly: ignore [missing-attribute]
    attention.AttentionMetadataMap = AttentionMetadataMap  # pyrefly: ignore [missing-attribute]
    attention.prepare_cp_attention_metadata = prepare_cp_attention_metadata  # pyrefly: ignore [missing-attribute]
    if hasattr(attention, "__all__"):
        for name in (
            "VarlenAttentionMetadata",
            "AttentionMetadata",
            "AttentionMetadataMap",
            "prepare_cp_attention_metadata",
        ):
            if name not in attention.__all__:
                attention.__all__.append(name)

    if getattr(Module.Config, "_npu_attention_metadata_contract", False):
        return
    Module.Config.build_attention_metadata = build_attention_metadata  # pyrefly: ignore [bad-assignment]
    Module.Config.attention_metadata_key = property(attention_metadata_key)  # pyrefly: ignore [bad-assignment]
    Module.attention_metadata_key = property(  # pyrefly: ignore [bad-assignment]
        _module_attention_metadata_key,
        _set_module_attention_metadata_key,
    )
    # Upstream #4897's ``BaseAttention.attention_metadata_key``: the layer's
    # block reads the attention wrapper, which dispatches by its inner
    # attention's key.
    attention.BaseAttention.attention_metadata_key = property(  # pyrefly: ignore [bad-assignment]
        _attention_wrapper_metadata_key
    )

    # ``Module.Config.build`` builds every module; stamping the key there is the
    # 0.3.0 stand-in for upstream's ``InnerAttention.__init__`` assignment, and
    # it is what keeps a geometry-keyed record reachable from the layer that
    # reads it through ``layer.attention.attention_metadata_key``.
    original_build = Module.Config.build

    def build(self, **kwargs):
        # Unconditional, as upstream's ``InnerAttention.__init__`` does it: 0.3.0 has no
        # such ``__init__``, and GraphTrainer's SimpleFSDP rewrites the ``__class__`` of every
        # module it wraps, parameter-free ones included.  A build-time type test would skip
        # the default key, so the getter below would read the rewritten runtime class, miss
        # the entry the config built the map with and hand the layer ``None`` metadata.
        instance = original_build(self, **kwargs)
        instance._attention_metadata_key = self.attention_metadata_key
        return instance

    Module.Config.build = build  # pyrefly: ignore [bad-assignment]
    Module.Config._npu_attention_metadata_contract = True


apply()
