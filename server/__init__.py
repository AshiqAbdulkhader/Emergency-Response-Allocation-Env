# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Emergency Response Allocation environment server components."""

try:
    from .emergency_response_allocation_environment import EmergencyResponseAllocationEnvironment
except ImportError:
    from emergency_response_allocation_environment import EmergencyResponseAllocationEnvironment

__all__ = ["EmergencyResponseAllocationEnvironment"]
