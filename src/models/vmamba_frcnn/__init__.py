"""VMamba-T Faster R-CNN factory and importer."""

from src.models.vmamba_frcnn.factory import (
    VMAMBA_FACTORY_MODEL_ID,
    VMambaBuildResult,
    VMambaFRCNNFactory,
)
from src.models.vmamba_frcnn.features import VMambaFeatureValidator
from src.models.vmamba_frcnn.importer import (
    detect_selective_scan_backend,
    register_vmamba_detection,
    verify_vmamba_revision,
)

__all__ = [
    "VMAMBA_FACTORY_MODEL_ID",
    "VMambaBuildResult",
    "VMambaFRCNNFactory",
    "VMambaFeatureValidator",
    "detect_selective_scan_backend",
    "register_vmamba_detection",
    "verify_vmamba_revision",
]
