"""Product profiles (Pydantic models).

A profile bundles everything that varies by recipe: expected piece geometry,
fused threshold, classical noise floor, camera exposure/gain, and the vertical
pixel lines that split the belt ROI from the cloth ROI and mark the transfer
line on the cloth.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class CropRegion(BaseModel):
    """Pixel inset from each edge of an ROI.

    All values are measured from the *edge* of the ROI in original
    (post-orientation) pixels.  Default = 0 on all sides = no crop.
    Applied as: roi[top : h-bottom, left : w-right].
    """

    top: int = Field(0, ge=0)
    bottom: int = Field(0, ge=0)
    left: int = Field(0, ge=0)
    right: int = Field(0, ge=0)


class ProductProfile(BaseModel):
    """Per-recipe inspection parameters.

    The defaults reflect the "Default" profile shipped in ``config/default.yaml``
    so an unfilled profile won't crash detection.
    """

    name: str = Field(..., min_length=1, max_length=64)
    expected_area_px: int = Field(60_000, gt=0)
    expected_width_px: int = Field(290, gt=0)
    expected_height_px: int = Field(290, gt=0)
    fused_threshold: float = Field(
        1.5,
        gt=1.0,
        description=(
            "Multiplier on expected_width/height above which a blob is classified "
            "as row_fused or column_fused.  Raised from 1.4 → 1.5 to accommodate "
            "natural ±50% size variation without false-positive fused calls."
        ),
    )
    noise_threshold: float = Field(
        0.5,
        gt=0.0,
        lt=1.0,
        description=(
            "Multiplier on expected_area_px below which a blob is dropped as noise."
        ),
    )
    single_min_ratio: float = Field(
        0.5,
        gt=0.0,
        lt=1.0,
        description=(
            "Minimum width/height ratio relative to expected reference for a blob "
            "to be considered a real piece (not debris).  Replaces the old hardcoded "
            "value of (1/fused_threshold)*0.5 ≈ 0.36."
        ),
    )
    unknown_max_ratio: float = Field(
        2.5,
        gt=1.0,
        description=(
            "If either blob dimension exceeds this multiple of the reference the blob "
            "is classified as 'unknown' regardless of other checks.  Prevents artifacts "
            "or merged blobs from being mis-labelled as fused (which would imply a "
            "specific physical defect)."
        ),
    )
    belt_adaptive_block: int = Field(
        0,
        ge=0,
        description=(
            "Block size for adaptive threshold during belt detection.  0 = auto "
            "(2 × expected_width_px, rounded to the nearest odd number).  Set "
            "manually to override.  Adaptive threshold prevents nearby donuts from "
            "merging into one blob due to uneven belt lighting."
        ),
    )
    camera_exposure_us: int = Field(2000, ge=10, le=1_000_000)
    camera_gain: float = Field(0.0, ge=0.0, le=48.0)
    roi_split_x: int = Field(
        1768,
        gt=0,
        description="Vertical pixel column separating belt (left) and cloth (right).",
    )
    transfer_line_x: int = Field(
        2400,
        gt=0,
        description=(
            "Vertical pixel column in FULL-FRAME coordinates that triggers "
            "StopTuchabzug when the cloth front row crosses it. "
            "Subtract roi_split_x to obtain the cloth-local offset."
        ),
    )
    dough_is_darker: bool = Field(
        True,
        description=(
            "True when dough pieces are darker than the background (e.g. dark dough on "
            "bright Gärtuch under ambient white light). False when dough is brighter "
            "(e.g. belt with dark conveyor and bright pieces, or IR back-illumination)."
        ),
    )
    belt_crop: CropRegion = Field(
        default_factory=CropRegion,
        description="Pixel inset from each edge of the belt ROI before detection.",
    )
    cloth_crop: CropRegion = Field(
        default_factory=CropRegion,
        description="Pixel inset from each edge of the cloth ROI before detection.",
    )
    belt_dough_is_darker: bool = Field(
        False,
        description=(
            "True when dough is darker than the belt background. "
            "Typically False: the belt is a dark wire-mesh / grating and the "
            "dough pieces are lighter — opposite of the cloth situation."
        ),
    )
    belt_min_solidity: float = Field(
        0.80,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum solidity (blob_area / convex_hull_area) for a belt blob to be "
            "accepted as a dough candidate. Grating reflections and metal-structure "
            "artifacts are irregular and score low; real dough pieces are compact. "
            "Blobs below this threshold are silently discarded before classification."
        ),
    )
    belt_max_aspect_ratio: float = Field(
        3.0,
        ge=1.0,
        description=(
            "Maximum aspect ratio (longer_side / shorter_side) for a belt blob. "
            "Elongated reflection streaks exceed this limit and are silently "
            "discarded before classification."
        ),
    )
    # Shape descriptors learned at Learn Reference (0.0 = not yet learned).
    ref_circularity_mean: float = Field(0.0, ge=0.0, le=1.0)
    ref_circularity_min: float = Field(0.0, ge=0.0, le=1.0)
    ref_solidity_mean: float = Field(0.0, ge=0.0, le=1.0)
    ref_solidity_min: float = Field(0.0, ge=0.0, le=1.0)
    # Tolerance multipliers applied to the learned minimums for runtime filtering.
    # gate = ref_*_min * *_tolerance.  0.0 ref value disables the gate entirely.
    circularity_tolerance: float = Field(
        0.7, gt=0.0, le=1.0,
        description=(
            "Fraction of ref_circularity_min below which a blob is rejected. "
            "Default 0.7: blob circularity must be >= 70% of the learned minimum."
        ),
    )
    solidity_tolerance: float = Field(
        0.85, gt=0.0, le=1.0,
        description=(
            "Fraction of ref_solidity_min below which a blob is rejected. "
            "Default 0.85: blob solidity must be >= 85% of the learned minimum."
        ),
    )
    fused_merge_distance_factor: float = Field(
        1.15, gt=0.0,
        description=(
            "Two detected blobs whose centroid distance is less than "
            "fused_merge_distance_factor × expected_diameter are treated as a "
            "fused pair and merged into one detection before classification."
        ),
    )
    dilation_kernel_size: int = Field(
        11,
        ge=0,
        description=(
            "Kernel size for morphological dilation applied after the opening step, "
            "before blob detection. Expands foreground blobs slightly to merge donuts "
            "that are physically stuck but separated by a thin bright gap at the contact "
            "point. Set to 0 to disable."
        ),
    )
    transfer_bridge_width_px: int = Field(
        40,
        ge=1,
        description=(
            "SECTION 4: total width (cloth-ROI pixels) of the 'transfer "
            "bridge' — a band CENTERED on transfer_line_x that replaces the "
            "fragile thin transfer line for stop decisions. A piece is 'at "
            "the transfer point' once its leading edge reaches/overlaps this "
            "band, which survives dropped frames (a piece can't slip across a "
            "thin line between two processed frames). Set in the wizard's "
            "Transfer Line step. transfer_line_x stays the bridge's center."
        ),
    )
    tripwire_half_width_px: int = Field(
        15,
        ge=1,
        description=(
            "Half-width of the transfer-line tripwire strip in cloth-ROI pixels. "
            "The strip spans [transfer_line_x - half_width, transfer_line_x + half_width] "
            "in cloth-local coordinates. Default 15 → 30 px wide strip."
        ),
    )
    tripwire_occupancy_threshold: float = Field(
        0.12,
        gt=0.0,
        lt=1.0,
        description=(
            "Fraction of tripwire strip pixels that must be classified as dough to "
            "fire StopTuchabzug. Default 0.12 = 12 % of the strip must be dough."
        ),
    )
    tripwire_debounce_frames: int = Field(
        2,
        ge=1,
        description=(
            "Number of consecutive frames the occupancy state must be stable before "
            "a rising or falling edge is accepted. Prevents pixel-noise chatter from "
            "causing spurious StopTuchabzug transitions."
        ),
    )

    @field_validator("name")
    @classmethod
    def _no_slashes(cls, v: str) -> str:
        if "/" in v or "\\" in v:
            raise ValueError("profile name must not contain path separators")
        return v.strip()

    def expected_area_mm2(self, px_per_mm: float) -> float:
        """Convert ``expected_area_px`` to mm² given a calibration factor."""
        if px_per_mm <= 0:
            raise ValueError("px_per_mm must be positive")
        return float(self.expected_area_px) / (px_per_mm * px_per_mm)


class ProfileStore:
    """In-memory collection of profiles indexed by name.

    Persistence is handled by ``core.config`` (which writes the full app config
    YAML). This class just owns the lookup/update logic.
    """

    def __init__(self, profiles: list[ProductProfile] | None = None) -> None:
        self._by_name: dict[str, ProductProfile] = {}
        for p in profiles or []:
            self._by_name[p.name] = p

    def names(self) -> list[str]:
        return sorted(self._by_name.keys())

    def get(self, name: str) -> ProductProfile:
        if name not in self._by_name:
            raise KeyError(f"unknown profile: {name!r}")
        return self._by_name[name]

    def has(self, name: str) -> bool:
        return name in self._by_name

    def upsert(self, profile: ProductProfile) -> None:
        """Insert or replace by name. Used by the wizard and Learn Reference."""
        self._by_name[profile.name] = profile

    def remove(self, name: str) -> None:
        if name not in self._by_name:
            raise KeyError(f"unknown profile: {name!r}")
        del self._by_name[name]

    def as_list(self) -> list[ProductProfile]:
        """Stable, name-sorted list suitable for serialization."""
        return [self._by_name[n] for n in self.names()]

    def __len__(self) -> int:
        return len(self._by_name)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._by_name
