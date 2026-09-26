"""
Configurable augmentation for DermaMNIST.

The augmentation pipeline is described by one flat, JSON-serialisable
`AugmentationConfig`. That shape is deliberate: a human passes
`--aug-preset default`, while a tuning agent (or Optuna, or a random search)
emits a dict of scalars, and both end up building the *same* pipeline through
`resolve_augmentation()`. The resolved config is stored verbatim in
`experiment_record.json`, so a run can be reproduced and two runs can be
compared field by field.

For agent use specifically:

    AUG_SEARCH_SPACE           declares every field's type, range, default and
                               the domain reason for its bounds
    AugmentationConfig.sample  draws a valid random config from that space
    from_dict                  rejects unknown keys instead of ignoring them
    validate                   raises on out-of-range values, naming the field
    describe                   lists the ops actually applied, for logging

What this modality justifies
----------------------------
Dermoscopic images have **no canonical orientation** — the lesion is
photographed in an arbitrary rotation. The full dihedral group (horizontal
flip x vertical flip x 90-degree rotation = 8 variants) is therefore exactly
label-preserving and free of interpolation artefacts. That is the cheapest
real diversity available, and it matters most for the long tail: class 3
(dermatofibroma) has 80 training images.

Colour, by contrast, *is* diagnostic signal in dermoscopy, and at 28x28
there is little of it to spare. Hue jitter is bounded tightly and defaults
to 0.02; widening it can erase the distinction between pigmented classes.

What was removed and why
------------------------
The previous pipeline used `RandomCrop(28, padding=4)`, whose torchvision
defaults are `fill=0, padding_mode="constant"`. On a dataset whose mean
pixel is (195, 137, 143) that injects a pure-black border into a bright
skin image. Worse, it mimics a *real* artefact: part of HAM10000 has
dermoscope vignetting (dark corners), so the augmentation was teaching the
model that dark borders are ignorable noise exactly where they are a
device-correlated confound. `RandomResizedCrop` replaces it: it samples a
region *inside* the image, so translation and scaling need no fill at all.

Every geometric op that cannot avoid a border (arbitrary-angle rotation)
fills with the dataset mean colour rather than black, for the same reason.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image
from torchvision import transforms

# Per-channel statistics of the DermaMNIST *training* split, RGB in [0, 1].
# Verified against the official dermamnist.npz (actual: mean
# [0.7636, 0.5372, 0.5614], std [0.1362, 0.1540, 0.1687]).
DERMAMNIST_MEAN = (0.7632, 0.5391, 0.5615)
DERMAMNIST_STD = (0.1409, 0.1523, 0.1699)

# The same mean as a 0-255 int triple, used as the fill colour for geometric
# ops so a rotation never injects a black wedge.
DERMAMNIST_FILL = tuple(int(round(c * 255)) for c in DERMAMNIST_MEAN)  # (195, 137, 143)

# Pillow's rotate constants moved under Image.Transpose in 9.1.
try:  # pragma: no cover - depends on the installed Pillow
    _ROTATIONS = (
        None,
        Image.Transpose.ROTATE_90,
        Image.Transpose.ROTATE_180,
        Image.Transpose.ROTATE_270,
    )
except AttributeError:  # pragma: no cover
    _ROTATIONS = (None, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270)


# --------------------------------------------------------------------------
# Transform primitives (plain classes, so they stay picklable for DataLoader
# workers — macOS/Windows spawn workers and re-pickle the transform)
# --------------------------------------------------------------------------
class RandomRotation90:
    """Rotates by a random multiple of 90 degrees.

    Exact by construction: no interpolation, no resampling, no border fill,
    because a square image maps onto itself. Combined with the two flips it
    generates the full dihedral group of 8 orientations.
    """

    def __init__(self, p: float = 1.0):
        self.p = p

    def __call__(self, img):
        if self.p < 1.0 and random.random() > self.p:
            return img
        k = random.randint(0, 3)
        if k == 0:
            return img
        if isinstance(img, torch.Tensor):
            return torch.rot90(img, k, dims=(-2, -1))
        return img.transpose(_ROTATIONS[k])

    def __repr__(self) -> str:
        return f"{type(self).__name__}(p={self.p})"


class Cutout:
    """Randomly masks out one square patch of the image (DeVries & Taylor, 2017).

    Applied after ToTensor/Normalize, operating directly on the tensor.
    Filling with 0.0 *after* normalisation means the patch takes the dataset
    mean colour, not black — which is what the original method does.

    Note that the patch is clipped at the borders, so patches sampled near an
    edge cover less than `size * size` and the mean occluded area is below
    the nominal one.
    """

    def __init__(self, size: int = 8, p: float = 0.5):
        self.size = size
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img
        c, h, w = img.shape
        half = self.size // 2
        cy, cx = random.randint(0, h - 1), random.randint(0, w - 1)
        y1, y2 = max(0, cy - half), min(h, cy + half)
        x1, x2 = max(0, cx - half), min(w, cx + half)
        img = img.clone()
        img[:, y1:y2, x1:x2] = 0.0
        return img

    def __repr__(self) -> str:
        return f"{type(self).__name__}(size={self.size}, p={self.p})"


# --------------------------------------------------------------------------
# The config
# --------------------------------------------------------------------------
@dataclass
class AugmentationConfig:
    """Flat, JSON-serialisable description of the train-time pipeline.

    Flat on purpose: every field is a scalar, so the config maps directly
    onto a hyperparameter search space without any nesting to unwrap.
    Only the *train* split is augmented; val and test never are.
    """

    # --- dihedral group: exact, artefact-free, and the cheapest diversity ---
    horizontal_flip: float = 0.5   # probability
    vertical_flip: float = 0.5     # probability
    rotate_90: bool = True         # random multiple of 90 degrees

    # --- arbitrary-angle rotation: needs a fill colour, so off by default ---
    rotation_degrees: float = 0.0  # 0 disables; otherwise samples in [-d, +d]

    # --- random resized crop: translation + scaling with no border fill ---
    random_resized_crop: bool = True
    rrc_scale_min: float = 0.8     # fraction of the ORIGINAL AREA kept
    rrc_scale_max: float = 1.0
    rrc_ratio_min: float = 0.9     # aspect-ratio jitter of the crop
    rrc_ratio_max: float = 1.111

    # --- light colour jitter: colour is diagnostic, keep it subtle ---
    color_jitter: bool = True
    cj_brightness: float = 0.1
    cj_contrast: float = 0.1
    cj_saturation: float = 0.1
    cj_hue: float = 0.02

    # --- Cutout, from the paper's stated recipe; off unless overfitting ---
    cutout: bool = False
    cutout_size: int = 8
    cutout_p: float = 0.5

    def __post_init__(self):
        self.validate()

    # -- validation ------------------------------------------------------
    def validate(self) -> "AugmentationConfig":
        """Raises ValueError naming the offending field.

        Called on construction, so an agent that emits a nonsensical config
        fails immediately with an actionable message rather than silently
        training something else.
        """
        def _prob(name: str):
            v = getattr(self, name)
            if not 0.0 <= float(v) <= 1.0:
                raise ValueError(f"{name} must be a probability in [0, 1], got {v!r}")

        _prob("horizontal_flip")
        _prob("vertical_flip")
        _prob("cutout_p")

        if not 0.0 <= float(self.rotation_degrees) <= 180.0:
            raise ValueError(
                f"rotation_degrees must be in [0, 180], got {self.rotation_degrees!r}"
            )

        if not 0.0 < float(self.rrc_scale_min) <= 1.0:
            raise ValueError(f"rrc_scale_min must be in (0, 1], got {self.rrc_scale_min!r}")
        if not 0.0 < float(self.rrc_scale_max) <= 1.0:
            raise ValueError(f"rrc_scale_max must be in (0, 1], got {self.rrc_scale_max!r}")
        if float(self.rrc_scale_min) > float(self.rrc_scale_max):
            raise ValueError(
                f"rrc_scale_min ({self.rrc_scale_min}) must be <= "
                f"rrc_scale_max ({self.rrc_scale_max})"
            )
        if not 0.0 < float(self.rrc_ratio_min) <= float(self.rrc_ratio_max):
            raise ValueError(
                f"need 0 < rrc_ratio_min <= rrc_ratio_max, got "
                f"{self.rrc_ratio_min!r} and {self.rrc_ratio_max!r}"
            )

        for name in ("cj_brightness", "cj_contrast", "cj_saturation"):
            v = float(getattr(self, name))
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {v!r}")
        if not 0.0 <= float(self.cj_hue) <= 0.5:
            raise ValueError(f"cj_hue must be in [0, 0.5], got {self.cj_hue!r}")

        if int(self.cutout_size) < 1 or int(self.cutout_size) > 28:
            raise ValueError(f"cutout_size must be in [1, 28], got {self.cutout_size!r}")

        return self

    # -- serialisation ---------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        return tuple(f.name for f in fields(cls))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AugmentationConfig":
        """Builds a config, rejecting unknown keys.

        Silently dropping a misspelled key would let an agent believe it had
        changed something it had not, which is the worst failure mode for an
        automated search: the run looks configured and is not.
        """
        known = set(cls.field_names())
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(
                f"unknown augmentation field(s): {', '.join(unknown)}. "
                f"Valid fields: {', '.join(sorted(known))}"
            )
        return cls(**data)

    def replace(self, **overrides) -> "AugmentationConfig":
        """Returns a new validated config with `overrides` applied."""
        merged = self.to_dict()
        merged.update(overrides)
        return type(self).from_dict(merged)

    # -- description -----------------------------------------------------
    def describe(self) -> list[str]:
        """Human/agent-readable list of the ops this config actually applies."""
        lines: list[str] = []
        if self.rotate_90:
            lines.append("RandomRotation90 (exact, dihedral)")
        if self.rotation_degrees > 0:
            lines.append(f"RandomRotation(+/-{self.rotation_degrees} deg, fill={DERMAMNIST_FILL})")
        if self.random_resized_crop:
            lines.append(
                f"RandomResizedCrop(28, scale=({self.rrc_scale_min}, {self.rrc_scale_max}), "
                f"ratio=({self.rrc_ratio_min}, {self.rrc_ratio_max}))"
            )
        if self.horizontal_flip > 0:
            lines.append(f"RandomHorizontalFlip(p={self.horizontal_flip})")
        if self.vertical_flip > 0:
            lines.append(f"RandomVerticalFlip(p={self.vertical_flip})")
        if self.color_jitter and max(self.cj_brightness, self.cj_contrast,
                                     self.cj_saturation, self.cj_hue) > 0:
            lines.append(
                f"ColorJitter(brightness={self.cj_brightness}, contrast={self.cj_contrast}, "
                f"saturation={self.cj_saturation}, hue={self.cj_hue})"
            )
        lines.append("ToTensor + Normalize")
        if self.cutout:
            lines.append(f"Cutout(size={self.cutout_size}, p={self.cutout_p})")
        if len(lines) == 1:
            lines.insert(0, "(no augmentation)")
        return lines

    # -- random sampling, for search without an external optimiser -------
    @classmethod
    def sample(cls, rng: Optional[random.Random] = None) -> "AugmentationConfig":
        """Draws a valid random config from AUG_SEARCH_SPACE.

        Enough for a random search; for anything sample-efficient hand
        AUG_SEARCH_SPACE to Optuna instead.
        """
        rng = rng or random.Random()
        values: dict[str, Any] = {}
        for name, spec in AUG_SEARCH_SPACE.items():
            if spec["type"] == "bool":
                values[name] = rng.random() < 0.5
            elif spec["type"] == "int":
                values[name] = rng.randint(spec["low"], spec["high"])
            else:
                values[name] = round(rng.uniform(spec["low"], spec["high"]), 4)

        # Repair the two coupled pairs so the draw is always valid.
        lo, hi = sorted((values["rrc_scale_min"], values["rrc_scale_max"]))
        values["rrc_scale_min"], values["rrc_scale_max"] = lo, hi
        lo, hi = sorted((values["rrc_ratio_min"], values["rrc_ratio_max"]))
        values["rrc_ratio_min"], values["rrc_ratio_max"] = lo, hi
        return cls.from_dict(values)


# --------------------------------------------------------------------------
# Search space: the machine-readable contract for a tuning agent
# --------------------------------------------------------------------------
# `note` carries the domain reason for each bound, so an agent proposing a
# config has the reasoning available and does not have to rediscover it.
AUG_SEARCH_SPACE: dict[str, dict[str, Any]] = {
    "horizontal_flip": {
        "type": "float", "low": 0.0, "high": 0.5, "default": 0.5,
        "note": "label-preserving; 0.5 is the natural maximum for a flip",
    },
    "vertical_flip": {
        "type": "float", "low": 0.0, "high": 0.5, "default": 0.5,
        "note": "dermoscopy has no canonical orientation, so this is free diversity",
    },
    "rotate_90": {
        "type": "bool", "default": True,
        "note": "exact rotation, no interpolation or fill; completes the dihedral group",
    },
    "rotation_degrees": {
        "type": "float", "low": 0.0, "high": 30.0, "default": 0.0,
        "note": "arbitrary angles need a fill colour and blur a 28x28 image; "
                "prefer rotate_90 and keep this small or zero",
    },
    "random_resized_crop": {
        "type": "bool", "default": True,
        "note": "provides translation and scaling without any border fill",
    },
    "rrc_scale_min": {
        "type": "float", "low": 0.6, "high": 1.0, "default": 0.8,
        "note": "lower bound on kept area; below ~0.6 a 28x28 image is upscaled "
                "from fewer than 22x22 px and detail is interpolated away",
    },
    "rrc_scale_max": {
        "type": "float", "low": 0.6, "high": 1.0, "default": 1.0,
        "note": "keep at 1.0 so the untouched image stays in the distribution",
    },
    "rrc_ratio_min": {
        "type": "float", "low": 0.75, "high": 1.0, "default": 0.9,
        "note": "aspect distortion changes lesion shape, which is diagnostic (the "
                "'A' and 'B' of ABCD); stay near 1.0",
    },
    "rrc_ratio_max": {
        "type": "float", "low": 1.0, "high": 1.33, "default": 1.111,
        "note": "as rrc_ratio_min",
    },
    "color_jitter": {
        "type": "bool", "default": True,
        "note": "guards against per-device colour balance, but see cj_hue",
    },
    "cj_brightness": {
        "type": "float", "low": 0.0, "high": 0.3, "default": 0.1,
        "note": "illumination varies between acquisitions, so some is realistic",
    },
    "cj_contrast": {
        "type": "float", "low": 0.0, "high": 0.3, "default": 0.1,
        "note": "as cj_brightness",
    },
    "cj_saturation": {
        "type": "float", "low": 0.0, "high": 0.3, "default": 0.1,
        "note": "saturation carries pigmentation information; keep it modest",
    },
    "cj_hue": {
        "type": "float", "low": 0.0, "high": 0.05, "default": 0.02,
        "note": "TIGHT ON PURPOSE. Hue is diagnostic signal in dermoscopy "
                "(pigmented vs vascular vs keratotic); large shifts move an image "
                "towards another class's appearance and corrupt the label",
    },
    "cutout": {
        "type": "bool", "default": False,
        "note": "from the paper's recipe, but it is a regulariser: it only pays off "
                "once train accuracy exceeds validation accuracy",
    },
    "cutout_size": {
        "type": "int", "low": 4, "high": 12, "default": 8,
        "note": "8 on a 28x28 image occludes up to 8.2% of the pixels",
    },
    "cutout_p": {
        "type": "float", "low": 0.0, "high": 1.0, "default": 0.5,
        "note": "probability of occluding at all",
    },
}


# --------------------------------------------------------------------------
# Presets: named points in the space, so a coarse choice needs one token
# --------------------------------------------------------------------------
PRESETS: dict[str, dict[str, Any]] = {
    # No augmentation at all. Use to diagnose underfitting, and to measure how
    # much the augmentation is costing you in a short run.
    "none": {
        "horizontal_flip": 0.0, "vertical_flip": 0.0, "rotate_90": False,
        "rotation_degrees": 0.0, "random_resized_crop": False,
        "color_jitter": False, "cutout": False,
    },
    # The free, exactly label-preserving symmetries only: no interpolation, no
    # fill, no colour change. The safest thing to always have on.
    "dihedral": {
        "horizontal_flip": 0.5, "vertical_flip": 0.5, "rotate_90": True,
        "rotation_degrees": 0.0, "random_resized_crop": False,
        "color_jitter": False, "cutout": False,
    },
    # Recommended starting point: dihedral group + mild scaling + light colour.
    "default": {},
    # Wider ranges plus Cutout. For long runs where overfitting has appeared.
    "strong": {
        "horizontal_flip": 0.5, "vertical_flip": 0.5, "rotate_90": True,
        "rotation_degrees": 15.0,
        "random_resized_crop": True, "rrc_scale_min": 0.7, "rrc_scale_max": 1.0,
        "color_jitter": True, "cj_brightness": 0.2, "cj_contrast": 0.2,
        "cj_saturation": 0.2, "cj_hue": 0.03,
        "cutout": True, "cutout_size": 8, "cutout_p": 0.5,
    },
    # Closest coherent reading of the FPViT paper's recipe (lightweight
    # augmentation + Cutout), minus the black-fill RandomCrop: the crop's
    # translation role is taken by a mild RandomResizedCrop instead. Not a
    # bit-exact reproduction of the paper — see the module docstring.
    "paper": {
        "horizontal_flip": 0.5, "vertical_flip": 0.0, "rotate_90": False,
        "rotation_degrees": 0.0,
        "random_resized_crop": True, "rrc_scale_min": 0.8, "rrc_scale_max": 1.0,
        "color_jitter": False,
        "cutout": True, "cutout_size": 8, "cutout_p": 0.5,
    },
}


def preset(name: str) -> AugmentationConfig:
    """Returns the named preset as a validated config."""
    if name not in PRESETS:
        raise ValueError(
            f"unknown augmentation preset {name!r}. "
            f"Available: {', '.join(sorted(PRESETS))}"
        )
    return AugmentationConfig.from_dict(dict(PRESETS[name]))


# --------------------------------------------------------------------------
# Resolution: one entry point shared by the CLI and by programmatic callers
# --------------------------------------------------------------------------
def _coerce(name: str, raw: str) -> Any:
    """Parses a `key=value` string against the field's declared type."""
    spec = AUG_SEARCH_SPACE.get(name)
    if spec is None:
        raise ValueError(
            f"unknown augmentation field {name!r}. "
            f"Valid fields: {', '.join(sorted(AugmentationConfig.field_names()))}"
        )
    if spec["type"] == "bool":
        low = raw.strip().lower()
        if low in {"1", "true", "yes", "on"}:
            return True
        if low in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"{name} is a boolean; got {raw!r}")
    if spec["type"] == "int":
        return int(raw)
    return float(raw)


def parse_overrides(items: Optional[list[str]]) -> dict[str, Any]:
    """Turns ["cj_hue=0.01", "cutout=true"] into a typed dict."""
    out: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"expected key=value, got {item!r}")
        key, _, raw = item.partition("=")
        key = key.strip()
        out[key] = _coerce(key, raw)
    return out


def resolve_augmentation(preset_name: str = "default",
                          config: Optional[str | dict[str, Any]] = None,
                          overrides: Optional[dict[str, Any]] = None
                          ) -> AugmentationConfig:
    """Builds the final config from the three ways of specifying one.

    Precedence, lowest to highest:
      1. `preset_name`  — a name from PRESETS
      2. `config`       — a dict, a JSON string, or a path to a JSON file;
                          replaces the preset entirely
      3. `overrides`    — individual field values, applied last

    Both the CLI and a programmatic caller go through here, so a config that
    an agent produces in Python and one passed on the command line resolve
    identically.
    """
    base = preset(preset_name)

    if config is not None:
        if isinstance(config, dict):
            data = config
        else:
            text = str(config).strip()
            if text.startswith("{"):
                data = json.loads(text)
            else:
                path = Path(text)
                if not path.exists():
                    raise FileNotFoundError(
                        f"augmentation config {text!r} is neither inline JSON "
                        "(it does not start with '{') nor an existing file"
                    )
                data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError("augmentation config JSON must be an object")
        # An experiment_record.json can be fed straight back in.
        data = data.get("augmentation", data)
        base = AugmentationConfig.from_dict(data)

    if overrides:
        base = base.replace(**overrides)
    return base


# --------------------------------------------------------------------------
# Pipeline construction
# --------------------------------------------------------------------------
def build_train_transform(cfg: Optional[AugmentationConfig] = None):
    """Composes the train-time pipeline described by `cfg`.

    Order: exact rotation -> arbitrary rotation -> crop/scale -> flips ->
    colour -> ToTensor/Normalize -> Cutout.

    The arbitrary rotation runs *before* the crop so that the crop can trim
    away part of the filled corners it introduces. Cutout runs last, after
    normalisation, which is where the original method puts it.
    """
    cfg = cfg or AugmentationConfig()
    ops: list[Any] = []

    if cfg.rotate_90:
        ops.append(RandomRotation90())

    if cfg.rotation_degrees > 0:
        ops.append(transforms.RandomRotation(
            degrees=float(cfg.rotation_degrees),
            fill=list(DERMAMNIST_FILL),  # never black; see module docstring
        ))

    if cfg.random_resized_crop:
        ops.append(transforms.RandomResizedCrop(
            28,
            scale=(float(cfg.rrc_scale_min), float(cfg.rrc_scale_max)),
            ratio=(float(cfg.rrc_ratio_min), float(cfg.rrc_ratio_max)),
        ))

    if cfg.horizontal_flip > 0:
        ops.append(transforms.RandomHorizontalFlip(p=float(cfg.horizontal_flip)))
    if cfg.vertical_flip > 0:
        ops.append(transforms.RandomVerticalFlip(p=float(cfg.vertical_flip)))

    if cfg.color_jitter and max(cfg.cj_brightness, cfg.cj_contrast,
                                 cfg.cj_saturation, cfg.cj_hue) > 0:
        ops.append(transforms.ColorJitter(
            brightness=float(cfg.cj_brightness),
            contrast=float(cfg.cj_contrast),
            saturation=float(cfg.cj_saturation),
            hue=float(cfg.cj_hue),
        ))

    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(mean=DERMAMNIST_MEAN, std=DERMAMNIST_STD))

    if cfg.cutout:
        ops.append(Cutout(size=int(cfg.cutout_size), p=float(cfg.cutout_p)))

    return transforms.Compose(ops)


def build_eval_transform():
    """Deterministic pipeline for val/test and for single-image inference.

    Never augmented: the evaluation protocol must not depend on a random draw.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=DERMAMNIST_MEAN, std=DERMAMNIST_STD),
    ])
