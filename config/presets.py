# config/presets.py
# Bone strength material presets.
#
# Each preset maps a clinical bone condition to a set of orthotropic material
# constants (E, G, NU) that fall within the bounds the kiki-tool model was
# trained on.  All presets have been verified to satisfy the Lempriere (1968)
# physics constraints for Poisson's ratios.
#
# Usage:
#   from config.presets import BONE_PRESETS
#   params = BONE_PRESETS["normal"]["material"]

BONE_PRESETS = {
    "osteoporotic": {
        "label": "Osteoporotic",
        "description": (
            "Severely reduced bone density (e.g. osteoporosis). "
            "Very soft, compliant lattice needed to avoid stress shielding."
        ),
        "keywords": ["osteoporosis", "osteoporotic", "very soft", "fragile", "brittle", "severe"],
        "material": {
            "E1": 3.0,  "E2": 1.5,  "E3": 1.5,
            "G12": 1.0, "G23": 0.8, "G31": 1.0,
            "NU12": 0.30, "NU23": 0.25, "NU31": 0.30,
        },
    },
    "elderly": {
        "label": "Elderly",
        "description": (
            "Age-related bone density reduction (typically 65+ years). "
            "Moderately soft lattice to accommodate reduced bone stiffness."
        ),
        "keywords": ["elderly", "older", "aged", "reduced density", "age-related", "low density", "senior"],
        "material": {
            "E1": 10.0, "E2": 6.0,  "E3": 5.0,
            "G12": 3.0, "G23": 2.5, "G31": 3.0,
            "NU12": 0.30, "NU23": 0.25, "NU31": 0.30,
        },
    },
    "normal": {
        "label": "Normal",
        "description": (
            "Healthy adult bone density (typical adult patient). "
            "Default preset — matches the training data baseline values."
        ),
        "keywords": ["normal", "healthy", "standard", "average", "typical", "default", "adult"],
        "material": {
            "E1": 21.0, "E2": 10.0, "E3": 10.0,
            "G12": 6.0, "G23": 4.0,  "G31": 6.0,
            "NU12": 0.30, "NU23": 0.20, "NU31": 0.30,
        },
    },
    "athletic": {
        "label": "Athletic",
        "description": (
            "Dense, high-stiffness bone (young, athletic or active patient). "
            "Stiffer lattice to match the higher bone modulus."
        ),
        "keywords": ["athletic", "athlete", "young", "dense", "strong", "high density", "active", "sport"],
        "material": {
            "E1": 35.0, "E2": 18.0, "E3": 16.0,
            "G12": 10.0, "G23": 8.0, "G31": 9.0,
            "NU12": 0.32, "NU23": 0.28, "NU31": 0.30,
        },
    },
}

PRESET_NAMES = list(BONE_PRESETS.keys())


def resolve_preset(user_text: str) -> str | None:
    """
    Match a user's free-text bone strength description to a preset name.
    Returns the preset key (e.g. 'normal') or None if no match found.
    """
    text = user_text.lower()
    for key, preset in BONE_PRESETS.items():
        if key in text:
            return key
        for kw in preset["keywords"]:
            if kw in text:
                return key
    return None


def get_preset_summary() -> str:
    """Return a human-readable summary of all presets including material values."""
    lines = []
    for key, preset in BONE_PRESETS.items():
        m = preset["material"]
        lines.append(
            f"  - {key} ({preset['label']}): {preset['description']}\n"
            f"    E1={m['E1']}, E2={m['E2']}, E3={m['E3']} GPa | "
            f"G12={m['G12']}, G23={m['G23']}, G31={m['G31']} GPa | "
            f"NU12={m['NU12']}, NU23={m['NU23']}, NU31={m['NU31']}"
        )
    return "\n".join(lines)
