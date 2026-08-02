"""Convert a photographed dental prescription into a DentNode entry payload.

The vision extraction is shared with ``case_from_image``.  This module owns the
deterministic second half: translating the model's portable output into the
shape accepted by app.dentnode.com's ``POST /entry/create`` endpoint.

No database or Node API calls are made here.  Doctor and product IDs are
tenant-specific and cannot be inferred from an image, so their detected names
are returned as resolution hints and the payload is deliberately marked as not
ready until the caller resolves them.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .case_from_image import CaseExtractionResult, extract_case_from_image


def _text(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _tooth_numbers(raw: Any) -> List[str]:
    """Return unique, valid permanent FDI numbers without inventing teeth."""
    if not isinstance(raw, list):
        return []
    result: List[str] = []
    for value in raw:
        tooth = str(value).strip()
        if (
            len(tooth) == 2
            and tooth[0] in "1234"
            and tooth[1] in "12345678"
            and tooth not in result
        ):
            result.append(tooth)
    return result


def build_tooth_chart(teeth: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Build the JSON emitted by DentNode's helpers/tooth/createChart.ts."""
    selected = set(_tooth_numbers(teeth))
    chart: Dict[str, List[Dict[str, Any]]] = {}
    for quadrant in range(1, 5):
        names = list(range(1, 9))
        if quadrant % 2 == 1:
            names.reverse()
        chart[str(quadrant - 1)] = [
            {
                "name": name,
                "isSelected": f"{quadrant}{name}" in selected,
            }
            for name in names
        ]
    return chart


def _jaw_type(raw: Any, teeth: List[str]) -> str:
    value = (_text(raw) or "").upper()
    aliases = {
        "UPPER": "UPPER",
        "UPPER_ARCH": "UPPER",
        "LOWER": "LOWER",
        "LOWER_ARCH": "LOWER",
        "BOTH": "FULL",
        "FULL": "FULL",
        "FULL_ARCH": "FULL",
    }
    if value in aliases:
        return aliases[value]
    quadrants = {tooth[0] for tooth in teeth}
    has_upper = bool(quadrants & {"1", "2"})
    has_lower = bool(quadrants & {"3", "4"})
    if has_upper and has_lower:
        return "FULL"
    if has_upper:
        return "UPPER"
    if has_lower:
        return "LOWER"
    return "NA"


def _shade_guide(raw: Any) -> str:
    value = (_text(raw) or "").upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "VITA_CLASSIC": "VITA_CLASSIC",
        "VITA_3D": "VITA_3D_MASTER",
        "VITA_3D_MASTER": "VITA_3D_MASTER",
        "VITA_CLASSIC_WITH_BLEACH": "VITA_CLASSIC_WITH_BLEACH",
        "VITA_3D_MASTER_WITH_BLEACH": "VITA_3D_MASTER_WITH_BLEACH",
        "VITA_LINEARGUIDE_3D_MASTER": "VITA_LINEARGUIDE_3D_MASTER",
        "NO_SHADE": "NO_SHADE",
    }
    return aliases.get(value, "VITA_CLASSIC")


def _positive_units(raw: Any) -> int:
    try:
        units = int(raw)
    except (TypeError, ValueError):
        return 1
    return units if units > 0 else 1


def build_entry_payload(extracted: Dict[str, Any]) -> Dict[str, Any]:
    """Translate portable vision output into a safe DentNode draft payload."""
    doctor = extracted.get("doctor") if isinstance(extracted.get("doctor"), dict) else {}
    patient = extracted.get("patient") if isinstance(extracted.get("patient"), dict) else {}
    case = extracted.get("case") if isinstance(extracted.get("case"), dict) else {}
    raw_work = extracted.get("work") if isinstance(extracted.get("work"), list) else []

    work: List[Dict[str, Any]] = []
    detected_work: List[Dict[str, Any]] = []
    unresolved_products: List[Dict[str, Any]] = []
    for index, item in enumerate(raw_work):
        if not isinstance(item, dict):
            continue
        teeth = _tooth_numbers(item.get("teeth"))
        product_name = _text(item.get("product_name"))
        work.append(
            {
                "categoryId": None,
                "productId": None,
                "tooth": build_tooth_chart(teeth),
                "shade": _text(item.get("shade")) or "NA",
                "shade_guide": _shade_guide(item.get("shade_guide")),
                "total_units": _positive_units(item.get("units")),
                "jaw_type": _jaw_type(item.get("jaw_type"), teeth),
                "instructions": _text(item.get("instructions")),
            }
        )
        detected_work.append({"product_name": product_name, "teeth": teeth})
        unresolved_products.append(
            {"work_index": len(work) - 1, "name": product_name}
        )

    entry_type = (_text(case.get("entry_type")) or "PHYSICAL").upper()
    if entry_type not in {"PHYSICAL", "DIGITAL"}:
        entry_type = "PHYSICAL"

    patient_payload = {
        key: value
        for key, value in {
            "first_name": _text(patient.get("first_name")) or "",
            "last_name": _text(patient.get("last_name")) or "",
            "phone": _text(patient.get("phone")),
            "gender": (_text(patient.get("gender")) or "").upper() or None,
            "age": patient.get("age") if isinstance(patient.get("age"), int) else None,
        }.items()
        if value is not None
    }
    patient_payload["doctor_id"] = None
    if patient_payload.get("gender") not in {"MALE", "FEMALE"}:
        patient_payload.pop("gender", None)

    doctor_name = _text(doctor.get("name"))
    entry_payload: Dict[str, Any] = {
        "entry": {
            "doctor_id": None,
            "entry_type": entry_type,
            "creation_type": "NEW",
            "status": "ORDERED",
            "submission_status": "PENDING",
            "payment_status": "UNPAID",
            "priority": "MEDIUM",
            "case_custom_id": _text(case.get("case_id")),
            # Keep DentNode's historical schema spelling; /entry/create expects it.
            "epxected_delivery_date": _text(case.get("delivery_date")),
            "notes": _text(extracted.get("notes")),
        },
        "patient": patient_payload,
        "work": work,
    }

    unresolved: Dict[str, Any] = {
        "doctor": {"name": doctor_name, "phone": _text(doctor.get("phone"))},
        "products": unresolved_products,
    }
    missing = ["entry.doctor_id", "patient.doctor_id"]
    if not doctor_name:
        missing.append("detected.doctor.name")
    missing.extend(f"work[{item['work_index']}].productId" for item in unresolved_products)
    if not work:
        missing.append("work")

    return {
        "entry_payload": entry_payload,
        "detected": {
            "doctor": unresolved["doctor"],
            "work": detected_work,
        },
        "resolution_required": unresolved,
        "missing_fields": missing,
        "ready_to_create": False,
    }


@dataclass
class ImageToEntryResult:
    payload: Dict[str, Any]
    model: str
    usage: Dict[str, int]
    cost_usd: Optional[float]
    latency_ms: int


async def image_to_entry(*, image_base64: str, mime_type: str) -> ImageToEntryResult:
    extraction: CaseExtractionResult = await extract_case_from_image(
        image_base64=image_base64,
        mime_type=mime_type,
    )
    return ImageToEntryResult(
        payload=build_entry_payload(extraction.extracted),
        model=extraction.model,
        usage=extraction.usage,
        cost_usd=extraction.cost_usd,
        latency_ms=extraction.latency_ms,
    )
