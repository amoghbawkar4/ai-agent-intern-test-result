import json
import re
from pathlib import Path
from typing import Any


ORDERS_FILE = (
    Path(__file__).resolve().parents[2] / "data" / "orders.json"
)

# Fields that are explicitly safe to expose to the customer.
SAFE_ORDER_FIELDS = {
    "order_id",
    "status",
    "carrier",
    "tracking_number",
    "estimated_delivery",
    "customer_safe_message",
}


def _load_orders() -> dict[str, dict[str, Any]]:
    """Load orders from the supplied JSON dataset."""

    with ORDERS_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return {
        order["order_id"]: order
        for order in data["orders"]
    }


def normalize_order_id(order_id: str) -> str:
    """
    Normalize harmless user input differences.

    Examples:
        "ord-1007"      -> "ORD-1007"
        " ORD-1007 "    -> "ORD-1007"
        "  ord-1007  "  -> "ORD-1007"
    """

    return order_id.strip().upper()


def _is_valid_order_id(order_id: str) -> bool:
    """Validate the expected order ID format."""

    return bool(re.fullmatch(r"ORD-\d{4}", order_id))


def _sanitize_order(order: dict[str, Any]) -> dict[str, Any]:
    """
    Return only customer-safe order information.

    Internal fields such as customer email, address, risk score,
    warehouse notes, and support tags are deliberately excluded.
    """

    result = {
        key: order[key]
        for key in SAFE_ORDER_FIELDS
        if key in order
    }

    # Cancelled/returned orders must not expose stale delivery fields.
    if order["status"] in {"cancelled", "returned"}:
        result.pop("carrier", None)
        result.pop("tracking_number", None)
        result.pop("estimated_delivery", None)

    return result


def lookup_order(order_id: str | None) -> dict[str, Any]:
    """
    Look up an order and return a sanitized customer-safe result.

    The function does not expose the raw order record.
    """

    if order_id is None or not order_id.strip():
        return {
            "found": False,
            "error": "missing_order_id",
            "message": "Please provide your order ID so I can check the order.",
        }

    normalized_id = normalize_order_id(order_id)

    if not _is_valid_order_id(normalized_id):
        return {
            "found": False,
            "error": "invalid_order_id",
            "order_id": normalized_id,
            "message": (
                "That does not look like a valid order ID. "
                "Please check the order ID and try again."
            ),
        }

    orders = _load_orders()
    order = orders.get(normalized_id)

    if order is None:
        return {
            "found": False,
            "error": "order_not_found",
            "order_id": normalized_id,
            "message": (
                f"Order {normalized_id} was not found. "
                "Please check the order ID or contact support."
            ),
        }

    return {
        "found": True,
        "order": _sanitize_order(order),
    }