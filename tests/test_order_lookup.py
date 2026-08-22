from app.tools.order_lookup import lookup_order


def test_valid_order_lookup():
    result = lookup_order("ORD-1007")

    assert result["found"] is True
    assert result["order"]["order_id"] == "ORD-1007"
    assert result["order"]["status"] == "shipped"
    assert result["order"]["carrier"] == "UPS"
    assert result["order"]["estimated_delivery"] == "2026-08-22"


def test_order_id_is_normalized():
    result = lookup_order("  ord-1007 ")

    assert result["found"] is True
    assert result["order"]["order_id"] == "ORD-1007"


def test_missing_order_id_does_not_call_lookup():
    result = lookup_order(None)

    assert result["found"] is False
    assert result["error"] == "missing_order_id"


def test_empty_order_id():
    result = lookup_order("   ")

    assert result["found"] is False
    assert result["error"] == "missing_order_id"


def test_malformed_order_id():
    result = lookup_order("banana")

    assert result["found"] is False
    assert result["error"] == "invalid_order_id"


def test_unknown_order():
    result = lookup_order("ORD-9999")

    assert result["found"] is False
    assert result["error"] == "order_not_found"
    assert "ORD-9999" in result["message"]


def test_cancelled_order_does_not_expose_stale_shipping_data():
    result = lookup_order("ORD-1004")

    assert result["found"] is True
    assert result["order"]["status"] == "cancelled"

    assert "carrier" not in result["order"]
    assert "tracking_number" not in result["order"]
    assert "estimated_delivery" not in result["order"]


def test_shipped_order_without_eta_does_not_invent_eta():
    result = lookup_order("ORD-1011")

    assert result["found"] is True
    assert result["order"]["status"] == "shipped"
    assert result["order"]["carrier"] == "Canada Post"
    assert result["order"]["estimated_delivery"] is None


def test_internal_fields_are_never_exposed():
    result = lookup_order("ORD-1007")

    assert result["found"] is True

    order = result["order"]

    assert "email" not in order
    assert "shipping_address" not in order
    assert "internal" not in order
    assert "risk_score" not in order
    assert "warehouse_note" not in order
    assert "support_tags" not in order