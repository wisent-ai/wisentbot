#!/usr/bin/env python3
"""Tests for PaymentSkill - invoice management, payment processing, and revenue tracking."""

import json
import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch

from singularity.skills.payment import PaymentSkill, PAYMENT_FILE, DEFAULT_PRICING


@pytest.fixture
def skill(tmp_path):
    """Create a PaymentSkill with temp data directory."""
    test_file = tmp_path / "payments.json"
    with patch("singularity.skills.payment.PAYMENT_FILE", test_file):
        s = PaymentSkill()
        yield s


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestInvoiceCreation:
    def test_create_basic_invoice(self, skill):
        result = run(skill.execute("create_invoice", {
            "customer_email": "alice@example.com",
            "service": "code_review",
        }))
        assert result.success
        inv = result.data["invoice"]
        assert inv["id"].startswith("INV-")
        assert inv["customer_email"] == "alice@example.com"
        assert inv["status"] == "draft"
        assert inv["amount"] > 0

    def test_create_invoice_custom_amount(self, skill):
        result = run(skill.execute("create_invoice", {
            "customer_email": "bob@test.com",
            "service": "custom",
            "amount": 42.50,
            "description": "Custom analysis",
        }))
        assert result.success
        assert result.data["invoice"]["amount"] == 42.50

    def test_create_invoice_missing_email(self, skill):
        result = run(skill.execute("create_invoice", {"service": "code_review"}))
        assert not result.success

    def test_create_invoice_invalid_currency(self, skill):
        result = run(skill.execute("create_invoice", {
            "customer_email": "x@y.com", "service": "custom", "currency": "BTC",
        }))
        assert not result.success


class TestPaymentRecording:
    def test_record_payment(self, skill):
        # Create invoice first
        inv = run(skill.execute("create_invoice", {
            "customer_email": "payer@co.com", "service": "code_review", "amount": 25.0,
        }))
        inv_id = inv.data["invoice"]["id"]
        # Record payment
        result = run(skill.execute("record_payment", {"invoice_id": inv_id}))
        assert result.success
        assert result.revenue == 25.0
        assert result.data["invoice"]["status"] == "paid"

    def test_cannot_pay_twice(self, skill):
        inv = run(skill.execute("create_invoice", {
            "customer_email": "p@c.com", "service": "custom", "amount": 10,
        }))
        inv_id = inv.data["invoice"]["id"]
        run(skill.execute("record_payment", {"invoice_id": inv_id}))
        result = run(skill.execute("record_payment", {"invoice_id": inv_id}))
        assert not result.success

    def test_pay_nonexistent_invoice(self, skill):
        result = run(skill.execute("record_payment", {"invoice_id": "INV-FAKE"}))
        assert not result.success


class TestRefunds:
    def test_full_refund(self, skill):
        inv = run(skill.execute("create_invoice", {
            "customer_email": "r@x.com", "service": "custom", "amount": 30,
        }))
        inv_id = inv.data["invoice"]["id"]
        run(skill.execute("record_payment", {"invoice_id": inv_id}))
        result = run(skill.execute("refund", {"invoice_id": inv_id, "reason": "not satisfied"}))
        assert result.success
        assert result.data["invoice"]["status"] == "refunded"
        assert result.data["full_refund"] is True

    def test_partial_refund(self, skill):
        inv = run(skill.execute("create_invoice", {
            "customer_email": "r@x.com", "service": "custom", "amount": 50,
        }))
        inv_id = inv.data["invoice"]["id"]
        run(skill.execute("record_payment", {"invoice_id": inv_id}))
        result = run(skill.execute("refund", {"invoice_id": inv_id, "amount": 20}))
        assert result.success
        assert result.data["full_refund"] is False

    def test_cannot_refund_unpaid(self, skill):
        inv = run(skill.execute("create_invoice", {
            "customer_email": "r@x.com", "service": "custom", "amount": 10,
        }))
        result = run(skill.execute("refund", {"invoice_id": inv.data["invoice"]["id"]}))
        assert not result.success


class TestPricingAndReporting:
    def test_calculate_price(self, skill):
        result = run(skill.execute("calculate_price", {
            "service": "code_review", "quantity": 100,
        }))
        assert result.success
        assert result.data["pricing"]["total"] > 0

    def test_update_pricing(self, skill):
        result = run(skill.execute("update_pricing", {
            "service": "code_review", "base": 15.0,
        }))
        assert result.success

    def test_financial_report(self, skill):
        # Create some data
        run(skill.execute("create_invoice", {
            "customer_email": "a@b.com", "service": "custom", "amount": 100,
        }))
        result = run(skill.execute("financial_report", {"period_days": 7}))
        assert result.success
        report = result.data["report"]
        assert "revenue" in report
        assert "invoices" in report
        assert report["invoices"]["created"] == 1

    def test_customer_info_create(self, skill):
        result = run(skill.execute("customer_info", {
            "email": "new@customer.com", "name": "New Customer",
        }))
        assert result.success
        assert result.data.get("new") is True

    def test_list_invoices(self, skill):
        run(skill.execute("create_invoice", {
            "customer_email": "a@b.com", "service": "custom", "amount": 10,
        }))
        run(skill.execute("create_invoice", {
            "customer_email": "c@d.com", "service": "custom", "amount": 20,
        }))
        result = run(skill.execute("list_invoices", {}))
        assert result.success
        assert result.data["summary"]["total"] == 2

    def test_payment_link_without_stripe(self, skill):
        inv = run(skill.execute("create_invoice", {
            "customer_email": "a@b.com", "service": "custom", "amount": 10,
        }))
        inv_id = inv.data["invoice"]["id"]
        result = run(skill.execute("create_payment_link", {"invoice_id": inv_id}))
        assert result.success
        assert "payment_link" in result.data
        assert result.data["stripe_available"] is False
