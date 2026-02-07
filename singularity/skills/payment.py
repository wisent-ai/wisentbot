#!/usr/bin/env python3
"""
Payment Skill - Accept payments and manage billing for agent services.

This is the critical revenue infrastructure that connects service delivery
to actual income. Without a payment skill, the agent can offer services
(code review, data analysis, etc.) but has no way to get paid for them.

Capabilities:
- Create invoices for completed work
- Generate Stripe payment links (when API key available)
- Track payment status and revenue
- Manage customer accounts
- Calculate pricing based on work complexity
- Record payment history for financial reporting
- Support multiple currencies

Architecture:
  PaymentSkill manages the full billing lifecycle:
  1. Agent completes work via RevenueServiceSkill
  2. PaymentSkill creates an invoice with line items
  3. Invoice generates a payment link (Stripe) or records manual payment
  4. Payment confirmation updates revenue tracking
  5. Financial reports show earnings vs costs

Part of the Revenue Generation pillar: the money collection mechanism.
"""

import json
import uuid
import hashlib
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from .base import Skill, SkillResult, SkillManifest, SkillAction


PAYMENT_FILE = Path(__file__).parent.parent / "data" / "payments.json"
MAX_INVOICES = 1000
MAX_CUSTOMERS = 500
MAX_TRANSACTIONS = 2000


# Service pricing tiers (USD)
DEFAULT_PRICING = {
    "code_review": {"base": 5.00, "per_line": 0.01, "max": 50.00},
    "summarize_text": {"base": 2.00, "per_word": 0.001, "max": 20.00},
    "analyze_data": {"base": 10.00, "per_row": 0.005, "max": 100.00},
    "seo_audit": {"base": 8.00, "per_page": 1.00, "max": 50.00},
    "generate_api_docs": {"base": 5.00, "per_endpoint": 2.00, "max": 75.00},
    "custom": {"base": 10.00, "max": 200.00},
}

INVOICE_STATUSES = ["draft", "sent", "paid", "overdue", "cancelled", "refunded"]
CURRENCIES = ["USD", "EUR", "GBP"]


class PaymentSkill(Skill):
    """
    Payment processing and invoice management for agent services.

    Enables the agent to:
    - Create and send invoices for completed work
    - Generate payment links via Stripe integration
    - Track customer payment history
    - Calculate dynamic pricing based on work complexity
    - Generate financial reports showing revenue vs costs
    - Manage refunds and cancellations
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._stripe_key = (credentials or {}).get("STRIPE_SECRET_KEY", "")
        self._stripe_available = bool(self._stripe_key)
        self._ensure_data()

    def _ensure_data(self):
        PAYMENT_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not PAYMENT_FILE.exists():
            self._save(self._default_state())

    def _default_state(self) -> Dict:
        return {
            "invoices": [],
            "customers": {},
            "transactions": [],
            "pricing": dict(DEFAULT_PRICING),
            "stats": {
                "total_revenue": 0.0,
                "total_invoiced": 0.0,
                "total_refunded": 0.0,
                "invoices_created": 0,
                "invoices_paid": 0,
                "currency": "USD",
            },
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        }

    def _load(self) -> Dict:
        try:
            with open(PAYMENT_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return self._default_state()

    def _save(self, data: Dict):
        data["last_updated"] = datetime.now().isoformat()
        # Cap stored data
        if len(data.get("invoices", [])) > MAX_INVOICES:
            data["invoices"] = data["invoices"][-MAX_INVOICES:]
        if len(data.get("transactions", [])) > MAX_TRANSACTIONS:
            data["transactions"] = data["transactions"][-MAX_TRANSACTIONS:]
        with open(PAYMENT_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="payment",
            name="Payment & Billing",
            version="1.0.0",
            category="revenue",
            description="Invoice management, payment processing, and revenue tracking",
            actions=[
                SkillAction(
                    name="create_invoice",
                    description="Create an invoice for completed work",
                    parameters={
                        "customer_email": {"type": "string", "required": True,
                                          "description": "Customer email address"},
                        "service": {"type": "string", "required": True,
                                   "description": "Service type (code_review, summarize_text, etc.)"},
                        "description": {"type": "string", "required": False,
                                       "description": "Work description"},
                        "amount": {"type": "number", "required": False,
                                  "description": "Override amount (auto-calculated if omitted)"},
                        "quantity": {"type": "number", "required": False,
                                    "description": "Units of work (lines, words, rows, etc.)"},
                        "currency": {"type": "string", "required": False,
                                    "description": "Currency code (default: USD)"},
                        "due_days": {"type": "number", "required": False,
                                    "description": "Days until due (default: 7)"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="get_invoice",
                    description="Get details of a specific invoice",
                    parameters={
                        "invoice_id": {"type": "string", "required": True,
                                      "description": "Invoice ID"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="list_invoices",
                    description="List invoices with optional filters",
                    parameters={
                        "status": {"type": "string", "required": False,
                                  "description": "Filter by status (draft, sent, paid, etc.)"},
                        "customer_email": {"type": "string", "required": False,
                                          "description": "Filter by customer"},
                        "limit": {"type": "number", "required": False,
                                 "description": "Max results (default: 20)"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="record_payment",
                    description="Record a payment received for an invoice",
                    parameters={
                        "invoice_id": {"type": "string", "required": True,
                                      "description": "Invoice ID being paid"},
                        "amount": {"type": "number", "required": False,
                                  "description": "Amount paid (defaults to invoice total)"},
                        "method": {"type": "string", "required": False,
                                  "description": "Payment method (stripe, crypto, manual)"},
                        "reference": {"type": "string", "required": False,
                                     "description": "External payment reference/txn ID"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="create_payment_link",
                    description="Generate a Stripe payment link for an invoice",
                    parameters={
                        "invoice_id": {"type": "string", "required": True,
                                      "description": "Invoice to create link for"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="calculate_price",
                    description="Calculate price for a service based on complexity",
                    parameters={
                        "service": {"type": "string", "required": True,
                                   "description": "Service type"},
                        "quantity": {"type": "number", "required": False,
                                    "description": "Units (lines, words, rows, etc.)"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="update_pricing",
                    description="Update pricing for a service type",
                    parameters={
                        "service": {"type": "string", "required": True,
                                   "description": "Service type to update"},
                        "base": {"type": "number", "required": False,
                                "description": "Base price in USD"},
                        "per_unit": {"type": "number", "required": False,
                                    "description": "Per-unit price (line, word, row)"},
                        "max": {"type": "number", "required": False,
                               "description": "Maximum price cap"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="customer_info",
                    description="Get or create customer information",
                    parameters={
                        "email": {"type": "string", "required": True,
                                 "description": "Customer email"},
                        "name": {"type": "string", "required": False,
                                "description": "Customer name (for creation)"},
                        "company": {"type": "string", "required": False,
                                   "description": "Company name (for creation)"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="refund",
                    description="Process a refund for a paid invoice",
                    parameters={
                        "invoice_id": {"type": "string", "required": True,
                                      "description": "Invoice ID to refund"},
                        "reason": {"type": "string", "required": False,
                                  "description": "Reason for refund"},
                        "amount": {"type": "number", "required": False,
                                  "description": "Partial refund amount (full if omitted)"},
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="financial_report",
                    description="Generate financial summary report",
                    parameters={
                        "period_days": {"type": "number", "required": False,
                                       "description": "Report period in days (default: 30)"},
                    },
                    estimated_cost=0,
                ),
            ],
            required_credentials=["STRIPE_SECRET_KEY"],
        )

    def check_credentials(self) -> bool:
        # Payment skill works without Stripe (manual payments still work)
        return True

    async def execute(self, action: str, params: Dict) -> SkillResult:
        actions = {
            "create_invoice": self._create_invoice,
            "get_invoice": self._get_invoice,
            "list_invoices": self._list_invoices,
            "record_payment": self._record_payment,
            "create_payment_link": self._create_payment_link,
            "calculate_price": self._calculate_price,
            "update_pricing": self._update_pricing,
            "customer_info": self._customer_info,
            "refund": self._refund,
            "financial_report": self._financial_report,
        }
        handler = actions.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        return await handler(params)

    async def initialize(self) -> bool:
        return True

    # --- Invoice Management ---

    async def _create_invoice(self, params: Dict) -> SkillResult:
        """Create a new invoice for completed work."""
        customer_email = params.get("customer_email", "").strip()
        if not customer_email:
            return SkillResult(success=False, message="customer_email is required")

        service = params.get("service", "custom").strip()
        description = params.get("description", f"{service} service")
        currency = params.get("currency", "USD").upper()
        due_days = int(params.get("due_days", 7))
        quantity = params.get("quantity")

        if currency not in CURRENCIES:
            return SkillResult(success=False,
                             message=f"Unsupported currency: {currency}. Use: {CURRENCIES}")

        # Calculate or use provided amount
        amount = params.get("amount")
        if amount is None:
            price_info = self._compute_price(service, quantity)
            amount = price_info["total"]
        else:
            amount = float(amount)

        if amount <= 0:
            return SkillResult(success=False, message="Amount must be positive")

        data = self._load()

        # Ensure customer exists
        if customer_email not in data["customers"]:
            data["customers"][customer_email] = {
                "email": customer_email,
                "name": params.get("customer_name", ""),
                "company": "",
                "created_at": datetime.now().isoformat(),
                "total_invoiced": 0.0,
                "total_paid": 0.0,
                "invoice_count": 0,
            }

        invoice_id = f"INV-{uuid.uuid4().hex[:8].upper()}"
        due_date = (datetime.now() + timedelta(days=due_days)).isoformat()

        invoice = {
            "id": invoice_id,
            "customer_email": customer_email,
            "service": service,
            "description": description,
            "amount": amount,
            "currency": currency,
            "status": "draft",
            "due_date": due_date,
            "created_at": datetime.now().isoformat(),
            "paid_at": None,
            "payment_method": None,
            "payment_reference": None,
            "payment_link": None,
            "line_items": [{
                "description": description,
                "quantity": quantity or 1,
                "unit_price": amount / (quantity or 1),
                "total": amount,
            }],
            "refunded": False,
            "refund_amount": 0.0,
        }

        data["invoices"].append(invoice)
        data["stats"]["total_invoiced"] += amount
        data["stats"]["invoices_created"] += 1

        # Update customer stats
        data["customers"][customer_email]["total_invoiced"] += amount
        data["customers"][customer_email]["invoice_count"] += 1

        self._save(data)

        return SkillResult(
            success=True,
            message=f"Invoice {invoice_id} created for ${amount:.2f} {currency}",
            data={"invoice": invoice},
            revenue=0,  # Not yet paid
        )

    async def _get_invoice(self, params: Dict) -> SkillResult:
        """Get details of a specific invoice."""
        invoice_id = params.get("invoice_id", "").strip()
        if not invoice_id:
            return SkillResult(success=False, message="invoice_id is required")

        data = self._load()
        invoice = next((inv for inv in data["invoices"] if inv["id"] == invoice_id), None)

        if not invoice:
            return SkillResult(success=False, message=f"Invoice {invoice_id} not found")

        # Check if overdue
        if invoice["status"] == "sent":
            due = datetime.fromisoformat(invoice["due_date"])
            if datetime.now() > due:
                invoice["status"] = "overdue"
                self._save(data)

        return SkillResult(
            success=True,
            message=f"Invoice {invoice_id}: {invoice['status']} - ${invoice['amount']:.2f}",
            data={"invoice": invoice},
        )

    async def _list_invoices(self, params: Dict) -> SkillResult:
        """List invoices with optional filters."""
        data = self._load()
        invoices = data.get("invoices", [])

        # Apply filters
        status_filter = params.get("status")
        if status_filter:
            invoices = [inv for inv in invoices if inv["status"] == status_filter]

        customer_filter = params.get("customer_email")
        if customer_filter:
            invoices = [inv for inv in invoices
                       if inv["customer_email"] == customer_filter]

        limit = int(params.get("limit", 20))
        invoices = invoices[-limit:]  # Most recent

        summary = {
            "total": len(invoices),
            "by_status": {},
        }
        for inv in invoices:
            status = inv["status"]
            summary["by_status"][status] = summary["by_status"].get(status, 0) + 1

        return SkillResult(
            success=True,
            message=f"Found {len(invoices)} invoices",
            data={"invoices": invoices, "summary": summary},
        )

    # --- Payment Processing ---

    async def _record_payment(self, params: Dict) -> SkillResult:
        """Record a payment received for an invoice."""
        invoice_id = params.get("invoice_id", "").strip()
        if not invoice_id:
            return SkillResult(success=False, message="invoice_id is required")

        data = self._load()
        invoice = next((inv for inv in data["invoices"] if inv["id"] == invoice_id), None)

        if not invoice:
            return SkillResult(success=False, message=f"Invoice {invoice_id} not found")

        if invoice["status"] == "paid":
            return SkillResult(success=False, message=f"Invoice {invoice_id} already paid")

        if invoice["status"] == "cancelled":
            return SkillResult(success=False, message=f"Invoice {invoice_id} is cancelled")

        amount = float(params.get("amount", invoice["amount"]))
        method = params.get("method", "manual")
        reference = params.get("reference", f"MANUAL-{uuid.uuid4().hex[:8]}")

        # Mark as paid
        invoice["status"] = "paid"
        invoice["paid_at"] = datetime.now().isoformat()
        invoice["payment_method"] = method
        invoice["payment_reference"] = reference

        # Record transaction
        transaction = {
            "id": f"TXN-{uuid.uuid4().hex[:8].upper()}",
            "invoice_id": invoice_id,
            "type": "payment",
            "amount": amount,
            "currency": invoice["currency"],
            "method": method,
            "reference": reference,
            "customer_email": invoice["customer_email"],
            "timestamp": datetime.now().isoformat(),
        }
        data["transactions"].append(transaction)

        # Update stats
        data["stats"]["total_revenue"] += amount
        data["stats"]["invoices_paid"] += 1

        # Update customer stats
        customer = data["customers"].get(invoice["customer_email"])
        if customer:
            customer["total_paid"] += amount

        self._save(data)

        return SkillResult(
            success=True,
            message=f"Payment of ${amount:.2f} recorded for invoice {invoice_id}",
            data={"transaction": transaction, "invoice": invoice},
            revenue=amount,
        )

    async def _create_payment_link(self, params: Dict) -> SkillResult:
        """Generate a Stripe payment link for an invoice."""
        invoice_id = params.get("invoice_id", "").strip()
        if not invoice_id:
            return SkillResult(success=False, message="invoice_id is required")

        data = self._load()
        invoice = next((inv for inv in data["invoices"] if inv["id"] == invoice_id), None)

        if not invoice:
            return SkillResult(success=False, message=f"Invoice {invoice_id} not found")

        if invoice["status"] == "paid":
            return SkillResult(success=False, message="Invoice already paid")

        if self._stripe_available:
            # Create real Stripe payment link
            try:
                link_result = await self._create_stripe_link(invoice)
                if link_result["success"]:
                    invoice["payment_link"] = link_result["url"]
                    invoice["status"] = "sent"
                    self._save(data)
                    return SkillResult(
                        success=True,
                        message=f"Payment link created: {link_result['url']}",
                        data={"payment_link": link_result["url"], "invoice": invoice},
                    )
                else:
                    return SkillResult(success=False, message=link_result.get("error", "Stripe error"))
            except Exception as e:
                return SkillResult(success=False, message=f"Stripe error: {str(e)}")
        else:
            # Generate a placeholder link with invoice reference
            link_hash = hashlib.sha256(
                f"{invoice_id}:{invoice['amount']}:{int(time.time())}".encode()
            ).hexdigest()[:16]
            placeholder_link = f"https://pay.singularity.ai/invoice/{invoice_id}?ref={link_hash}"

            invoice["payment_link"] = placeholder_link
            invoice["status"] = "sent"
            self._save(data)

            return SkillResult(
                success=True,
                message=(f"Payment link generated (Stripe not configured, using placeholder): "
                        f"{placeholder_link}"),
                data={
                    "payment_link": placeholder_link,
                    "invoice": invoice,
                    "stripe_available": False,
                    "note": "Configure STRIPE_SECRET_KEY for real payment processing",
                },
            )

    async def _create_stripe_link(self, invoice: Dict) -> Dict:
        """Create a real Stripe payment link using the API."""
        try:
            import stripe
            stripe.api_key = self._stripe_key

            # Create a price object
            price = stripe.Price.create(
                unit_amount=int(invoice["amount"] * 100),  # cents
                currency=invoice["currency"].lower(),
                product_data={
                    "name": f"Singularity AI - {invoice['service']}",
                    "description": invoice.get("description", "AI service"),
                },
            )

            # Create payment link
            link = stripe.PaymentLink.create(
                line_items=[{"price": price.id, "quantity": 1}],
                metadata={
                    "invoice_id": invoice["id"],
                    "customer_email": invoice["customer_email"],
                },
                after_completion={
                    "type": "redirect",
                    "redirect": {"url": "https://singularity.ai/payment/success"},
                },
            )

            return {"success": True, "url": link.url}
        except ImportError:
            return {"success": False, "error": "stripe package not installed"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # --- Pricing ---

    def _compute_price(self, service: str, quantity: Optional[float] = None) -> Dict:
        """Compute price for a service based on pricing rules."""
        data = self._load()
        pricing = data.get("pricing", DEFAULT_PRICING)
        tier = pricing.get(service, pricing.get("custom", {"base": 10.00, "max": 200.00}))

        base = tier.get("base", 10.00)
        max_price = tier.get("max", 200.00)

        if quantity and quantity > 0:
            # Find per-unit key
            per_unit_key = None
            for key in tier:
                if key.startswith("per_"):
                    per_unit_key = key
                    break

            if per_unit_key:
                per_unit = tier[per_unit_key]
                total = base + (quantity * per_unit)
            else:
                total = base
        else:
            total = base

        total = min(total, max_price)
        total = round(total, 2)

        return {
            "service": service,
            "base": base,
            "quantity": quantity,
            "total": total,
            "max": max_price,
            "currency": "USD",
        }

    async def _calculate_price(self, params: Dict) -> SkillResult:
        """Calculate price for a service."""
        service = params.get("service", "custom")
        quantity = params.get("quantity")
        if quantity is not None:
            quantity = float(quantity)

        price_info = self._compute_price(service, quantity)

        return SkillResult(
            success=True,
            message=f"Price for {service}: ${price_info['total']:.2f} USD",
            data={"pricing": price_info},
        )

    async def _update_pricing(self, params: Dict) -> SkillResult:
        """Update pricing for a service type."""
        service = params.get("service", "").strip()
        if not service:
            return SkillResult(success=False, message="service is required")

        data = self._load()
        pricing = data.get("pricing", dict(DEFAULT_PRICING))

        if service not in pricing:
            pricing[service] = {"base": 10.00, "max": 200.00}

        if "base" in params:
            pricing[service]["base"] = float(params["base"])
        if "per_unit" in params:
            # Determine appropriate per-unit key
            unit_key = f"per_unit"
            for key in pricing[service]:
                if key.startswith("per_"):
                    unit_key = key
                    break
            pricing[service][unit_key] = float(params["per_unit"])
        if "max" in params:
            pricing[service]["max"] = float(params["max"])

        data["pricing"] = pricing
        self._save(data)

        return SkillResult(
            success=True,
            message=f"Pricing updated for {service}",
            data={"service": service, "pricing": pricing[service]},
        )

    # --- Customer Management ---

    async def _customer_info(self, params: Dict) -> SkillResult:
        """Get or create customer information."""
        email = params.get("email", "").strip()
        if not email:
            return SkillResult(success=False, message="email is required")

        data = self._load()

        if email in data["customers"]:
            customer = data["customers"][email]
            # Enrich with invoice history
            customer_invoices = [
                inv for inv in data["invoices"]
                if inv["customer_email"] == email
            ]
            return SkillResult(
                success=True,
                message=f"Customer: {email} ({len(customer_invoices)} invoices)",
                data={
                    "customer": customer,
                    "recent_invoices": customer_invoices[-5:],
                },
            )
        else:
            # Create new customer
            customer = {
                "email": email,
                "name": params.get("name", ""),
                "company": params.get("company", ""),
                "created_at": datetime.now().isoformat(),
                "total_invoiced": 0.0,
                "total_paid": 0.0,
                "invoice_count": 0,
            }
            data["customers"][email] = customer
            self._save(data)

            return SkillResult(
                success=True,
                message=f"Customer created: {email}",
                data={"customer": customer, "new": True},
            )

    # --- Refunds ---

    async def _refund(self, params: Dict) -> SkillResult:
        """Process a refund for a paid invoice."""
        invoice_id = params.get("invoice_id", "").strip()
        if not invoice_id:
            return SkillResult(success=False, message="invoice_id is required")

        data = self._load()
        invoice = next((inv for inv in data["invoices"] if inv["id"] == invoice_id), None)

        if not invoice:
            return SkillResult(success=False, message=f"Invoice {invoice_id} not found")

        if invoice["status"] != "paid":
            return SkillResult(success=False,
                             message=f"Cannot refund: invoice status is '{invoice['status']}' (must be 'paid')")

        if invoice.get("refunded"):
            return SkillResult(success=False, message="Invoice already refunded")

        refund_amount = float(params.get("amount", invoice["amount"]))
        reason = params.get("reason", "Customer request")

        if refund_amount > invoice["amount"]:
            return SkillResult(success=False,
                             message=f"Refund amount ${refund_amount:.2f} exceeds invoice total ${invoice['amount']:.2f}")

        # Process refund
        is_full_refund = abs(refund_amount - invoice["amount"]) < 0.01
        invoice["refunded"] = True
        invoice["refund_amount"] = refund_amount
        if is_full_refund:
            invoice["status"] = "refunded"

        # Record refund transaction
        transaction = {
            "id": f"TXN-{uuid.uuid4().hex[:8].upper()}",
            "invoice_id": invoice_id,
            "type": "refund",
            "amount": -refund_amount,
            "currency": invoice["currency"],
            "method": invoice.get("payment_method", "manual"),
            "reference": f"REFUND-{uuid.uuid4().hex[:8]}",
            "customer_email": invoice["customer_email"],
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
        }
        data["transactions"].append(transaction)

        # Update stats
        data["stats"]["total_refunded"] += refund_amount
        data["stats"]["total_revenue"] -= refund_amount

        # Update customer stats
        customer = data["customers"].get(invoice["customer_email"])
        if customer:
            customer["total_paid"] -= refund_amount

        self._save(data)

        return SkillResult(
            success=True,
            message=f"Refund of ${refund_amount:.2f} processed for invoice {invoice_id}",
            data={
                "transaction": transaction,
                "invoice": invoice,
                "full_refund": is_full_refund,
            },
        )

    # --- Financial Reporting ---

    async def _financial_report(self, params: Dict) -> SkillResult:
        """Generate a financial summary report."""
        period_days = int(params.get("period_days", 30))
        cutoff = (datetime.now() - timedelta(days=period_days)).isoformat()

        data = self._load()

        # Filter to period
        period_invoices = [
            inv for inv in data["invoices"]
            if inv.get("created_at", "") >= cutoff
        ]
        period_transactions = [
            txn for txn in data["transactions"]
            if txn.get("timestamp", "") >= cutoff
        ]

        # Calculate metrics
        revenue = sum(
            txn["amount"] for txn in period_transactions
            if txn["type"] == "payment"
        )
        refunds = sum(
            abs(txn["amount"]) for txn in period_transactions
            if txn["type"] == "refund"
        )
        net_revenue = revenue - refunds

        invoices_created = len(period_invoices)
        invoices_paid = sum(1 for inv in period_invoices if inv["status"] == "paid")
        invoices_outstanding = sum(
            1 for inv in period_invoices
            if inv["status"] in ("draft", "sent", "overdue")
        )
        outstanding_amount = sum(
            inv["amount"] for inv in period_invoices
            if inv["status"] in ("draft", "sent", "overdue")
        )

        # Revenue by service
        revenue_by_service = {}
        for inv in period_invoices:
            if inv["status"] == "paid":
                svc = inv.get("service", "unknown")
                revenue_by_service[svc] = revenue_by_service.get(svc, 0) + inv["amount"]

        # Top customers
        customer_revenue = {}
        for txn in period_transactions:
            if txn["type"] == "payment":
                email = txn.get("customer_email", "unknown")
                customer_revenue[email] = customer_revenue.get(email, 0) + txn["amount"]

        top_customers = sorted(
            customer_revenue.items(), key=lambda x: x[1], reverse=True
        )[:5]

        # Payment rate
        payment_rate = (invoices_paid / invoices_created * 100) if invoices_created > 0 else 0

        report = {
            "period_days": period_days,
            "revenue": {
                "gross": round(revenue, 2),
                "refunds": round(refunds, 2),
                "net": round(net_revenue, 2),
            },
            "invoices": {
                "created": invoices_created,
                "paid": invoices_paid,
                "outstanding": invoices_outstanding,
                "outstanding_amount": round(outstanding_amount, 2),
                "payment_rate_pct": round(payment_rate, 1),
            },
            "revenue_by_service": {k: round(v, 2) for k, v in revenue_by_service.items()},
            "top_customers": [
                {"email": email, "revenue": round(rev, 2)}
                for email, rev in top_customers
            ],
            "all_time": {
                "total_revenue": round(data["stats"]["total_revenue"], 2),
                "total_invoiced": round(data["stats"]["total_invoiced"], 2),
                "total_refunded": round(data["stats"]["total_refunded"], 2),
                "total_customers": len(data["customers"]),
            },
        }

        return SkillResult(
            success=True,
            message=(f"Financial report ({period_days}d): "
                    f"Net revenue ${net_revenue:.2f}, "
                    f"{invoices_paid}/{invoices_created} invoices paid "
                    f"({payment_rate:.0f}% rate)"),
            data={"report": report},
        )
