#!/usr/bin/env python3
"""
Revenue Skill - Enables agents to define services, track earnings, and manage finances.

This skill gives agents the infrastructure for economic survival:
- Define services they can offer (with pricing)
- Create and track invoices for work performed
- Record payments and expenses
- Generate financial reports (P&L, revenue trends)
- Manage a service catalog that can be advertised

Without revenue tracking, an agent cannot measure its economic viability
or make informed decisions about which activities to pursue.
"""

import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from .base import Skill, SkillManifest, SkillAction, SkillResult


class InvoiceStatus(Enum):
    """Status of an invoice."""
    DRAFT = "draft"
    SENT = "sent"
    PAID = "paid"
    CANCELLED = "cancelled"
    OVERDUE = "overdue"


class ExpenseCategory(Enum):
    """Categories of expenses."""
    COMPUTE = "compute"      # LLM API calls, hosting
    SERVICES = "services"    # Third-party APIs
    CREATION = "creation"    # Agent spawning costs
    OTHER = "other"


@dataclass
class Service:
    """A service the agent can offer."""
    id: str
    name: str
    description: str
    price: float
    currency: str = "USD"
    price_unit: str = "per_task"  # per_task, per_hour, per_word, per_request
    active: bool = True
    created_at: str = ""
    times_sold: int = 0
    total_earned: float = 0.0


@dataclass
class Invoice:
    """An invoice for work performed."""
    id: str
    service_id: str
    service_name: str
    client: str
    amount: float
    currency: str = "USD"
    status: InvoiceStatus = InvoiceStatus.DRAFT
    description: str = ""
    created_at: str = ""
    paid_at: str = ""


@dataclass
class Expense:
    """A recorded expense."""
    id: str
    amount: float
    category: str
    description: str
    created_at: str = ""


class RevenueSkill(Skill):
    """
    Skill for managing revenue, services, invoices, and financial tracking.

    Gives the agent economic awareness and the ability to:
    - Define and price services it can offer
    - Track work done and payments received
    - Monitor expenses and compute costs
    - Generate financial reports for decision-making
    """

    def __init__(self, credentials: Dict = None):
        super().__init__(credentials)
        self._services: Dict[str, Service] = {}
        self._invoices: Dict[str, Invoice] = {}
        self._expenses: List[Expense] = []
        self._payments: List[Dict[str, Any]] = []

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="revenue",
            name="Revenue Management",
            version="1.0.0",
            category="economics",
            description="Define services, track earnings, manage finances for economic survival",
            actions=[
                # === Service Catalog ===
                SkillAction(
                    name="create_service",
                    description="Define a new service you can offer with pricing",
                    parameters={
                        "name": {
                            "type": "string",
                            "required": True,
                            "description": "Name of the service (e.g., 'Code Review', 'Content Writing')"
                        },
                        "description": {
                            "type": "string",
                            "required": True,
                            "description": "What this service provides"
                        },
                        "price": {
                            "type": "number",
                            "required": True,
                            "description": "Price in USD"
                        },
                        "price_unit": {
                            "type": "string",
                            "required": False,
                            "description": "Pricing unit: per_task, per_hour, per_word, per_request (default: per_task)"
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="list_services",
                    description="List all services you offer",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="update_service",
                    description="Update a service's details or pricing",
                    parameters={
                        "service_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the service to update"
                        },
                        "price": {
                            "type": "number",
                            "required": False,
                            "description": "New price"
                        },
                        "active": {
                            "type": "boolean",
                            "required": False,
                            "description": "Whether service is active"
                        },
                        "description": {
                            "type": "string",
                            "required": False,
                            "description": "New description"
                        },
                    },
                    estimated_cost=0,
                ),
                # === Invoicing ===
                SkillAction(
                    name="create_invoice",
                    description="Create an invoice for work done",
                    parameters={
                        "service_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the service provided"
                        },
                        "client": {
                            "type": "string",
                            "required": True,
                            "description": "Client name or identifier"
                        },
                        "amount": {
                            "type": "number",
                            "required": False,
                            "description": "Override amount (defaults to service price)"
                        },
                        "description": {
                            "type": "string",
                            "required": False,
                            "description": "Work description for the invoice"
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="list_invoices",
                    description="List all invoices with optional status filter",
                    parameters={
                        "status": {
                            "type": "string",
                            "required": False,
                            "description": "Filter by status: draft, sent, paid, cancelled, overdue"
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="record_payment",
                    description="Record a payment received for an invoice",
                    parameters={
                        "invoice_id": {
                            "type": "string",
                            "required": True,
                            "description": "ID of the invoice being paid"
                        },
                        "amount": {
                            "type": "number",
                            "required": False,
                            "description": "Amount paid (defaults to invoice amount)"
                        },
                    },
                    estimated_cost=0,
                ),
                # === Expense Tracking ===
                SkillAction(
                    name="record_expense",
                    description="Record an expense (compute costs, API fees, etc.)",
                    parameters={
                        "amount": {
                            "type": "number",
                            "required": True,
                            "description": "Expense amount in USD"
                        },
                        "category": {
                            "type": "string",
                            "required": False,
                            "description": "Category: compute, services, creation, other (default: other)"
                        },
                        "description": {
                            "type": "string",
                            "required": True,
                            "description": "What the expense was for"
                        },
                    },
                    estimated_cost=0,
                ),
                # === Financial Reports ===
                SkillAction(
                    name="financial_report",
                    description="Generate a financial summary: revenue, expenses, profit/loss",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="top_services",
                    description="Show which services generate the most revenue",
                    parameters={},
                    estimated_cost=0,
                ),
            ],
            required_credentials=[],
        )

    def check_credentials(self) -> bool:
        """Revenue tracking needs no external credentials."""
        return True

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "create_service": self._create_service,
            "list_services": self._list_services,
            "update_service": self._update_service,
            "create_invoice": self._create_invoice,
            "list_invoices": self._list_invoices,
            "record_payment": self._record_payment,
            "record_expense": self._record_expense,
            "financial_report": self._financial_report,
            "top_services": self._top_services,
        }

        handler = handlers.get(action)
        if handler:
            return await handler(params)
        return SkillResult(success=False, message=f"Unknown action: {action}")

    # === Service Catalog ===

    async def _create_service(self, params: Dict) -> SkillResult:
        """Define a new service offering."""
        name = params.get("name", "").strip()
        description = params.get("description", "").strip()
        price = params.get("price", 0)
        price_unit = params.get("price_unit", "per_task").strip()

        if not name:
            return SkillResult(success=False, message="Service name is required")
        if not description:
            return SkillResult(success=False, message="Service description is required")
        if price <= 0:
            return SkillResult(success=False, message="Price must be positive")

        valid_units = ["per_task", "per_hour", "per_word", "per_request"]
        if price_unit not in valid_units:
            return SkillResult(
                success=False,
                message=f"Invalid price_unit. Must be one of: {', '.join(valid_units)}"
            )

        service_id = f"svc_{uuid.uuid4().hex[:8]}"
        service = Service(
            id=service_id,
            name=name,
            description=description,
            price=price,
            price_unit=price_unit,
            created_at=datetime.now().isoformat(),
        )
        self._services[service_id] = service

        return SkillResult(
            success=True,
            message=f"Service '{name}' created at ${price:.2f} {price_unit}",
            data={
                "service_id": service_id,
                "name": name,
                "price": price,
                "price_unit": price_unit,
            },
            revenue=0,
        )

    async def _list_services(self, params: Dict) -> SkillResult:
        """List all services."""
        services = []
        for svc in self._services.values():
            services.append({
                "id": svc.id,
                "name": svc.name,
                "description": svc.description,
                "price": svc.price,
                "price_unit": svc.price_unit,
                "active": svc.active,
                "times_sold": svc.times_sold,
                "total_earned": svc.total_earned,
            })

        active_count = sum(1 for s in services if s["active"])
        return SkillResult(
            success=True,
            message=f"{len(services)} service(s) defined, {active_count} active",
            data={"services": services, "total": len(services), "active": active_count},
        )

    async def _update_service(self, params: Dict) -> SkillResult:
        """Update a service."""
        service_id = params.get("service_id", "").strip()
        if not service_id:
            return SkillResult(success=False, message="service_id is required")

        service = self._services.get(service_id)
        if not service:
            return SkillResult(success=False, message=f"Service not found: {service_id}")

        updated = []
        if "price" in params and params["price"] is not None:
            price = params["price"]
            if price <= 0:
                return SkillResult(success=False, message="Price must be positive")
            service.price = price
            updated.append(f"price=${price:.2f}")

        if "active" in params and params["active"] is not None:
            service.active = bool(params["active"])
            updated.append(f"active={service.active}")

        if "description" in params and params["description"]:
            service.description = params["description"].strip()
            updated.append("description")

        if not updated:
            return SkillResult(success=False, message="No updates provided")

        return SkillResult(
            success=True,
            message=f"Service '{service.name}' updated: {', '.join(updated)}",
            data={"service_id": service_id, "updates": updated},
        )

    # === Invoicing ===

    async def _create_invoice(self, params: Dict) -> SkillResult:
        """Create an invoice for work done."""
        service_id = params.get("service_id", "").strip()
        client = params.get("client", "").strip()
        description = params.get("description", "").strip()

        if not service_id:
            return SkillResult(success=False, message="service_id is required")
        if not client:
            return SkillResult(success=False, message="client is required")

        service = self._services.get(service_id)
        if not service:
            return SkillResult(success=False, message=f"Service not found: {service_id}")

        if not service.active:
            return SkillResult(success=False, message=f"Service '{service.name}' is not active")

        amount = params.get("amount", service.price)
        if amount <= 0:
            return SkillResult(success=False, message="Amount must be positive")

        invoice_id = f"inv_{uuid.uuid4().hex[:8]}"
        invoice = Invoice(
            id=invoice_id,
            service_id=service_id,
            service_name=service.name,
            client=client,
            amount=amount,
            description=description or f"{service.name} for {client}",
            status=InvoiceStatus.DRAFT,
            created_at=datetime.now().isoformat(),
        )
        self._invoices[invoice_id] = invoice

        return SkillResult(
            success=True,
            message=f"Invoice created: ${amount:.2f} for '{service.name}' to {client}",
            data={
                "invoice_id": invoice_id,
                "service": service.name,
                "client": client,
                "amount": amount,
                "status": "draft",
            },
        )

    async def _list_invoices(self, params: Dict) -> SkillResult:
        """List invoices with optional status filter."""
        status_filter = params.get("status", "").strip().lower()

        invoices = []
        for inv in self._invoices.values():
            if status_filter and inv.status.value != status_filter:
                continue
            invoices.append({
                "id": inv.id,
                "service": inv.service_name,
                "client": inv.client,
                "amount": inv.amount,
                "status": inv.status.value,
                "description": inv.description,
                "created_at": inv.created_at,
                "paid_at": inv.paid_at,
            })

        total_amount = sum(i["amount"] for i in invoices)
        return SkillResult(
            success=True,
            message=f"{len(invoices)} invoice(s), total: ${total_amount:.2f}",
            data={"invoices": invoices, "count": len(invoices), "total_amount": total_amount},
        )

    async def _record_payment(self, params: Dict) -> SkillResult:
        """Record a payment for an invoice."""
        invoice_id = params.get("invoice_id", "").strip()
        if not invoice_id:
            return SkillResult(success=False, message="invoice_id is required")

        invoice = self._invoices.get(invoice_id)
        if not invoice:
            return SkillResult(success=False, message=f"Invoice not found: {invoice_id}")

        if invoice.status == InvoiceStatus.PAID:
            return SkillResult(success=False, message="Invoice already paid")

        if invoice.status == InvoiceStatus.CANCELLED:
            return SkillResult(success=False, message="Invoice was cancelled")

        amount = params.get("amount", invoice.amount)
        if amount <= 0:
            return SkillResult(success=False, message="Payment amount must be positive")

        # Mark invoice as paid
        invoice.status = InvoiceStatus.PAID
        invoice.paid_at = datetime.now().isoformat()

        # Update service stats
        service = self._services.get(invoice.service_id)
        if service:
            service.times_sold += 1
            service.total_earned += amount

        # Record the payment
        payment = {
            "invoice_id": invoice_id,
            "amount": amount,
            "client": invoice.client,
            "service": invoice.service_name,
            "paid_at": invoice.paid_at,
        }
        self._payments.append(payment)

        return SkillResult(
            success=True,
            message=f"Payment of ${amount:.2f} recorded from {invoice.client}",
            data=payment,
            revenue=amount,
        )

    # === Expense Tracking ===

    async def _record_expense(self, params: Dict) -> SkillResult:
        """Record an expense."""
        amount = params.get("amount", 0)
        category = params.get("category", "other").strip().lower()
        description = params.get("description", "").strip()

        if amount <= 0:
            return SkillResult(success=False, message="Amount must be positive")
        if not description:
            return SkillResult(success=False, message="Description is required")

        valid_categories = [e.value for e in ExpenseCategory]
        if category not in valid_categories:
            category = "other"

        expense_id = f"exp_{uuid.uuid4().hex[:8]}"
        expense = Expense(
            id=expense_id,
            amount=amount,
            category=category,
            description=description,
            created_at=datetime.now().isoformat(),
        )
        self._expenses.append(expense)

        return SkillResult(
            success=True,
            message=f"Expense recorded: ${amount:.2f} ({category}) - {description}",
            data={
                "expense_id": expense_id,
                "amount": amount,
                "category": category,
                "description": description,
            },
            cost=amount,
        )

    # === Financial Reports ===

    async def _financial_report(self, params: Dict) -> SkillResult:
        """Generate a financial summary."""
        total_revenue = sum(p["amount"] for p in self._payments)
        total_expenses = sum(e.amount for e in self._expenses)
        profit = total_revenue - total_expenses

        # Breakdown by expense category
        expense_by_category = {}
        for e in self._expenses:
            expense_by_category[e.category] = expense_by_category.get(e.category, 0) + e.amount

        # Invoice stats
        invoice_stats = {
            "total": len(self._invoices),
            "draft": sum(1 for i in self._invoices.values() if i.status == InvoiceStatus.DRAFT),
            "sent": sum(1 for i in self._invoices.values() if i.status == InvoiceStatus.SENT),
            "paid": sum(1 for i in self._invoices.values() if i.status == InvoiceStatus.PAID),
            "cancelled": sum(1 for i in self._invoices.values() if i.status == InvoiceStatus.CANCELLED),
        }

        # Outstanding amount (draft + sent)
        outstanding = sum(
            i.amount for i in self._invoices.values()
            if i.status in (InvoiceStatus.DRAFT, InvoiceStatus.SENT)
        )

        status = "profitable" if profit > 0 else "break-even" if profit == 0 else "losing money"

        return SkillResult(
            success=True,
            message=f"Financial Report: ${total_revenue:.2f} revenue, ${total_expenses:.2f} expenses, ${profit:.2f} profit ({status})",
            data={
                "total_revenue": total_revenue,
                "total_expenses": total_expenses,
                "profit": profit,
                "status": status,
                "expense_breakdown": expense_by_category,
                "invoice_stats": invoice_stats,
                "outstanding_amount": outstanding,
                "total_payments": len(self._payments),
                "total_services": len(self._services),
                "active_services": sum(1 for s in self._services.values() if s.active),
            },
        )

    async def _top_services(self, params: Dict) -> SkillResult:
        """Show which services generate the most revenue."""
        services = sorted(
            self._services.values(),
            key=lambda s: s.total_earned,
            reverse=True,
        )

        rankings = []
        for i, svc in enumerate(services):
            rankings.append({
                "rank": i + 1,
                "id": svc.id,
                "name": svc.name,
                "total_earned": svc.total_earned,
                "times_sold": svc.times_sold,
                "price": svc.price,
                "active": svc.active,
                "avg_revenue": svc.total_earned / svc.times_sold if svc.times_sold > 0 else 0,
            })

        top_name = rankings[0]["name"] if rankings else "none"
        return SkillResult(
            success=True,
            message=f"Top service: {top_name}" if rankings else "No services defined yet",
            data={"rankings": rankings, "total_services": len(rankings)},
        )
