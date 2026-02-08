#!/usr/bin/env python3
"""Tests for ServiceCatalogSkill."""

import pytest
from singularity.skills.service_catalog import (
    ServiceCatalogSkill,
    SERVICE_OFFERINGS,
    SERVICE_BUNDLES,
    PRICING_TIERS,
)


@pytest.fixture
def skill():
    s = ServiceCatalogSkill()
    s._deployed = {}
    s._deployed_bundles = {}
    s._revenue_log = []
    return s


def test_list_offerings(skill):
    result = skill.execute("list_offerings")
    assert result.success
    assert len(result.data["offerings"]) == len(SERVICE_OFFERINGS)
    assert "categories" in result.data


def test_list_bundles(skill):
    result = skill.execute("list_bundles")
    assert result.success
    bundles = result.data["bundles"]
    assert len(bundles) == len(SERVICE_BUNDLES)
    for b in bundles:
        assert b["bundle_price"] < b["total_base_price"]  # discount applied


def test_preview_offering(skill):
    result = skill.execute("preview", {"offering_id": "code_review"})
    assert result.success
    assert "pricing_by_tier" in result.data
    for tier in PRICING_TIERS:
        assert tier in result.data["pricing_by_tier"]


def test_preview_bundle(skill):
    result = skill.execute("preview", {"offering_id": "developer_essentials"})
    assert result.success
    assert result.data["bundle_discount_pct"] == 15
    assert len(result.data["services"]) == 3


def test_preview_unknown(skill):
    result = skill.execute("preview", {"offering_id": "nonexistent"})
    assert not result.success


def test_deploy_service(skill):
    result = skill.execute("deploy", {"offering_id": "code_review", "tier": "pro"})
    assert result.success
    dep = result.data["deployment"]
    assert dep["tier"] == "pro"
    assert dep["price"] == round(0.10 * 2.5, 4)
    assert "code_review" in skill._deployed


def test_deploy_unknown_offering(skill):
    result = skill.execute("deploy", {"offering_id": "fake"})
    assert not result.success


def test_deploy_unknown_tier(skill):
    result = skill.execute("deploy", {"offering_id": "code_review", "tier": "platinum"})
    assert not result.success


def test_deploy_bundle(skill):
    result = skill.execute("deploy_bundle", {"bundle_id": "developer_essentials", "tier": "basic"})
    assert result.success
    assert result.data["bundle"]["services_deployed"] == 3
    assert "code_review" in skill._deployed
    assert "api_docs" in skill._deployed
    assert "developer_essentials" in skill._deployed_bundles


def test_deploy_bundle_discount(skill):
    result = skill.execute("deploy_bundle", {"bundle_id": "full_stack", "tier": "basic"})
    assert result.success
    # Each service should have 25% discount applied
    for dep in result.data["services"]:
        offering = SERVICE_OFFERINGS[dep["offering_id"]]
        expected = round(offering["base_price"] * 1.0 * 0.75, 4)
        assert dep["price"] == expected


def test_undeploy_service(skill):
    skill.execute("deploy", {"offering_id": "seo_audit"})
    assert "seo_audit" in skill._deployed
    result = skill.execute("undeploy", {"offering_id": "seo_audit"})
    assert result.success
    assert "seo_audit" not in skill._deployed


def test_undeploy_bundle(skill):
    skill.execute("deploy_bundle", {"bundle_id": "content_creator"})
    assert "content_creator" in skill._deployed_bundles
    result = skill.execute("undeploy", {"offering_id": "content_creator"})
    assert result.success
    assert "content_creator" not in skill._deployed_bundles


def test_undeploy_nonexistent(skill):
    result = skill.execute("undeploy", {"offering_id": "nope"})
    assert not result.success


def test_status(skill):
    skill.execute("deploy", {"offering_id": "code_review"})
    skill.execute("deploy", {"offering_id": "data_analysis"})
    result = skill.execute("status")
    assert result.success
    assert result.data["summary"]["total_services"] == 2


def test_project_revenue(skill):
    result = skill.execute("project_revenue", {"daily_requests": 100, "tier": "basic"})
    assert result.success
    summary = result.data["summary"]
    assert summary["total_monthly_revenue"] > 0
    assert summary["total_monthly_profit"] > 0
    assert summary["annual_revenue_estimate"] == round(summary["total_monthly_revenue"] * 12, 2)


def test_project_revenue_free_tier(skill):
    result = skill.execute("project_revenue", {"daily_requests": 50, "tier": "free"})
    assert result.success
    assert result.data["summary"]["total_monthly_revenue"] == 0


def test_unknown_action(skill):
    result = skill.execute("nonexistent_action")
    assert not result.success


def test_manifest(skill):
    m = skill.manifest
    assert m.skill_id == "service_catalog"
    assert m.category == "revenue"
    assert len(m.actions) == 8
