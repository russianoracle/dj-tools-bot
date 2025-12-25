#!/usr/bin/env python3
"""
Create Yandex Cloud Monitoring dashboards from JSON configs.

Usage:
    python3 scripts/create-dashboards.py
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yandexcloud
from yandex.cloud.monitoring.v3 import dashboard_service_pb2, dashboard_service_pb2_grpc
from yandex.cloud.monitoring.v3 import dashboard_pb2


def load_dashboard_config(config_path: str) -> dict:
    """Load dashboard configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def convert_widget_to_proto(widget_config: dict) -> dashboard_pb2.Widget:
    """Convert JSON widget config to protobuf Widget."""
    widget = dashboard_pb2.Widget()

    # Position
    if 'position' in widget_config:
        pos = widget_config['position']
        widget.position.x = pos.get('x', 0)
        widget.position.y = pos.get('y', 0)
        widget.position.w = pos.get('w', 12)
        widget.position.h = pos.get('h', 8)

    # Text (title)
    if 'title' in widget_config:
        widget.text.text = widget_config['title']

    # Chart
    if 'chart' in widget_config:
        chart = widget_config['chart']

        # Chart ID
        if 'id' in chart:
            widget.chart.id = chart['id']

        # Queries
        if 'queries' in chart and 'targets' in chart['queries']:
            for target in chart['queries']['targets']:
                query = widget.chart.queries.targets.add()
                query.query = target.get('query', '')
                if 'text' in target:
                    query.text_mode = target['text']
                query.hidden = target.get('hidden', False)

        # Title
        if 'title' in widget_config:
            widget.chart.name_hiding_settings.positive = False

        # Description
        if 'description' in widget_config:
            widget.chart.description = widget_config['description']

    return widget


def create_dashboard(folder_id: str, config_path: str) -> str:
    """Create dashboard in Yandex Cloud Monitoring."""

    # Load config
    config = load_dashboard_config(config_path)

    # Get IAM token
    sdk = yandexcloud.SDK()

    # Create dashboard request
    dashboard_service = sdk.client(dashboard_service_pb2_grpc.DashboardServiceStub)

    # Build dashboard
    dashboard = dashboard_pb2.Dashboard()
    dashboard.name = config['name']
    dashboard.description = config.get('description', '')

    # Add widgets
    for widget_config in config.get('widgets', []):
        widget = convert_widget_to_proto(widget_config)
        dashboard.widgets.append(widget)

    # Create request
    request = dashboard_service_pb2.CreateDashboardRequest()
    request.folder_id = folder_id
    request.dashboard.CopyFrom(dashboard)

    # Execute
    operation = dashboard_service.Create(request)

    # Wait for completion
    operation_result = sdk.wait_operation_and_get_result(
        operation,
        response_type=dashboard_pb2.Dashboard,
        meta_type=dashboard_service_pb2.CreateDashboardMetadata
    )

    return operation_result.response.id


def main():
    """Main entry point."""

    # Get folder ID from environment
    folder_id = os.getenv('FOLDER_ID')
    if not folder_id:
        print("Error: FOLDER_ID environment variable not set")
        print("Usage: FOLDER_ID=b1ge0vpe8dp87vc3n73l python3 scripts/create-dashboards.py")
        sys.exit(1)

    # Project root
    root = Path(__file__).parent.parent

    # Dashboard configs
    dashboards = [
        ('Business Metrics', root / 'monitoring' / 'dashboard-business.json'),
        ('System Metrics', root / 'monitoring' / 'dashboard-system.json'),
    ]

    # Create dashboards
    created_ids = []
    for name, config_path in dashboards:
        try:
            print(f"Creating dashboard: {name}...")
            dashboard_id = create_dashboard(folder_id, str(config_path))
            created_ids.append((name, dashboard_id))
            print(f"✓ Created: {name} (ID: {dashboard_id})")
        except Exception as e:
            print(f"✗ Failed to create {name}: {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    if created_ids:
        print("\n" + "="*80)
        print("Dashboards created successfully!")
        print("="*80)
        for name, dashboard_id in created_ids:
            url = f"https://console.yandex.cloud/folders/{folder_id}/monitoring/dashboards/{dashboard_id}"
            print(f"\n{name}:")
            print(f"  URL: {url}")
        print("\n")


if __name__ == '__main__':
    main()
