"""Integration tests for streamDeliveryResources on MemoryClient.

Requires environment variables:
    MEMORY_KINESIS_ARN: ARN of a pre-existing Kinesis stream
    MEMORY_ROLE_ARN: ARN of an IAM role the memory service can assume

Run with:
    pytest tests_integ/memory/test_stream_delivery.py -xvs --log-cli-level=INFO
"""

import os
import time

import pytest

from bedrock_agentcore.memory import MemoryClient


@pytest.mark.integration
class TestStreamDeliveryResources:
    """Integration tests for streamDeliveryResources."""

    @classmethod
    def setup_class(cls):
        cls.kinesis_stream_arn = os.environ.get("MEMORY_KINESIS_ARN")
        cls.execution_role_arn = os.environ.get("MEMORY_ROLE_ARN")

        if not cls.kinesis_stream_arn or not cls.execution_role_arn:
            pytest.fail("MEMORY_KINESIS_ARN and MEMORY_ROLE_ARN must be set")

        cls.region = os.environ.get("BEDROCK_TEST_REGION", "us-west-2")
        cls.client = MemoryClient(region_name=cls.region)
        cls.test_prefix = f"test_stream_{int(time.time())}"
        cls.memory_ids = []

    @classmethod
    def teardown_class(cls):
        for memory_id in cls.memory_ids:
            try:
                cls.client.delete_memory(memory_id)
            except Exception as e:
                print(f"Failed to delete memory {memory_id}: {e}")

    def _make_delivery_config(self, level="FULL_CONTENT"):
        return {
            "resources": [
                {
                    "kinesis": {
                        "dataStreamArn": self.kinesis_stream_arn,
                        "contentConfigurations": [{"type": "MEMORY_RECORDS", "level": level}],
                    }
                }
            ]
        }

    def _get_delivery_level(self, memory_id):
        """Get the current stream delivery level for a memory."""
        detail = self.client.gmcp_client.get_memory(memoryId=memory_id)["memory"]
        for resource in detail.get("streamDeliveryResources", {}).get("resources", []):
            configs = resource.get("kinesis", {}).get("contentConfigurations", [])
            if configs:
                return configs[0].get("level")
        return None

    def _wait_for_active(self, memory_id, max_wait=180, poll_interval=10):
        """Poll until memory returns to ACTIVE status."""
        deadline = time.time() + max_wait
        while time.time() < deadline:
            status = self.client.get_memory_status(memory_id)
            if status == "ACTIVE":
                return
            if status == "FAILED":
                pytest.fail(f"Memory {memory_id} entered FAILED status")
            time.sleep(poll_interval)
        pytest.fail(f"Memory {memory_id} did not reach ACTIVE within {max_wait}s")

    @pytest.mark.order(1)
    def test_stream_delivery_create(self):
        """Create memory with stream delivery config and verify via get_memory."""
        delivery_config = self._make_delivery_config("FULL_CONTENT")

        memory = self.client.create_memory_and_wait(
            name=f"{self.test_prefix}_stream",
            strategies=[],
            memory_execution_role_arn=self.execution_role_arn,
            stream_delivery_resources=delivery_config,
        )

        memory_id = memory.get("memoryId", memory.get("id"))
        self.__class__.stream_memory_id = memory_id
        self.memory_ids.append(memory_id)

        assert memory["streamDeliveryResources"] == delivery_config

    @pytest.mark.order(2)
    def test_stream_delivery_update(self):
        """Update delivery config from FULL_CONTENT to METADATA_ONLY, verify change."""
        memory_id = self.stream_memory_id

        assert self._get_delivery_level(memory_id) == "FULL_CONTENT"

        self.client.update_stream_delivery_config(
            memory_id=memory_id,
            stream_delivery_resources=self._make_delivery_config("METADATA_ONLY"),
        )
        self._wait_for_active(memory_id)

        assert self._get_delivery_level(memory_id) == "METADATA_ONLY"
