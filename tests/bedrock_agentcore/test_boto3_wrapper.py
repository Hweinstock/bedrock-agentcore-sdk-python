"""Tests for boto3_wrapper utilities."""

from unittest.mock import MagicMock

import pytest

from bedrock_agentcore._utils.boto3_wrapper import snake_to_camel, wrap_boto3_method


class TestSnakeToCamel:
    """Tests for snake_to_camel conversion."""

    def test_single_word(self):
        assert snake_to_camel("name") == "name"

    def test_two_words(self):
        assert snake_to_camel("memory_id") == "memoryId"

    def test_three_words(self):
        assert snake_to_camel("actor_session_id") == "actorSessionId"

    def test_already_camel_case_passthrough(self):
        assert snake_to_camel("memoryId") == "memoryId"

    def test_already_camel_case_long(self):
        assert snake_to_camel("streamDeliveryResources") == "streamDeliveryResources"

    def test_multi_segment_snake(self):
        assert snake_to_camel("memory_execution_role_arn") == "memoryExecutionRoleArn"

    def test_empty_string(self):
        assert snake_to_camel("") == ""

    def test_leading_underscore(self):
        """Leading underscore produces empty first segment — title() on rest."""
        result = snake_to_camel("_private")
        assert result == "Private"

    def test_consecutive_underscores(self):
        """Consecutive underscores produce empty segments that title() to empty strings."""
        result = snake_to_camel("a__b")
        assert result == "aB"

    def test_trailing_underscore(self):
        result = snake_to_camel("name_")
        assert result == "name"

    def test_single_char_segments(self):
        assert snake_to_camel("a_b_c") == "aBC"

    def test_uppercase_in_snake(self):
        """If someone passes mixed like 'memory_ID', title() lowercases the rest."""
        assert snake_to_camel("memory_ID") == "memoryId"

    def test_numeric_segment(self):
        assert snake_to_camel("field_2_name") == "field2Name"


class TestWrapBoto3Method:
    """Tests for wrap_boto3_method wrapper."""

    def setup_method(self):
        self.mock_method = MagicMock(return_value={"result": "ok"})

    def test_snake_case_converted(self):
        wrapped = wrap_boto3_method(self.mock_method)
        wrapped(memory_id="mem-1", actor_id="user-1")
        self.mock_method.assert_called_once_with(memoryId="mem-1", actorId="user-1")

    def test_camel_case_passthrough(self):
        wrapped = wrap_boto3_method(self.mock_method)
        wrapped(memoryId="mem-1", actorId="user-1")
        self.mock_method.assert_called_once_with(memoryId="mem-1", actorId="user-1")

    def test_mixed_snake_and_camel_different_params(self):
        """Snake and camel for *different* params is fine."""
        wrapped = wrap_boto3_method(self.mock_method)
        wrapped(memory_id="mem-1", actorId="user-1")
        self.mock_method.assert_called_once_with(memoryId="mem-1", actorId="user-1")

    def test_collision_raises_type_error(self):
        wrapped = wrap_boto3_method(self.mock_method)
        with pytest.raises(TypeError, match="memoryId.*memory_id"):
            wrapped(memoryId="mem-1", memory_id="mem-2")

    def test_collision_same_value_still_raises(self):
        """Even if values match, ambiguity should be rejected."""
        wrapped = wrap_boto3_method(self.mock_method)
        with pytest.raises(TypeError):
            wrapped(memoryId="mem-1", memory_id="mem-1")

    def test_return_value_forwarded(self):
        wrapped = wrap_boto3_method(self.mock_method)
        result = wrapped(memory_id="mem-1")
        assert result == {"result": "ok"}

    def test_positional_args_forwarded(self):
        wrapped = wrap_boto3_method(self.mock_method)
        wrapped("pos1", "pos2", memory_id="mem-1")
        self.mock_method.assert_called_once_with("pos1", "pos2", memoryId="mem-1")

    def test_no_kwargs(self):
        wrapped = wrap_boto3_method(self.mock_method)
        wrapped()
        self.mock_method.assert_called_once_with()

    def test_exception_propagated(self):
        self.mock_method.side_effect = ValueError("boom")
        wrapped = wrap_boto3_method(self.mock_method)
        with pytest.raises(ValueError, match="boom"):
            wrapped(memory_id="mem-1")

    def test_preserves_function_name(self):
        def my_boto3_method():
            pass

        wrapped = wrap_boto3_method(my_boto3_method)
        assert wrapped.__name__ == "my_boto3_method"

    def test_multi_word_snake_conversion(self):
        wrapped = wrap_boto3_method(self.mock_method)
        wrapped(stream_delivery_resources={"arn": "..."}, client_token="tok-1")
        self.mock_method.assert_called_once_with(
            streamDeliveryResources={"arn": "..."}, clientToken="tok-1"
        )

    def test_collision_order_independent(self):
        """Collision detected regardless of which form comes first."""
        wrapped = wrap_boto3_method(self.mock_method)
        with pytest.raises(TypeError):
            wrapped(actor_id="u1", actorId="u2")

    def test_single_word_key_unchanged(self):
        wrapped = wrap_boto3_method(self.mock_method)
        wrapped(name="test", namespace="ns/")
        self.mock_method.assert_called_once_with(name="test", namespace="ns/")


class TestWrapBoto3MethodIntegrationWithGetattr:
    """Verify snake_case kwargs reach boto3 through the actual __getattr__ path."""

    def test_memory_client_passthrough_accepts_snake_case(self):
        from unittest.mock import MagicMock, patch

        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_gmdp = MagicMock()
            mock_gmcp = MagicMock()

            def pick_client(service, **kw):
                return mock_gmdp if service == "bedrock-agentcore" else mock_gmcp

            mock_session.client.side_effect = pick_client
            mock_session_class.return_value = mock_session

            from bedrock_agentcore.memory.client import MemoryClient

            client = MemoryClient(region_name="us-west-2")

            # GMDP passthrough with snake_case
            mock_gmdp.get_event.return_value = {"event": {}}
            client.get_event(memory_id="mem-1", actor_id="a", session_id="s", event_id="e")
            mock_gmdp.get_event.assert_called_once_with(
                memoryId="mem-1", actorId="a", sessionId="s", eventId="e"
            )

            # GMCP passthrough with snake_case
            mock_gmcp.update_memory.return_value = {"memory": {}}
            client.update_memory(memory_id="mem-1", client_token="tok")
            mock_gmcp.update_memory.assert_called_once_with(memoryId="mem-1", clientToken="tok")

    def test_session_manager_passthrough_accepts_snake_case(self):
        from unittest.mock import MagicMock, patch

        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client = MagicMock()
            mock_session.client.return_value = mock_client
            mock_session_class.return_value = mock_session

            from bedrock_agentcore.memory.session import MemorySessionManager

            mgr = MemorySessionManager(memory_id="mem-1", region_name="us-west-2")

            mock_client.retrieve_memory_records.return_value = {"memoryRecordSummaries": []}
            mgr.retrieve_memory_records(memory_id="mem-1", namespace="ns/", search_criteria={"searchQuery": "q"})
            mock_client.retrieve_memory_records.assert_called_once_with(
                memoryId="mem-1", namespace="ns/", searchCriteria={"searchQuery": "q"}
            )

    def test_passthrough_still_accepts_camel_case(self):
        from unittest.mock import MagicMock, patch

        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_gmdp = MagicMock()
            mock_gmcp = MagicMock()

            def pick_client(service, **kw):
                return mock_gmdp if service == "bedrock-agentcore" else mock_gmcp

            mock_session.client.side_effect = pick_client
            mock_session_class.return_value = mock_session

            from bedrock_agentcore.memory.client import MemoryClient

            client = MemoryClient(region_name="us-west-2")

            mock_gmcp.update_memory.return_value = {"memory": {}}
            client.update_memory(memoryId="mem-1", clientToken="tok")
            mock_gmcp.update_memory.assert_called_once_with(memoryId="mem-1", clientToken="tok")


class TestListEventsBackwardCompat:
    """Tests for list_events eventMetadata → event_metadata backward compat."""

    def _make_manager(self):
        from unittest.mock import MagicMock, patch

        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.region_name = "us-west-2"
            mock_client = MagicMock()
            mock_session.client.return_value = mock_client
            mock_session_class.return_value = mock_session

            from bedrock_agentcore.memory.session import MemorySessionManager

            mgr = MemorySessionManager(memory_id="mem-1", region_name="us-west-2")
            mock_client.list_events.return_value = {"events": [], "nextToken": None}
            return mgr, mock_client

    def test_snake_case_event_metadata(self):
        mgr, mock_client = self._make_manager()
        filters = [{"left": {"metadataKey": "k"}, "operator": "EQUALS_TO", "right": {"metadataValue": {"stringValue": "v"}}}]
        mgr.list_events(actor_id="a", session_id="s", event_metadata=filters)
        call_kwargs = mock_client.list_events.call_args[1]
        assert call_kwargs["filter"]["eventMetadata"] == filters

    def test_legacy_camel_case_event_metadata(self):
        mgr, mock_client = self._make_manager()
        filters = [{"left": {"metadataKey": "k"}, "operator": "EQUALS_TO", "right": {"metadataValue": {"stringValue": "v"}}}]
        mgr.list_events(actor_id="a", session_id="s", eventMetadata=filters)
        call_kwargs = mock_client.list_events.call_args[1]
        assert call_kwargs["filter"]["eventMetadata"] == filters

    def test_both_raises_type_error(self):
        mgr, _ = self._make_manager()
        filters = [{"left": {"metadataKey": "k"}, "operator": "EQUALS_TO", "right": {"metadataValue": {"stringValue": "v"}}}]
        with pytest.raises(TypeError, match="event_metadata.*eventMetadata"):
            mgr.list_events(actor_id="a", session_id="s", event_metadata=filters, eventMetadata=filters)

    def test_unknown_kwarg_raises_type_error(self):
        mgr, _ = self._make_manager()
        with pytest.raises(TypeError, match="unexpected keyword arguments"):
            mgr.list_events(actor_id="a", session_id="s", bogus_param=True)
