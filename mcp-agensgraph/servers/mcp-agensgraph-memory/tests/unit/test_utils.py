import argparse
import os
from unittest.mock import patch
import pytest

from mcp_agensgraph_memory.utils import process_config, build_connection_url


class TestBuildConnectionUrl:
    """Test the build_connection_url function."""

    def test_build_connection_url_basic(self):
        """Test building connection URL with basic inputs."""
        base_url = "postgresql://localhost:5432"
        result = build_connection_url(base_url, "user", "pass", "dbname")
        assert result == "postgresql://user:pass@localhost:5432/dbname"

    def test_build_connection_url_custom_port(self):
        """Test building connection URL with custom port."""
        base_url = "postgresql://myhost:9999"
        result = build_connection_url(base_url, "testuser", "testpass", "testdb")
        assert result == "postgresql://testuser:testpass@myhost:9999/testdb"

    def test_build_connection_url_invalid_scheme(self):
        """Test that invalid scheme raises ValueError."""
        with pytest.raises(ValueError, match="Invalid database URL scheme"):
            build_connection_url("mysql://localhost:3306", "user", "pass", "db")

    def test_build_connection_url_default_port(self):
        """Test that default port 5432 is used when not specified."""
        base_url = "postgresql://localhost"
        result = build_connection_url(base_url, "user", "pass", "db")
        assert result == "postgresql://user:pass@localhost:5432/db"


@pytest.fixture
def clean_env():
    """Fixture to clean environment variables before each test."""
    env_vars = [
        "AGENSGRAPH_URL", "AGENSGRAPH_USERNAME", "AGENSGRAPH_PASSWORD",
        "AGENSGRAPH_DB", "AGENSGRAPH_GRAPH_NAME", "AGENSGRAPH_TRANSPORT",
        "AGENSGRAPH_MCP_SERVER_HOST", "AGENSGRAPH_MCP_SERVER_PORT",
        "AGENSGRAPH_MCP_SERVER_PATH", "AGENSGRAPH_MCP_SERVER_ALLOW_ORIGINS",
        "AGENSGRAPH_MCP_SERVER_ALLOWED_HOSTS", "AGENSGRAPH_NAMESPACE"
    ]
    # Store original values
    original_values = {}
    for var in env_vars:
        if var in os.environ:
            original_values[var] = os.environ[var]
            del os.environ[var]

    yield

    # Restore original values
    for var, value in original_values.items():
        os.environ[var] = value


@pytest.fixture
def args_factory():
    """Factory fixture to create argparse.Namespace objects with default None values."""
    def _create_args(**kwargs):
        defaults = {
            "db_url": None,
            "username": None,
            "password": None,
            "database": None,
            "graphname": None,
            "transport": None,
            "server_host": None,
            "server_port": None,
            "server_path": None,
            "allow_origins": None,
            "allowed_hosts": None,
            "namespace": None,
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)
    return _create_args


@pytest.fixture
def mock_logger():
    """Fixture to provide a mocked logger."""
    with patch('mcp_agensgraph_memory.utils.logger') as mock:
        yield mock


@pytest.fixture
def sample_cli_args(args_factory):
    """Fixture providing sample CLI arguments."""
    return args_factory(
        db_url="postgresql://test:5432",
        username="testuser",
        password="testpass",
        database="testdb",
        graphname="testgraph",
        transport="http",
        server_host="localhost",
        server_port=9000,
        server_path="/test/",
        allow_origins="http://localhost:3000,https://trusted-site.com",
        allowed_hosts="localhost,127.0.0.1,example.com"
    )


@pytest.fixture
def sample_env_vars():
    """Fixture providing sample environment variables."""
    return {
        "AGENSGRAPH_URL": "postgresql://env:5432",
        "AGENSGRAPH_USERNAME": "envuser",
        "AGENSGRAPH_PASSWORD": "envpass",
        "AGENSGRAPH_DB": "envdb",
        "AGENSGRAPH_GRAPH_NAME": "envgraph",
        "AGENSGRAPH_TRANSPORT": "sse",
        "AGENSGRAPH_MCP_SERVER_HOST": "envhost",
        "AGENSGRAPH_MCP_SERVER_PORT": "8080",
        "AGENSGRAPH_MCP_SERVER_PATH": "/env/",
        "AGENSGRAPH_MCP_SERVER_ALLOW_ORIGINS": "http://env-site.com,https://env-secure.com",
        "AGENSGRAPH_MCP_SERVER_ALLOWED_HOSTS": "envhost.com,api.envhost.com"
    }


@pytest.fixture
def set_env_vars(sample_env_vars):
    """Fixture to set environment variables and clean up after test."""
    for key, value in sample_env_vars.items():
        os.environ[key] = value
    yield sample_env_vars
    # Cleanup handled by clean_env fixture


@pytest.fixture
def expected_defaults():
    """Fixture providing expected default configuration values."""
    return {
        "agensgraph_url": "postgresql://localhost:5432",
        "agensgraph_user": "agens",
        "agensgraph_password": "agens",
        "agensgraph_database": "agens",
        "agensgraph_graphname": "memory",
        "transport": "stdio",
        "host": None,
        "port": None,
        "path": None,
        "allow_origins": [],
        "allowed_hosts": ["localhost", "127.0.0.1"],
    }


def test_all_cli_args_provided(clean_env, sample_cli_args):
    """Test when all CLI arguments are provided."""
    config = process_config(sample_cli_args)

    assert config["agensgraph_url"] == "postgresql://test:5432"
    assert config["agensgraph_user"] == "testuser"
    assert config["agensgraph_password"] == "testpass"
    assert config["agensgraph_database"] == "testdb"
    assert config["agensgraph_graphname"] == "testgraph"
    assert config["transport"] == "http"
    assert config["host"] == "localhost"
    assert config["port"] == 9000
    assert config["path"] == "/test/"
    assert config["allow_origins"] == ["http://localhost:3000", "https://trusted-site.com"]
    assert config["allowed_hosts"] == ["localhost", "127.0.0.1", "example.com"]


def test_all_env_vars_provided(clean_env, set_env_vars, args_factory):
    """Test when all environment variables are provided."""
    args = args_factory()
    config = process_config(args)

    assert config["agensgraph_url"] == "postgresql://env:5432"
    assert config["agensgraph_user"] == "envuser"
    assert config["agensgraph_password"] == "envpass"
    assert config["agensgraph_database"] == "envdb"
    assert config["agensgraph_graphname"] == "envgraph"
    assert config["transport"] == "sse"
    assert config["host"] == "envhost"
    assert config["port"] == 8080
    assert config["path"] == "/env/"
    assert config["allow_origins"] == ["http://env-site.com", "https://env-secure.com"]
    assert config["allowed_hosts"] == ["envhost.com", "api.envhost.com"]


def test_cli_args_override_env_vars(clean_env, args_factory):
    """Test that CLI arguments take precedence over environment variables."""
    os.environ["AGENSGRAPH_URL"] = "postgresql://env:5432"
    os.environ["AGENSGRAPH_USERNAME"] = "envuser"

    args = args_factory(
        db_url="postgresql://cli:5432",
        username="cliuser"
    )

    config = process_config(args)

    assert config["agensgraph_url"] == "postgresql://cli:5432"
    assert config["agensgraph_user"] == "cliuser"


def test_default_values_with_warnings(clean_env, args_factory, expected_defaults, mock_logger):
    """Test default values are used and warnings are logged when nothing is provided."""
    args = args_factory()
    config = process_config(args)

    for key, expected_value in expected_defaults.items():
        assert config[key] == expected_value

    # Check that warnings were logged
    warning_calls = [call for call in mock_logger.warning.call_args_list]
    assert len(warning_calls) == 6  # 6 warnings: url, user, password, database, graphname, transport


def test_stdio_transport_ignores_server_config(clean_env, args_factory, mock_logger):
    """Test that stdio transport ignores server host/port/path and logs warnings."""
    args = args_factory(
        transport="stdio",
        server_host="localhost",
        server_port=8000,
        server_path="/test/"
    )

    config = process_config(args)

    assert config["transport"] == "stdio"
    assert config["host"] == "localhost"  # Set but ignored
    assert config["port"] == 8000  # Set but ignored
    assert config["path"] == "/test/"  # Set but ignored

    # Check that warnings were logged for ignored server config
    warning_calls = [call.args[0] for call in mock_logger.warning.call_args_list]
    stdio_warnings = [msg for msg in warning_calls if "stdio" in msg and "ignored" in msg]
    assert len(stdio_warnings) == 3  # host, port, path warnings


def test_non_stdio_transport_uses_defaults(clean_env, args_factory, mock_logger):
    """Test that non-stdio transport uses default server config when not provided."""
    args = args_factory(transport="http")
    config = process_config(args)

    assert config["transport"] == "http"
    assert config["host"] == "127.0.0.1"
    assert config["port"] == 8000
    assert config["path"] == "/mcp/"

    # Check that warnings were logged for using defaults
    warning_calls = [call.args[0] for call in mock_logger.warning.call_args_list]
    default_warnings = [msg for msg in warning_calls if "Using default" in msg]
    assert len(default_warnings) >= 3  # host, port, path defaults


def test_env_var_port_conversion(clean_env, args_factory, mock_logger):
    """Test that environment variable port is converted to int."""
    os.environ["AGENSGRAPH_MCP_SERVER_PORT"] = "8080"
    os.environ["AGENSGRAPH_TRANSPORT"] = "http"

    args = args_factory()
    config = process_config(args)

    assert config["port"] == 8080
    assert isinstance(config["port"], int)


def test_allow_origins_cli_args(clean_env, args_factory):
    """Test allow_origins configuration from CLI arguments."""
    origins = "http://localhost:3000,https://trusted-site.com"
    expected_origins = ["http://localhost:3000", "https://trusted-site.com"]
    args = args_factory(allow_origins=origins)
    config = process_config(args)

    assert config["allow_origins"] == expected_origins


def test_allow_origins_defaults(clean_env, args_factory, mock_logger):
    """Test allow_origins uses empty list as default when not provided."""
    args = args_factory()
    config = process_config(args)

    assert config["allow_origins"] == []

    # Check that info message was logged about using defaults
    info_calls = [call.args[0] for call in mock_logger.info.call_args_list]
    allow_origins_info = [
        msg
        for msg in info_calls
        if "allow origins" in msg and "Defaulting to no" in msg
    ]
    assert len(allow_origins_info) == 1


def test_allowed_hosts_defaults(clean_env, args_factory, mock_logger):
    """Test allowed_hosts uses secure defaults when not provided."""
    args = args_factory()
    config = process_config(args)

    assert config["allowed_hosts"] == ["localhost", "127.0.0.1"]

    # Check that info message was logged about secure defaults
    info_calls = [call.args[0] for call in mock_logger.info.call_args_list]
    allowed_hosts_info = [
        msg
        for msg in info_calls
        if "allowed hosts" in msg and "secure mode" in msg
    ]
    assert len(allowed_hosts_info) == 1


class TestNamespaceConfigProcessing:
    """Test namespace configuration processing in process_config."""

    def test_process_config_namespace_cli(self, clean_env, args_factory):
        """Test process_config when namespace is provided via CLI argument."""
        args = args_factory(
            db_url="postgresql://localhost:5432",
            username="agens",
            password="agens",
            database="agens",
            graphname="memory",
            namespace="test-cli"
        )
        config = process_config(args)
        assert config["namespace"] == "test-cli"

    def test_process_config_namespace_env_var(self, clean_env, args_factory):
        """Test process_config when namespace is provided via environment variable."""
        os.environ["AGENSGRAPH_NAMESPACE"] = "test-env"
        args = args_factory(
            db_url="postgresql://localhost:5432",
            username="agens",
            password="agens",
            database="agens",
            graphname="memory"
        )
        config = process_config(args)
        assert config["namespace"] == "test-env"

    def test_process_config_namespace_precedence(self, clean_env, args_factory):
        """Test that CLI namespace argument takes precedence over environment variable."""
        os.environ["AGENSGRAPH_NAMESPACE"] = "test-env"
        args = args_factory(
            db_url="postgresql://localhost:5432",
            username="agens",
            password="agens",
            database="agens",
            graphname="memory",
            namespace="test-cli"
        )
        config = process_config(args)
        assert config["namespace"] == "test-cli"

    def test_process_config_namespace_default(self, clean_env, args_factory, mock_logger):
        """Test process_config when no namespace is provided (defaults to empty string)."""
        args = args_factory(
            db_url="postgresql://localhost:5432",
            username="agens",
            password="agens",
            database="agens",
            graphname="memory"
        )
        config = process_config(args)
        assert config["namespace"] == ""
        mock_logger.info.assert_any_call("Info: No namespace provided for tools. No namespace will be used.")
