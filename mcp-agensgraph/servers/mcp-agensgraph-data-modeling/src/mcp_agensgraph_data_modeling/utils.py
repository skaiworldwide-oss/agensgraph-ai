import argparse
import logging
import os
import re
from typing import Literal, Union

logger = logging.getLogger(__name__)

ALLOWED_TRANSPORTS = ["stdio", "http", "sse"]

def format_namespace(namespace: str) -> str:
    """
    Format the namespace to ensure it ends with a hyphen.

    Parameters
    ----------
    namespace : str
        The namespace to format.

    Returns
    -------
    formatted_namespace : str
        The namespace in format: namespace-toolname
    """
    if namespace:
        if namespace.endswith("-"):
            return namespace
        else:
            return namespace + "-"
    else:
        return ""
    
def parse_transport(args: argparse.Namespace) -> Literal["stdio", "http", "sse"]:
    """
    Parse the transport from the command line arguments or environment variables.

    Parameters
    ----------
    args : argparse.Namespace
        The command line arguments.

    Returns
    -------
    transport : str
    The transport.

    Raises
    ------
    ValueError: If no transport is provided or is invalid.
    """

    # parse transport
    if args.transport is not None:
        if args.transport not in ALLOWED_TRANSPORTS:
            logger.error(
                f"Invalid transport: {args.transport}. Allowed transports are: {ALLOWED_TRANSPORTS}"
            )
            raise ValueError(
                f"Invalid transport: {args.transport}. Allowed transports are: {ALLOWED_TRANSPORTS}"
            )
        return args.transport
    else:
        if os.getenv("AGENSGRAPH_TRANSPORT") is not None:
            if os.getenv("AGENSGRAPH_TRANSPORT") not in ALLOWED_TRANSPORTS:
                logger.error(
                    f"Invalid transport: {os.getenv('AGENSGRAPH_TRANSPORT')}. Allowed transports are: {ALLOWED_TRANSPORTS}"
                )
                raise ValueError(
                    f"Invalid transport: {os.getenv('AGENSGRAPH_TRANSPORT')}. Allowed transports are: {ALLOWED_TRANSPORTS}"
                )
            return os.getenv("AGENSGRAPH_TRANSPORT")
        else:
            logger.info("Info: No transport type provided. Using default: stdio")
            return "stdio"


def parse_server_host(
    args: argparse.Namespace, transport: Literal["stdio", "http", "sse"]
) -> str:
    """
    Parse the server host from the command line arguments or environment variables.

    Parameters
    ----------
    args : argparse.Namespace
        The command line arguments.
    transport : Literal["stdio", "http", "sse"]
        The transport.

    Returns
    -------
    server_host : str
    The server host.
    """
    # check cli argument
    if args.server_host is not None:
        if transport == "stdio":
            logger.warning(
                "Warning: Server host provided, but transport is `stdio`. The `server_host` argument will be set, but ignored."
            )
        return args.server_host
    # check environment variable
    else:
        # if environment variable exists
        if os.getenv("AGENSGRAPH_MCP_SERVER_HOST") is not None:
            if transport == "stdio":
                logger.warning(
                    "Warning: Server host provided, but transport is `stdio`. The `AGENSGRAPH_MCP_SERVER_HOST` environment variable will be set, but ignored."
                )
            return os.getenv("AGENSGRAPH_MCP_SERVER_HOST")
        # if environment variable does not exist and not using stdio transport
        elif transport != "stdio":
            logger.warning(
                "Warning: No server host provided and transport is not `stdio`. Using default server host: 127.0.0.1"
            )
            return "127.0.0.1"
        # if environment variable does not exist and using stdio transport
        else:
            logger.info(
                "Info: No server host provided and transport is `stdio`. `server_host` will be None."
            )
            return None


def parse_server_port(
    args: argparse.Namespace, transport: Literal["stdio", "http", "sse"]
) -> int:
    """
    Parse the server port from the command line arguments or environment variables.

    Parameters
    ----------
    args : argparse.Namespace
        The command line arguments.
    transport : Literal["stdio", "http", "sse"]
        The transport.

    Returns
    -------
    server_port : int
    The server port.
    """
    # check cli argument
    if args.server_port is not None:
        if transport == "stdio":
            logger.warning(
                "Warning: Server port provided, but transport is `stdio`. The `server_port` argument will be set, but ignored."
            )
        return args.server_port
    # check environment variable
    else:
        # if environment variable exists
        if os.getenv("AGENSGRAPH_MCP_SERVER_PORT") is not None:
            if transport == "stdio":
                logger.warning(
                    "Warning: Server port provided, but transport is `stdio`. The `AGENSGRAPH_MCP_SERVER_PORT` environment variable will be set, but ignored."
                )
            return int(os.getenv("AGENSGRAPH_MCP_SERVER_PORT"))
        # if environment variable does not exist and not using stdio transport
        elif transport != "stdio":
            logger.warning(
                "Warning: No server port provided and transport is not `stdio`. Using default server port: 8000"
            )
            return 8000
        # if environment variable does not exist and using stdio transport
        else:
            logger.info(
                "Info: No server port provided and transport is `stdio`. `server_port` will be None."
            )
            return None


def parse_server_path(
    args: argparse.Namespace, transport: Literal["stdio", "http", "sse"]
) -> str:
    """
    Parse the server path from the command line arguments or environment variables.

    Parameters
    ----------
    args : argparse.Namespace
        The command line arguments.
    transport : Literal["stdio", "http", "sse"]
        The transport.

    Returns
    -------
    server_path : str
    The server path.
    """
    # check cli argument
    if args.server_path is not None:
        if transport == "stdio":
            logger.warning(
                "Warning: Server path provided, but transport is `stdio`. The `server_path` argument will be set, but ignored."
            )
        return args.server_path
    # check environment variable
    else:
        # if environment variable exists
        if os.getenv("AGENSGRAPH_MCP_SERVER_PATH") is not None:
            if transport == "stdio":
                logger.warning(
                    "Warning: Server path provided, but transport is `stdio`. The `AGENSGRAPH_MCP_SERVER_PATH` environment variable will be set, but ignored."
                )
            return os.getenv("AGENSGRAPH_MCP_SERVER_PATH")
        # if environment variable does not exist and not using stdio transport
        elif transport != "stdio":
            logger.warning(
                "Warning: No server path provided and transport is not `stdio`. Using default server path: /mcp/"
            )
            return "/mcp/"
        # if environment variable does not exist and using stdio transport
        else:
            logger.info(
                "Info: No server path provided and transport is `stdio`. `server_path` will be None."
            )
            return None


def parse_allow_origins(args: argparse.Namespace) -> list[str]:
    """
    Parse the allow origins from the command line arguments or environment variables.

    Parameters
    ----------
    args : argparse.Namespace
        The command line arguments.

    Returns
    -------
    allow_origins : list[str]
    The allow origins.
    """
    # check cli argument
    if args.allow_origins is not None:
        # Handle comma-separated string from CLI
        return [
            origin.strip() for origin in args.allow_origins.split(",") if origin.strip()
        ]
    # check environment variable.
    else:
        if os.getenv("AGENSGRAPH_MCP_SERVER_ALLOW_ORIGINS") is not None:
            # split comma-separated string into list.
            return [
                origin.strip()
                for origin in os.getenv("AGENSGRAPH_MCP_SERVER_ALLOW_ORIGINS", "").split(",")
                if origin.strip()
            ]
        else:
            logger.info(
                "Info: No allow origins provided. Defaulting to no allowed origins."
            )
            return list()


def parse_allowed_hosts(args: argparse.Namespace) -> list[str]:
    """
    Parse the allowed hosts from the command line arguments or environment variables.

    Parameters
    ----------
    args : argparse.Namespace
        The command line arguments.

    Returns
    -------
    allowed_hosts : list[str]
    The allowed hosts.
    """
    # check cli argument
    if args.allowed_hosts is not None:
        # Handle comma-separated string from CLI
        return [host.strip() for host in args.allowed_hosts.split(",") if host.strip()]

    else:
        if os.getenv("AGENSGRAPH_MCP_SERVER_ALLOWED_HOSTS") is not None:
            # split comma-separated string into list
            return [
                host.strip()
                for host in os.getenv("AGENSGRAPH_MCP_SERVER_ALLOWED_HOSTS", "").split(",")
                if host.strip()
            ]
        else:
            logger.info(
                "Info: No allowed hosts provided. Defaulting to secure mode - only localhost and 127.0.0.1 allowed."
            )
            return ["localhost", "127.0.0.1"]

def parse_namespace(args: argparse.Namespace) -> str:
    """
    Parse the namespace from the command line arguments or environment variables.
    """
        # namespace configuration
    if args.namespace is not None:
        logger.info(f"Info: Namespace provided for tools: {args.namespace}")
        return args.namespace
    else:
        if os.getenv("AGENSGRAPH_NAMESPACE") is not None:
            logger.info(f"Info: Namespace provided for tools: {os.getenv('AGENSGRAPH_NAMESPACE')}")
            return os.getenv("AGENSGRAPH_NAMESPACE")
        else:
            logger.info("Info: No namespace provided for tools. No namespace will be used.")
            return ""

def process_config(args: argparse.Namespace) -> dict[str, Union[str, int, None]]:
    """
    Process the command line arguments and environment variables to create a config dictionary.
    This may then be used as input to the main server function.
    If any value is not provided, then a warning is logged and a default value is used, if appropriate.

    Parameters
    ----------
    args : argparse.Namespace
        The command line arguments.

    Returns
    -------
    config : dict[str, str]
        The configuration dictionary.
    """

    config = dict()

    # server configuration
    config["transport"] = parse_transport(args)
    config["host"] = parse_server_host(args, config["transport"])
    config["port"] = parse_server_port(args, config["transport"])
    config["path"] = parse_server_path(args, config["transport"])

    # namespace configuration
    config["namespace"] = parse_namespace(args)

    # middleware configuration
    config["allow_origins"] = parse_allow_origins(args)
    config["allowed_hosts"] = parse_allowed_hosts(args)



    return config


def _quote_identifiers(query: str) -> str:
    """
    Quote identifiers with uppercase letters for AgensGraph case sensitivity.

    AgensGraph (like PostgreSQL) treats unquoted identifiers as case-insensitive (lowercase).
    To preserve case, identifiers must be quoted with double quotes.

    This function automatically quotes:
    - Labels that contain uppercase letters: :Person -> :"Person"
    - Property names that contain uppercase letters: .Name -> ."Name"
    - Label names after ON keyword: ON Person -> ON "Person"

    Examples:
        MATCH (p:Person) -> MATCH (p:"Person")
        MATCH (p: Person) -> MATCH (p: "Person")
        RETURN p.FirstName -> RETURN p."FirstName"
        CREATE (n:MyLabel {MyProp: 'value'}) -> CREATE (n:"MyLabel" {"MyProp": 'value'})
        CREATE CONSTRAINT x ON Person -> CREATE CONSTRAINT x ON "Person"
        CREATE VLABEL IF NOT EXISTS Person -> CREATE VLABEL IF NOT EXISTS "Person"
        CREATE ELABEL IF NOT EXISTS Friend -> CREATE ELABEL IF NOT EXISTS "Friend"
    """
    # Quote VLABEL/ELABEL IF NOT EXISTS Label: VLABEL IF NOT EXISTS Label -> VLABEL IF NOT EXISTS "Label"
    # Do this FIRST to avoid conflicts with other patterns
    query = re.sub(
        r'\b(VLABEL|ELABEL)\s+IF\s+NOT\s+EXISTS\s+(?!")([A-Za-z][a-zA-Z0-9_]*)\b',
        r'\1 IF NOT EXISTS "\2"',
        query
    )

    # Quote label names after ON keyword: ON Label -> ON "Label"
    # Used in constraint syntax: CREATE CONSTRAINT x ON Person ASSERT ...
    # Must exclude keywords that might follow ON
    query = re.sub(
        r'\bON\s+(?!")([A-Za-z][a-zA-Z0-9_]*)\b(?!\s+ASSERT)',
        r'ON "\1"',
        query
    )
    
    # Quote the label after ON when followed by ASSERT
    query = re.sub(
        r'\bON\s+(?!")([A-Za-z][a-zA-Z0-9_]*)\b(\s+ASSERT)',
        r'ON "\1"\2',
        query
    )

    # Quote labels with uppercase: :Label or : Label -> :"Label"
    # Avoid already quoted labels: :"Label" should remain unchanged
    # Handle optional whitespace after colon
    # Must have at least one uppercase letter to be quoted
    query = re.sub(
        r':\s*(?!")([A-Z][a-zA-Z0-9_]*)\b',
        r': "\1"',
        query
    )

    # Quote property names with uppercase in patterns: {PropName: -> {"PropName":
    # This handles property names in CREATE/MERGE/SET statements
    # Match properties after { or , to handle multiple properties
    # Updated to handle both PascalCase and camelCase (e.g., productId, userId)
    query = re.sub(
        r'([{,]\s*)([a-zA-Z][a-zA-Z0-9_]*[A-Z][a-zA-Z0-9_]*)\s*:',
        r'\1"\2":',
        query
    )

    # Quote property access with uppercase: .PropName -> ."PropName"
    # Avoid already quoted properties
    # Updated to handle both PascalCase and camelCase (e.g., .productId, .userId)
    query = re.sub(
        r'\.(?!")([a-zA-Z][a-zA-Z0-9_]*[A-Z][a-zA-Z0-9_]*)\b',
        r'."\1"',
        query
    )

    # Quote property names in ASSERT clause (for constraints)
    # ASSERT propertyName IS UNIQUE -> ASSERT "propertyName" IS UNIQUE
    # Handles both PascalCase and camelCase property names
    query = re.sub(
        r'\bASSERT\s+(?!")([a-zA-Z][a-zA-Z0-9_]*[A-Z][a-zA-Z0-9_]*)\b',
        r'ASSERT "\1"',
        query
    )

    return query
