# 🔍📊 AgensGraph Data Modeling MCP Server

## 🌟 Overview

A Model Context Protocol (MCP) server implementation that provides tools for creating, visualizing, and managing AgensGraph graph data models. This server enables you to define nodes, relationships, and properties to design graph database schemas that can be visualized interactively.

## 🧩 Features

- **Data Model Creation**: Define nodes, relationships, and properties
- **Validation Tools**: Ensure structural integrity of data models
- **Visualization**: Generate Mermaid diagrams for model visualization
- **Arrows.app Integration**: Import/export models from Arrows visual editor
- **Cypher Generation**: Create ingest queries and constraint queries
- **Example Models**: 7 pre-built real-world data models

**No database connection.** Nothing here reads or writes a graph — the server takes a model as
JSON, answers about it, and hands back Cypher for somebody else to run. That somebody is
usually the cypher server's write tool, in the same conversation.

## 🔧 Installation

```bash
uvx mcp-agensgraph-data-modeling

# Or from source
git clone https://github.com/skaiworldwide-oss/agensgraph-ai.git
cd agensgraph-ai/mcp-agensgraph/servers/mcp-agensgraph-data-modeling
uv sync
```

Add it to your `claude_desktop_config.json`:

```json
"mcpServers": {
  "agensgraph-data-modeling": {
    "command": "uvx",
    "args": [ "mcp-agensgraph-data-modeling" ]
  }
}
```

CLI flags: `--transport` (`stdio`, `http`, `sse`; default `stdio`), `--server-host`,
`--server-port`, `--server-path`, `--allow-origins`, `--allowed-hosts`, `--namespace`. Each has
an environment variable — `AGENSGRAPH_TRANSPORT`, `AGENSGRAPH_MCP_SERVER_HOST` / `_PORT` /
`_PATH`, `AGENSGRAPH_MCP_SERVER_ALLOW_ORIGINS` / `_ALLOWED_HOSTS`, `AGENSGRAPH_NAMESPACE`.
`--namespace myapp` prefixes every tool name (`myapp-validate_node`), so several servers can
be reached from one client.

## 📖 Usage

### 🛠️ Tools

A model is a plain JSON object: `{"nodes": [...], "relationships": [...]}`. A node carries a
`label`, a `key_property` and its other `properties`; a relationship carries a `type`, a
`start_node_label`, an `end_node_label`, a `key_property` and its `properties`. A property is
`{"name", "type", "source", "description"}`.

**Validation** — each returns the model with defaults filled in, or names what is wrong.

| Tool | Input | What it checks |
|---|---|---|
| `validate_node` | `node` | the label, the key property and the property list are well formed |
| `validate_relationship` | `relationship` | the same, plus that the type is given |
| `validate_data_model` | `data_model` | every node and relationship, and that each relationship's endpoints name nodes the model contains |

**Interchange and drawing**

| Tool | Input | Returns |
|---|---|---|
| `load_from_arrows_json` | `arrows_data_model_dict` | a data model built from an [arrows.app](https://arrows.app) export |
| `export_to_arrows_json` | `data_model` | the model as arrows.app JSON, to open in the visual editor |
| `get_mermaid_config_str` | `data_model` | a Mermaid diagram of the model, which Claude Desktop renders inline |

**Cypher generation** — none of it is executed here.

| Tool | Input | Returns |
|---|---|---|
| `get_node_cypher_ingest_query` | `node` | `UNWIND %(records)s AS record MERGE ... SET n += {...}` for that label |
| `get_relationship_cypher_ingest_query` | `data_model`, `relationship_type`, `relationship_start_node_label`, `relationship_end_node_label` | the same shape, matching endpoints by each record's `sourceId` and `targetId` |
| `get_constraints_cypher_queries` | `data_model` | **a list, one whole statement per item**: the label declarations and a unique property index on each node's key |

The parameter placeholder is the database driver's `%(name)s`, not `$name`. Pass the batch as
`params: {"records": [...]}` — a list bound that way arrives as JSONB, which is what `UNWIND`
expects. `$records` is not rewritten by anything and reaches the server as a syntax error.

`get_constraints_cypher_queries` answers with a list because a joined script would have to be
split on a separator, and a label may hold whatever a label holds — including that separator.
Run each item as it stands. Every statement it emits carries `IF NOT EXISTS`, so applying them
again changes nothing.

No uniqueness is asserted on a relationship, even one with a key property: the generated ingest
merges on the endpoints *and* the key, so what the key identifies is one relationship between
one pair, while a constraint on the property alone would claim it is unique across the whole
label. Asserting that rejects the second relationship to reuse a key between another pair.

**Examples**

| Tool | Input | Returns |
|---|---|---|
| `list_example_data_models` | — | the seven examples with their descriptions and node/relationship counts |
| `get_example_data_model` | `example_name` | one of them, ready to hand to the tools above |

The seven: `patient_journey`, `supply_chain`, `software_dependency`, `oil_gas_monitoring`,
`customer_360`, `fraud_aml`, `health_insurance_fraud`.

### 📚 Resources

Twelve, all read-only and none of them a database lookup.

| URI | What it is |
|---|---|
| `resource://schema/node` | the JSON schema a node is validated against |
| `resource://schema/relationship` | the same for a relationship |
| `resource://schema/property` | the same for a property |
| `resource://schema/data_model` | the same for a whole model |
| `resource://static/agensgraph_data_ingest_process` | the order to ingest in — nodes before the relationships that join them |
| `resource://examples/patient_journey_model` | the patient-journey example |
| `resource://examples/supply_chain_model` | the supply-chain example |
| `resource://examples/software_dependency_model` | the software-dependency example |
| `resource://examples/oil_gas_monitoring_model` | the oil-and-gas monitoring example |
| `resource://examples/customer_360_model` | the customer-360 example |
| `resource://examples/fraud_aml_model` | the fraud/AML example |
| `resource://examples/health_insurance_fraud_model` | the health-insurance-fraud example |

### 💬 Prompt

`create_new_data_model` — one prompt, which Claude Desktop surfaces as a starter. It takes
`data_context` (what the data is, and what to pay attention to), `use_cases` (what the model
has to answer), and optionally `desired_nodes` and `desired_relationships`. Give it sample data
alongside the prompt.

## 📄 License

This MCP server is licensed under the Apache License 2.0, which is what `LICENSE`, `NOTICE` and the package metadata declare. You are free to use, modify and distribute it subject to that licence; see `LICENSE` for the terms.
