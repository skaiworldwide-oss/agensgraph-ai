import json
from collections import Counter
from typing import Any

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from agensgraph import (
    DesiredIndex,
    Unique,
    create_index_statement,
    create_label_statement,
)
from agensgraph.cypher import quote_identifier
from agensgraph.introspect import constraint_name

NODE_COLOR_PALETTE = [
    ("#e3f2fd", "#1976d2"),  # Light Blue / Blue
    ("#f3e5f5", "#7b1fa2"),  # Light Purple / Purple
    ("#e8f5e8", "#388e3c"),  # Light Green / Green
    ("#fff3e0", "#f57c00"),  # Light Orange / Orange
    ("#fce4ec", "#c2185b"),  # Light Pink / Pink
    ("#e0f2f1", "#00695c"),  # Light Teal / Teal
    ("#f1f8e9", "#689f38"),  # Light Lime / Lime
    ("#fff8e1", "#ffa000"),  # Light Amber / Amber
    ("#e8eaf6", "#3f51b5"),  # Light Indigo / Indigo
    ("#efebe9", "#5d4037"),  # Light Brown / Brown
    ("#fafafa", "#424242"),  # Light Grey / Dark Grey
    ("#e1f5fe", "#0277bd"),  # Light Cyan / Cyan
    ("#f9fbe7", "#827717"),  # Light Yellow-Green / Olive
    ("#fff1f0", "#d32f2f"),  # Light Red / Red
    ("#f4e6ff", "#6a1b9a"),  # Light Violet / Violet
    ("#e6f7ff", "#1890ff"),  # Very Light Blue / Bright Blue
]


def _generate_relationship_pattern(
    start_node_label: str, relationship_type: str, end_node_label: str
) -> str:
    "Helper function to generate a pattern for a relationship."
    return f"(:{start_node_label})-[:{relationship_type}]->(:{end_node_label})"


class PropertySource(BaseModel):
    "The source of a property."

    column_name: str | None = Field(
        default=None, description="The column name this property maps to, if known."
    )
    table_name: str | None = Field(
        default=None,
        description="The name of the table this property's column is in, if known. May also be the name of a file.",
    )
    location: str | None = Field(
        default=None,
        description="The location of the property, if known. May be a file path, URL, etc.",
    )


class Property(BaseModel):
    "A AgensGraph Property."

    name: str = Field(description="The name of the property. Should be in camelCase.")
    type: str = Field(
        default="STRING",
        description="The AgensGraph type of the property. Should be all caps.",
    )
    source: PropertySource | None = Field(
        default=None, description="The source of the property, if known."
    )
    description: str | None = Field(
        default=None, description="The description of the property"
    )

    @field_validator("type")
    def validate_type(cls, v: str) -> str:
        "Validate the type."

        return v.upper()

    @classmethod
    def from_arrows(cls, arrows_property: dict[str, str]) -> "Property":
        "Convert an Arrows Property in dict format to a Property."

        description = None

        if "|" in list(arrows_property.values())[0]:
            prop_props = [
                x.strip() for x in list(arrows_property.values())[0].split("|")
            ]

            prop_type = prop_props[0]
            description = prop_props[1] if prop_props[1].lower() != "key" else None
        else:
            prop_type = list(arrows_property.values())[0]

        return cls(
            name=list(arrows_property.keys())[0],
            type=prop_type,
            description=description,
        )

    def to_arrows(self, is_key: bool = False) -> dict[str, Any]:
        "Convert a Property to an Arrows property dictionary. Final JSON string formatting is done at the data model level."
        value = f"{self.type}"
        if self.description:
            value += f" | {self.description}"
        if is_key:
            value += " | KEY"
        return {
            self.name: value,
        }


class Node(BaseModel):
    "A AgensGraph Node."

    label: str = Field(
        description="The label of the node. Should be in PascalCase.", min_length=1
    )
    key_property: Property = Field(description="The key property of the node")
    properties: list[Property] = Field(
        default_factory=list, description="The properties of the node"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="The metadata of the node. This should only be used when converting data models.",
    )

    @field_validator("properties")
    def validate_properties(
        cls, properties: list[Property], info: ValidationInfo
    ) -> list[Property]:
        "Validate the properties."
        properties = [p for p in properties if p.name != info.data["key_property"].name]

        counts = Counter([p.name for p in properties])
        for name, count in counts.items():
            if count > 1:
                raise ValueError(
                    f"Property {name} appears {count} times in node {info.data['label']}"
                )
        return properties

    def add_property(self, prop: Property) -> None:
        "Add a new property to the node."
        if prop.name in [p.name for p in self.properties]:
            raise ValueError(
                f"Property {prop.name} already exists in node {self.label}"
            )
        self.properties.append(prop)

    def remove_property(self, prop: Property) -> None:
        "Remove a property from the node."
        try:
            self.properties.remove(prop)
        except ValueError:
            pass

    @property
    def all_properties_dict(self) -> dict[str, str]:
        "Return a dictionary of all properties of the node. {property_name: property_type}"
        props = {p.name: p.type for p in self.properties} if self.properties else {}
        if self.key_property:
            props.update({self.key_property.name: f"{self.key_property.type} | KEY"})
        return props

    def get_mermaid_config_str(self) -> str:
        "Get the Mermaid configuration string for the node."
        props = [f"<br/>{self.key_property.name}: {self.key_property.type} | KEY"]
        props.extend([f"<br/>{p.name}: {p.type}" for p in self.properties])
        return f'{self.label}["{self.label}{"".join(props)}"]'

    @classmethod
    def from_arrows(cls, arrows_node_dict: dict[str, Any]) -> "Node":
        "Convert an Arrows Node to a Node."
        props = [
            Property.from_arrows({k: v})
            for k, v in arrows_node_dict["properties"].items()
            if "KEY" not in v.upper()
        ]
        keys = [
            {k: v}
            for k, v in arrows_node_dict["properties"].items()
            if "KEY" in v.upper()
        ]
        key_prop = Property.from_arrows(keys[0]) if keys else None
        metadata = {
            "position": arrows_node_dict["position"],
            "caption": arrows_node_dict["caption"],
            "style": arrows_node_dict["style"],
        }
        return cls(
            label=arrows_node_dict["labels"][0],
            key_property=key_prop,
            properties=props,
            metadata=metadata,
        )

    def to_arrows(
        self, default_position: dict[str, float] = {"x": 0.0, "y": 0.0}
    ) -> dict[str, Any]:
        "Convert a Node to an Arrows Node dictionary. Final JSON string formatting is done at the data model level."
        props = dict()
        [props.update(p.to_arrows(is_key=False)) for p in self.properties]
        props.update(self.key_property.to_arrows(is_key=True))
        return {
            "id": self.label,
            "labels": [self.label],
            "properties": props,
            "style": self.metadata.get("style", {}),
            "position": self.metadata.get("position", default_position),
            "caption": self.metadata.get("caption", ""),
        }

    def get_cypher_ingest_query_for_many_records(self) -> str:
        """
        Generate a Cypher query to ingest a list of Node records into a AgensGraph database.
        This query takes a named parameter %(records)s that is a JSONB array of dictionaries.
        Note: For AgensGraph with psycopg, use: cursor.execute(query, {"records": Jsonb(records)})
        where Jsonb is imported from psycopg.types.json
        """
        label = quote_identifier(self.label)
        key = quote_identifier(self.key_property.name)
        formatted_props = ", ".join(
            f"{quote_identifier(p.name)}: record.{quote_identifier(p.name)}"
            for p in self.properties
        )
        return f"""UNWIND %(records)s as record
MERGE (n: {label} {{{key}: record.{key}}})
SET n += {{{formatted_props}}}"""

    def get_cypher_constraint_statements(self) -> list[str]:
        """
        Generate the statements that declare the node's label and make its key unique.

        A list rather than one string, because a caller given statements joined by a separator
        has to split on it, and a label is allowed to hold whatever a label holds -- including
        the separator. Splitting there executes a fragment of a name as though it were code.

        A unique property index rather than a constraint, because an index takes
        ``IF NOT EXISTS`` and a constraint does not -- the grammar has no arm for one, so a
        constraint script reports that the relation behind its name already exists the second
        time it is run. Both statements here can be run again.

        The label is declared without a promoted column, and the size of an element is what
        decides whether that is right. A promoted key reads the key out of a column beside the
        element instead of out of the property bag. Under about two kilobytes the bag is stored
        inline, so both read the same pages -- measured identical at 2,858 buffers -- and the
        column is the slower of the two: 25.8 ms against 19.4 mapping two thousand keys on
        elements of a kilobyte. Past that the bag is stored out of line and every key read walks
        its TOAST chain, which the column does not: at twelve kilobytes an element, 36.0 ms
        against 401.0.

        These models describe elements of a few properties, which is the first case. A model
        carrying embeddings or documents is the second, and wants the promoted column -- ``id``
        cannot have one, being a reserved promoted name, and the server takes the column only as
        ``generated``.

        Naming the keys rather than reading them all is the third case and wants neither: that is
        an index lookup per key with no bag to avoid, measured 0.85x at a kilobyte and 1.00x at
        twelve. The index below is what makes it cheap.
        """
        vlabel = create_label_statement(self.label, "v")
        index = create_index_statement(
            DesiredIndex(
                self.label,
                (self.key_property.name,),
                unique=True,
                name=constraint_name(Unique(self.label, self.key_property.name)),
            ),
            if_not_exists=True,
        )
        return [vlabel, index]


class Relationship(BaseModel):
    "A AgensGraph Relationship."

    type: str = Field(
        description="The type of the relationship. Should be in SCREAMING_SNAKE_CASE.",
        min_length=1,
    )
    start_node_label: str = Field(description="The label of the start node")
    end_node_label: str = Field(description="The label of the end node")
    key_property: Property | None = Field(
        default=None, description="The key property of the relationship, if any."
    )
    properties: list[Property] = Field(
        default_factory=list, description="The properties of the relationship, if any."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="The metadata of the relationship. This should only be used when converting data models.",
    )

    @field_validator("properties")
    def validate_properties(
        cls, properties: list[Property], info: ValidationInfo
    ) -> list[Property]:
        "Validate the properties."
        if info.data.get("key_property"):
            properties = [
                p for p in properties if p.name != info.data["key_property"].name
            ]

        counts = Counter([p.name for p in properties])
        for name, count in counts.items():
            if count > 1:
                raise ValueError(
                    f"Property {name} appears {count} times in relationship {_generate_relationship_pattern(info.data['start_node_label'], info.data['type'], info.data['end_node_label'])}"
                )
        return properties

    def add_property(self, prop: Property) -> None:
        "Add a new property to the relationship."
        if prop.name in [p.name for p in self.properties]:
            raise ValueError(
                f"Property {prop.name} already exists in relationship {self.pattern}"
            )
        self.properties.append(prop)

    def remove_property(self, prop: Property) -> None:
        "Remove a property from the relationship."
        try:
            self.properties.remove(prop)
        except ValueError:
            pass

    @property
    def pattern(self) -> str:
        "Return the pattern of the relationship."
        return _generate_relationship_pattern(
            self.start_node_label, self.type, self.end_node_label
        )

    @property
    def all_properties_dict(self) -> dict[str, str]:
        "Return a dictionary of all properties of the relationship. {property_name: property_type}"

        props = {p.name: p.type for p in self.properties} if self.properties else {}
        if self.key_property:
            props.update({self.key_property.name: f"{self.key_property.type} | KEY"})
        return props

    def get_mermaid_config_str(self) -> str:
        "Get the Mermaid configuration string for the relationship."
        props = (
            [f"<br/>{self.key_property.name}: {self.key_property.type} | KEY"]
            if self.key_property
            else []
        )
        props.extend([f"<br/>{p.name}: {p.type}" for p in self.properties])
        return f"{self.start_node_label} -->|{self.type}{''.join(props)}| {self.end_node_label}"

    @classmethod
    def from_arrows(
        cls,
        arrows_relationship_dict: dict[str, Any],
        node_id_to_label_map: dict[str, str],
    ) -> "Relationship":
        "Convert an Arrows Relationship to a Relationship."
        props = [
            Property.from_arrows({k: v})
            for k, v in arrows_relationship_dict["properties"].items()
            if "KEY" not in v.upper()
        ]
        keys = [
            {k: v}
            for k, v in arrows_relationship_dict["properties"].items()
            if "KEY" in v.upper()
        ]
        key_prop = Property.from_arrows(keys[0]) if keys else None
        metadata = {
            "style": arrows_relationship_dict["style"],
        }
        return cls(
            type=arrows_relationship_dict["type"],
            start_node_label=node_id_to_label_map[arrows_relationship_dict["fromId"]],
            end_node_label=node_id_to_label_map[arrows_relationship_dict["toId"]],
            key_property=key_prop,
            properties=props,
            metadata=metadata,
        )

    def to_arrows(self) -> dict[str, Any]:
        "Convert a Relationship to an Arrows Relationship dictionary. Final JSON string formatting is done at the data model level."
        props = dict()
        [props.update(p.to_arrows(is_key=False)) for p in self.properties]
        if self.key_property:
            props.update(self.key_property.to_arrows(is_key=True))
        return {
            "fromId": self.start_node_label,
            "toId": self.end_node_label,
            "type": self.type,
            "properties": props,
            "style": self.metadata.get("style", {}),
        }

    def get_cypher_ingest_query_for_many_records(
        self, start_node_key_property_name: str, end_node_key_property_name: str
    ) -> str:
        """
        Generate a Cypher query to ingest a list of Relationship records into a AgensGraph database.
        The sourceId and targetId properties are used to match the start and end nodes.
        This query takes a named parameter %(records)s that is a JSONB array of dictionaries.
        Note: For AgensGraph with psycopg, use: cursor.execute(query, {"records": Jsonb(records)})
        where Jsonb is imported from psycopg.types.json
        """
        formatted_props = ", ".join(
            f"{quote_identifier(p.name)}: record.{quote_identifier(p.name)}"
            for p in self.properties
        )
        key_prop = (
            f" {{{quote_identifier(self.key_property.name)}: "
            f"record.{quote_identifier(self.key_property.name)}}}"
            if self.key_property
            else ""
        )

        # The two fixed field names of an ingest record. Quoted like any other, because an
        # unquoted name is lowered by the lexer and a JSON key is not: `record.sourceId` would
        # look for `sourceid`, find nothing, and the match would silently return no rows.
        source = quote_identifier("sourceId")
        target = quote_identifier("targetId")
        # A relationship with no key property merges on the pair, so it is one relationship per
        # (start, end, type): two records for the same pair leave one carrying the first
        # record's values -- measured, quantity 2 and 7 left a single relationship with 2. Right
        # for a relationship that exists once between two things, silent loss for one that
        # happens repeatedly, which is why a key property is what separates the two.
        query = f"""UNWIND %(records)s as record
MATCH (startNode: {quote_identifier(self.start_node_label)} \
{{{quote_identifier(start_node_key_property_name)}: record.{source}}})
MATCH (endNode: {quote_identifier(self.end_node_label)} \
{{{quote_identifier(end_node_key_property_name)}: record.{target}}})
MERGE (startNode)-[r:{quote_identifier(self.type)}{key_prop}]->(endNode)"""
        if formatted_props:
            # Relationship properties belong on the relationship (r), not the end node.
            query += f"""
SET r += {{{formatted_props}}}"""
        return query

    def get_cypher_constraint_statements(self) -> list[str]:
        """
        Generate the statement that declares the relationship's label.

        Declaring it matters: a write to a label that does not exist makes one, which puts DDL
        inside the write's transaction, and two writers arriving together report ``42P07`` from
        each other's label.

        No uniqueness is asserted, and a key property does not change that. The ingest above
        merges on the endpoints *and* the key, so what the key identifies is one relationship
        between one pair -- while a constraint on the property alone would assert it is unique
        across the whole label, which is a different and stronger claim. Asserting it makes the
        ingest fail on the second relationship that reuses a key between another pair.
        """
        return [create_label_statement(self.type, "e")]


class DataModel(BaseModel):
    "A AgensGraph Graph Data Model."

    nodes: list[Node] = Field(
        default_factory=list, description="The nodes of the data model"
    )
    relationships: list[Relationship] = Field(
        default_factory=list, description="The relationships of the data model"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="The metadata of the data model. This should only be used when converting data models.",
    )

    @field_validator("nodes")
    def validate_nodes(cls, nodes: list[Node]) -> list[Node]:
        "Validate the nodes."

        counts = Counter([n.label for n in nodes])
        for label, count in counts.items():
            if count > 1:
                raise ValueError(
                    f"Node with label {label} appears {count} times in data model"
                )
        return nodes

    @field_validator("relationships")
    def validate_relationships(
        cls, relationships: list[Relationship], info: ValidationInfo
    ) -> list[Relationship]:
        "Validate the relationships."

        # ensure source and target nodes exist
        for relationship in relationships:
            if relationship.start_node_label not in [
                n.label for n in info.data["nodes"]
            ]:
                raise ValueError(
                    f"Relationship {relationship.pattern} has a start node that does not exist in data model"
                )
            if relationship.end_node_label not in [n.label for n in info.data["nodes"]]:
                raise ValueError(
                    f"Relationship {relationship.pattern} has an end node that does not exist in data model"
                )

        return relationships

    @property
    def nodes_dict(self) -> dict[str, Node]:
        "Return a dictionary of the nodes of the data model. {node_label: node_dict}"
        return {n.label: n for n in self.nodes}

    @property
    def relationships_dict(self) -> dict[str, Relationship]:
        "Return a dictionary of the relationships of the data model. {relationship_pattern: relationship_dict}"
        return {r.pattern: r for r in self.relationships}

    def add_node(self, node: Node) -> None:
        "Add a new node to the data model."
        if node.label in [n.label for n in self.nodes]:
            raise ValueError(
                f"Node with label {node.label} already exists in data model"
            )
        self.nodes.append(node)

    def add_relationship(self, relationship: Relationship) -> None:
        "Add a new relationship to the data model."
        if relationship.pattern in [r.pattern for r in self.relationships]:
            raise ValueError(
                f"Relationship {relationship.pattern} already exists in data model"
            )
        self.relationships.append(relationship)

    def remove_node(self, node_label: str) -> None:
        "Remove a node from the data model."
        try:
            [self.nodes.remove(x) for x in self.nodes if x.label == node_label]
        except ValueError:
            pass

    def remove_relationship(
        self,
        relationship_type: str,
        relationship_start_node_label: str,
        relationship_end_node_label: str,
    ) -> None:
        "Remove a relationship from the data model."
        pattern = _generate_relationship_pattern(
            relationship_start_node_label,
            relationship_type,
            relationship_end_node_label,
        )
        try:
            [
                self.relationships.remove(x)
                for x in self.relationships
                if x.pattern == pattern
            ]
        except ValueError:
            pass

    def _generate_mermaid_config_styling_str(self) -> str:
        "Generate the Mermaid configuration string for the data model."
        node_color_config = ""

        for idx, node in enumerate(self.nodes):
            node_color_config += f"classDef node_{idx}_color fill:{NODE_COLOR_PALETTE[idx % len(NODE_COLOR_PALETTE)][0]},stroke:{NODE_COLOR_PALETTE[idx % len(NODE_COLOR_PALETTE)][1]},stroke-width:3px,color:#000,font-size:12px\nclass {node.label} node_{idx}_color\n\n"

        return f"""
%% Styling 
{node_color_config}
        """

    def get_mermaid_config_str(self) -> str:
        "Get the Mermaid configuration string for the data model."
        mermaid_nodes = [n.get_mermaid_config_str() for n in self.nodes]
        mermaid_relationships = [r.get_mermaid_config_str() for r in self.relationships]
        mermaid_styling = self._generate_mermaid_config_styling_str()
        nodes_formatted = "\n".join(mermaid_nodes)
        relationships_formatted = "\n".join(mermaid_relationships)
        return f"""graph TD
%% Nodes
{nodes_formatted}

%% Relationships
{relationships_formatted}

{mermaid_styling}
"""

    @classmethod
    def from_arrows(cls, arrows_data_model_dict: dict[str, Any]) -> "DataModel":
        "Convert an Arrows Data Model to a Data Model."
        nodes = [Node.from_arrows(n) for n in arrows_data_model_dict["nodes"]]
        node_id_to_label_map = {
            n["id"]: n["labels"][0] for n in arrows_data_model_dict["nodes"]
        }
        relationships = [
            Relationship.from_arrows(r, node_id_to_label_map)
            for r in arrows_data_model_dict["relationships"]
        ]
        metadata = {
            "style": arrows_data_model_dict["style"],
        }
        return cls(nodes=nodes, relationships=relationships, metadata=metadata)

    def to_arrows_dict(self) -> dict[str, Any]:
        "Convert the data model to an Arrows Data Model Python dictionary."
        node_spacing: int = 200
        y_current = 0
        arrows_nodes = []
        for idx, n in enumerate(self.nodes):
            if (idx + 1) % 5 == 0:
                y_current -= 200
            arrows_nodes.append(
                n.to_arrows(
                    default_position={"x": node_spacing * (idx % 5), "y": y_current}
                )
            )
        arrows_relationships = [r.to_arrows() for r in self.relationships]
        return {
            "nodes": arrows_nodes,
            "relationships": arrows_relationships,
            "style": self.metadata.get("style", {}),
        }

    def to_arrows_json_str(self) -> str:
        "Convert the data model to an Arrows Data Model JSON string."
        return json.dumps(self.to_arrows_dict(), indent=2)

    def get_node_cypher_ingest_query_for_many_records(self, node_label: str) -> str:
        "Generate a Cypher query to ingest a list of Node records into a AgensGraph database."
        node = self.nodes_dict[node_label]
        return node.get_cypher_ingest_query_for_many_records()

    def get_relationship_cypher_ingest_query_for_many_records(
        self,
        relationship_type: str,
        relationship_start_node_label: str,
        relationship_end_node_label: str,
    ) -> str:
        "Generate a Cypher query to ingest a list of Relationship records into a AgensGraph database."
        pattern = _generate_relationship_pattern(
            relationship_start_node_label,
            relationship_type,
            relationship_end_node_label,
        )
        relationship = self.relationships_dict[pattern]
        start_node = self.nodes_dict[relationship.start_node_label]
        end_node = self.nodes_dict[relationship.end_node_label]
        return relationship.get_cypher_ingest_query_for_many_records(
            start_node.key_property.name, end_node.key_property.name
        )

    def get_cypher_constraints_query(self) -> list[str]:
        """
        Generate a list of Cypher queries to create constraints on the data model.
        This creates range indexes on the key properties of the nodes and relationships and enforces uniqueness and existence of the key properties.
        """
        statements: list[str] = []
        for node in self.nodes:
            statements.extend(node.get_cypher_constraint_statements())
        for relationship in self.relationships:
            statements.extend(relationship.get_cypher_constraint_statements())
        return statements
