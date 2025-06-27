'''
Copyright (c) 2025, SKAI Worldwide Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from typing import Any, Dict, Union, List
import psycopg2
from functools import wraps

class AgensQueryException(Exception):
    """Exception for the Agensgraph queries."""

    def __init__(self, exception: Union[str, Dict]) -> None:
        if isinstance(exception, dict):
            self.message = exception["message"] if "message" in exception else "unknown"
            self.details = exception["details"] if "details" in exception else "unknown"
        else:
            self.message = exception
            self.details = "unknown"

    def get_message(self) -> str:
        return self.message

    def get_details(self) -> Any:
        return self.details

def execute_query(curs, query, error_message = "Error executing query"):
    try:
        curs.execute(query)
    except psycopg2.Error as e:

        raise AgensQueryException(
            {
                "message": error_message,
                "details": str(e),
            }
        )

def require_psycopg2(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            import psycopg2
        except ImportError:
            raise ImportError(
                "Could not import psycopg2 python package. "
                "Please install it with `pip install psycopg2`."
            )
        return func(*args, **kwargs)
    return wrapper

def format_triples(triples: List[Dict[str, str]]) -> List[str]:
    """
    Convert a list of relationships from dictionaries to formatted strings
    to be better readable by an llm

    Args:
        triples (List[Dict[str,str]]): a list relationships in the form
            {'start':<from_label>, 'type':<edge_label>, 'end':<from_label>}

    Returns:
        List[str]: a list of relationships in the form
            "(:"<from_label>")-[:"<edge_label>"]->(:"<to_label>")"
    """
    triple_template = '(:"{start}")-[:"{type}"]->(:"{end}")'
    triple_schema = [triple_template.format(**triple) for triple in triples]

    return triple_schema
