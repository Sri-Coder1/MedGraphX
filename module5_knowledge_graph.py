import networkx as nx

from module4_nlp_preprocessing import build_graph_dynamic, visualize_graph_dynamic


def _normalize_entity(entity):
    if isinstance(entity, dict):
        text = entity.get("text") or entity.get("name") or entity.get("id")
        label = entity.get("label") or entity.get("type") or "ENTITY"
        return (text, label) if text else None

    if isinstance(entity, (tuple, list)) and len(entity) >= 2:
        text, label = entity[0], entity[1]
        return (text, label) if text else None

    entity_id = getattr(entity, "id", None)
    entity_name = getattr(entity, "name", None)
    entity_type = getattr(entity, "type", None)
    text = entity_name or entity_id
    label = entity_type or "ENTITY"
    return (text, label) if text else None


def _node_key(entity):
    if isinstance(entity, dict):
        return entity.get("text") or entity.get("name") or entity.get("id")

    if isinstance(entity, (tuple, list)) and entity:
        return entity[0]

    return getattr(entity, "name", None) or getattr(entity, "id", None) or entity


def _normalize_relation(relation):
    if isinstance(relation, dict):
        source = relation.get("subject") or relation.get("entity1_id") or relation.get("source")
        target = relation.get("object") or relation.get("entity2_id") or relation.get("target")
        label = relation.get("relation") or relation.get("relation_type") or relation.get("label") or ""
        confidence = relation.get("confidence")
        return source, target, label, confidence

    source = getattr(relation, "entity1_id", None) or getattr(relation, "subject", None)
    target = getattr(relation, "entity2_id", None) or getattr(relation, "object", None)
    label = getattr(relation, "relation_type", None) or getattr(relation, "relation", None) or ""
    confidence = getattr(relation, "confidence", None)
    return source, target, label, confidence


def build_knowledge_graph(entities, relations=None, data=None):
    """Build a NetworkX graph using the project's current entity/relation formats."""
    normalized_entities = []
    seen_entities = set()

    for entity in entities or []:
        normalized = _normalize_entity(entity)
        if normalized and normalized[0] not in seen_entities:
            seen_entities.add(normalized[0])
            normalized_entities.append(normalized)

    if data is not None:
        return build_graph_dynamic(normalized_entities, data=data)

    graph = build_graph_dynamic(normalized_entities)

    for relation in relations or []:
        source, target, label, confidence = _normalize_relation(relation)
        if not (source and target):
            continue

        if not graph.has_node(source):
            graph.add_node(source, type="ENTITY")
        if not graph.has_node(target):
            graph.add_node(target, type="ENTITY")

        edge_data = {}
        if label:
            edge_data["relation"] = label
            edge_data["label"] = label
        if confidence is not None:
            edge_data["confidence"] = confidence

        graph.add_edge(source, target, **edge_data)

    return graph


def visualize_graph(graph):
    """Compatibility wrapper around the current graph visualizer."""
    return visualize_graph_dynamic(graph)


def get_subgraph(graph, center_entity, depth=2):
    """Return a shallow neighborhood subgraph around the given entity."""
    center_key = _node_key(center_entity)
    if center_key not in graph:
        return nx.Graph()

    nodes = {center_key}
    current_nodes = {center_key}

    for _ in range(depth):
        next_nodes = set()
        for node in current_nodes:
            next_nodes.update(graph.neighbors(node))
        nodes.update(next_nodes)
        current_nodes = next_nodes

    return graph.subgraph(nodes).copy()
