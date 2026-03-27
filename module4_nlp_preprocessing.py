import importlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import networkx as nx
import spacy
from pyvis.network import Network


def _load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=False)
        return spacy.load("en_core_web_sm")


NLP = _load_spacy_model()


def _setup_backend_imports():
    current_dir = Path(__file__).resolve().parent
    workspace_root = current_dir.parent
    backend_root = workspace_root / "backend"

    for candidate in (workspace_root, backend_root):
        if candidate.exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)


_setup_backend_imports()


def _safe_import(module_name):
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


BACKEND_PREPROCESSING = _safe_import("backend.nlp.preprocessing")
BACKEND_NER = _safe_import("backend.nlp.ner")
BACKEND_RELATIONS = _safe_import("backend.nlp.relation_extraction")
BACKEND_TRIPLES = _safe_import("backend.nlp.triples")
BACKEND_GRAPH = _safe_import("backend.nlp.graph_builder")
BACKEND_ONTOLOGY = _safe_import("backend.nlp.ontology")
BACKEND_CROSS_DOMAIN = _safe_import("backend.nlp.cross_domain")


def extract_text(data):
    texts = []
    if not isinstance(data, dict):
        return texts

    source = (data.get("source") or "").lower()

    if source == "wikipedia":
        text = data.get("description") or data.get("summary") or ""
        if text:
            texts.append(text)

    elif source == "arxiv":
        for paper in data.get("papers", []):
            text = " ".join(
                part for part in [
                    paper.get("title", ""),
                    paper.get("summary", ""),
                ] if part
            ).strip()
            if text:
                texts.append(text)

    elif source == "pubmed":
        for article in data.get("articles", []):
            text = " ".join(
                part for part in [
                    article.get("title", ""),
                    article.get("journal", ""),
                ] if part
            ).strip()
            if text:
                texts.append(text)

    else:
        for value in data.values():
            if isinstance(value, str) and value.strip():
                texts.append(value.strip())

    return texts


def run_nlp_pipeline(data):
    texts = extract_text(data)
    processed = []

    for text in texts:
        if not text:
            processed.append([])
            continue

        if BACKEND_PREPROCESSING and hasattr(BACKEND_PREPROCESSING, "preprocess_text"):
            doc = BACKEND_PREPROCESSING.preprocess_text(text)
        else:
            doc = NLP(text)

        if not doc:
            processed.append([])
            continue

        tokens = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop and not token.is_punct and token.text.strip()
        ]
        processed.append(tokens)

    return processed


def extract_entities_spacy(data):
    return [(item["text"], item["label"]) for item in extract_entities_with_metadata(data)]


def _extract_relations_for_texts(texts):
    relations = []
    seen = set()

    for text in texts:
        if not text:
            continue

        if BACKEND_RELATIONS and hasattr(BACKEND_RELATIONS, "extract_relations"):
            extracted_relations = BACKEND_RELATIONS.extract_relations(text)
        else:
            extracted_relations = _extract_relations_fallback(text)

        for relation in extracted_relations:
            key = (
                relation.get("subject", "").lower(),
                relation.get("relation", "").lower(),
                relation.get("object", "").lower(),
            )
            if key not in seen and all(key):
                seen.add(key)
                relations.append(relation)

    return relations


def _extract_relations_fallback(text):
    doc = NLP(text)
    relations = []
    seen = set()

    for sent in doc.sents:
        subject = None
        obj = None
        predicate = None

        for token in sent:
            if token.dep_ in {"nsubj", "nsubjpass"} and token.text.strip():
                subject = token.text.strip()
            elif token.dep_ in {"dobj", "pobj", "attr", "acomp"} and token.text.strip():
                obj = token.text.strip()
            elif token.pos_ in {"VERB", "AUX"} and token.lemma_.strip():
                predicate = token.lemma_.strip().lower()

        if subject and predicate and obj:
            key = (subject.lower(), predicate.lower(), obj.lower())
            if key not in seen:
                seen.add(key)
                relations.append({
                    "subject": subject,
                    "relation": predicate,
                    "object": obj,
                    "confidence": 0.6,
                })

    return relations


def _classify_domain(entity_text):
    if BACKEND_CROSS_DOMAIN and hasattr(BACKEND_CROSS_DOMAIN, "classify_entity"):
        return BACKEND_CROSS_DOMAIN.classify_entity(entity_text)
    return "General"


def _normalize_entities(entities):
    if not BACKEND_ONTOLOGY or not hasattr(BACKEND_ONTOLOGY, "unify_entity"):
        return entities

    normalized = []
    seen = set()
    for entity, label in entities:
        unified = BACKEND_ONTOLOGY.unify_entity(entity)
        key = (unified.lower(), label)
        if unified and key not in seen:
            seen.add(key)
            normalized.append((unified, label))
    return normalized


def extract_entities_with_metadata(data):
    texts = extract_text(data)
    entities = []
    seen = set()

    for text in texts:
        if not text:
            continue

        if BACKEND_NER and hasattr(BACKEND_NER, "extract_entities"):
            extracted = BACKEND_NER.extract_entities(text)
            for item in extracted:
                raw_text = (item.get("text") or "").strip()
                label = item.get("label", "ENTITY")
                if not raw_text:
                    continue

                canonical_text = raw_text
                if BACKEND_ONTOLOGY and hasattr(BACKEND_ONTOLOGY, "unify_entity"):
                    canonical_text = BACKEND_ONTOLOGY.unify_entity(raw_text)

                key = (canonical_text.lower(), label)
                if key in seen:
                    continue

                seen.add(key)
                entities.append({
                    "text": canonical_text,
                    "raw_text": raw_text,
                    "label": label,
                    "domain": _classify_domain(canonical_text),
                })
        else:
            doc = NLP(text)
            for ent in doc.ents:
                raw_text = ent.text.strip()
                if not raw_text:
                    continue

                key = (raw_text.lower(), ent.label_)
                if key in seen:
                    continue

                seen.add(key)
                entities.append({
                    "text": raw_text,
                    "raw_text": raw_text,
                    "label": ent.label_,
                    "domain": _classify_domain(raw_text),
                })

    return entities


def extract_relations_from_data(data):
    texts = extract_text(data)
    raw_relations = _extract_relations_for_texts(texts)
    relations = []
    seen = set()

    for relation in raw_relations:
        subject = (relation.get("subject") or "").strip()
        predicate = (relation.get("relation") or "").strip()
        obj = (relation.get("object") or "").strip()

        if not (subject and predicate and obj):
            continue

        if BACKEND_ONTOLOGY and hasattr(BACKEND_ONTOLOGY, "unify_entity"):
            subject = BACKEND_ONTOLOGY.unify_entity(subject)
            obj = BACKEND_ONTOLOGY.unify_entity(obj)

        subject_domain = _classify_domain(subject)
        object_domain = _classify_domain(obj)
        cross_domain = (
            subject_domain != "General"
            and object_domain != "General"
            and subject_domain != object_domain
        )

        key = (subject.lower(), predicate.lower(), obj.lower())
        if key in seen:
            continue

        seen.add(key)
        relations.append({
            "subject": subject,
            "relation": predicate,
            "object": obj,
            "confidence": float(relation.get("confidence", 0.75)),
            "subject_domain": subject_domain,
            "object_domain": object_domain,
            "cross_domain": cross_domain,
        })

    return relations


def extract_triples_from_data(data):
    relations = extract_relations_from_data(data)
    if BACKEND_TRIPLES and hasattr(BACKEND_TRIPLES, "build_triples"):
        triples = BACKEND_TRIPLES.build_triples(relations)
    else:
        triples = [
            {"subject": r["subject"], "relation": r["relation"], "object": r["object"], "confidence": r["confidence"]}
            for r in relations
        ]

    enriched = []
    for triple in triples:
        subject = triple.get("subject", "")
        obj = triple.get("object", "")
        subject_domain = _classify_domain(subject)
        object_domain = _classify_domain(obj)
        enriched.append({
            **triple,
            "subject_domain": subject_domain,
            "object_domain": object_domain,
            "cross_domain": (
                subject_domain != "General"
                and object_domain != "General"
                and subject_domain != object_domain
            ),
        })
    return enriched


def analyze_knowledge_graph_data(data):
    entities = extract_entities_with_metadata(data)
    relations = extract_relations_from_data(data)
    triples = extract_triples_from_data(data)
    entity_pairs = [(item["text"], item["label"]) for item in entities]
    graph = build_graph_dynamic(entity_pairs, data=data)
    return {
        "texts": extract_text(data),
        "entities": entities,
        "relations": relations,
        "triples": triples,
        "graph": graph,
    }


def build_graph_dynamic(entities, data=None):
    normalized_entities = _normalize_entities(entities)

    if data is not None and BACKEND_TRIPLES and BACKEND_GRAPH:
        triples = extract_triples_from_data(data)

        if triples:
            graph = BACKEND_GRAPH.build_graph(triples)

            if BACKEND_CROSS_DOMAIN and hasattr(BACKEND_CROSS_DOMAIN, "detect_cross_domain"):
                cross_links = BACKEND_CROSS_DOMAIN.detect_cross_domain(triples)
                link_map = {
                    (item["subject"], item["object"], item["relation"]): item
                    for item in cross_links
                }
                for source, target, edge_data in graph.edges(data=True):
                    key = (source, target, edge_data.get("relation", ""))
                    link_data = link_map.get(key)
                    if link_data:
                        edge_data["cross_domain"] = link_data.get("cross_domain", False)
                        edge_data["subject_domain"] = link_data.get("subject_domain", "General")
                        edge_data["object_domain"] = link_data.get("object_domain", "General")

            for entity, label in normalized_entities:
                if not graph.has_node(entity):
                    graph.add_node(entity, type=label)
                else:
                    graph.nodes[entity]["type"] = graph.nodes[entity].get("type", label)

            return graph

    graph = nx.Graph()
    max_edges = 50
    edge_count = 0

    for entity, label in normalized_entities:
        graph.add_node(entity, type=label)

    for i in range(len(normalized_entities)):
        for j in range(i + 1, len(normalized_entities)):
            if edge_count >= max_edges:
                break

            left_entity, left_type = normalized_entities[i]
            right_entity, right_type = normalized_entities[j]

            if left_entity != right_entity and left_type == right_type:
                graph.add_edge(left_entity, right_entity, relation="similar")
                edge_count += 1

    return graph


def visualize_graph_dynamic(graph):
    net = Network(
        height="760px",
        width="100%",
        bgcolor="#f7fbff",
        font_color="#17343b",
        directed=isinstance(graph, nx.DiGraph),
    )

    color_map = {
        "PERSON": "#2563eb",
        "ORG": "#7c3aed",
        "GPE": "#16a34a",
        "DATE": "#f59e0b",
        "NOUN_CHUNK": "#0ea5e9",
        "entity": "#64748b",
    }

    domain_color_map = {
        "Healthcare": "#ef4444",
        "Technology": "#3b82f6",
        "Science": "#10b981",
        "Business": "#f59e0b",
        "Education": "#8b5cf6",
        "Climate": "#14b8a6",
        "General": "#64748b",
    }

    degree_map = dict(graph.degree())

    for node, data in graph.nodes(data=True):
        node_type = data.get("type", "OTHER")
        node_domain = "General"
        if BACKEND_CROSS_DOMAIN and hasattr(BACKEND_CROSS_DOMAIN, "classify_entity"):
            node_domain = BACKEND_CROSS_DOMAIN.classify_entity(str(node))

        color = domain_color_map.get(node_domain) if node_domain != "General" else color_map.get(node_type, "#64748b")
        title = f"{node} ({node_type})"
        if node_domain != "General":
            title = f"{title} - {node_domain}"

        node_size = 20 + min(degree_map.get(node, 0) * 3, 16)
        net.add_node(
            str(node),
            label=str(node),
            title=title,
            color=color,
            size=node_size,
            borderWidth=2,
            shape="dot",
        )

    for source, target, data in graph.edges(data=True):
        relation = data.get("label") or data.get("relation", "")
        cross_domain = data.get("cross_domain", False)
        edge_color = "#ef4444" if cross_domain else "#94a3b8"
        edge_width = 3 if cross_domain else 2
        edge_title = relation
        if data.get("subject_domain") and data.get("object_domain"):
            edge_title = f"{relation} ({data.get('subject_domain')} -> {data.get('object_domain')})"
        net.add_edge(
            str(source),
            str(target),
            title=edge_title,
            color=edge_color,
            width=edge_width,
            smooth={"type": "dynamic"},
        )

    try:
        net.toggle_physics(True)
        net.set_options(json.dumps({
            "interaction": {
                "hover": True,
                "tooltipDelay": 120,
                "navigationButtons": True,
                "keyboard": True
            },
            "nodes": {
                "font": {
                    "size": 16,
                    "face": "Segoe UI",
                    "color": "#17343b",
                    "strokeWidth": 4,
                    "strokeColor": "#ffffff"
                },
                "shadow": {
                    "enabled": True,
                    "color": "rgba(15, 79, 168, 0.12)",
                    "size": 10,
                    "x": 0,
                    "y": 4
                }
            },
            "edges": {
                "font": {
                    "size": 12,
                    "align": "middle",
                    "background": "rgba(255,255,255,0.85)"
                },
                "shadow": False,
                "selectionWidth": 2
            },
            "physics": {
                "enabled": True,
                "stabilization": {
                    "enabled": True,
                    "iterations": 900,
                    "updateInterval": 25
                },
                "barnesHut": {
                    "gravitationalConstant": -5000,
                    "centralGravity": 0.18,
                    "springLength": 170,
                    "springConstant": 0.045,
                    "damping": 0.14,
                    "avoidOverlap": 0.9
                }
            }
        }))
    except Exception:
        pass

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.save_graph(temp_file.name)
    return temp_file.name
