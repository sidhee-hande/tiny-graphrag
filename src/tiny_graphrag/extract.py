import spacy
import glirel  # noqa: F401 Import time side effect
from functools import cache
from dataclasses import dataclass
from typing import List, Tuple

from tiny_graphrag.rel_types import DEFAULT_RELS_LIST
from tiny_graphrag.entity_types import MIN_ENTITY_TYPES
from tiny_graphrag.config import DEVICE


@dataclass
class ExtractionResult:
    entities: List[Tuple[str, str]]  # (text, label)
    relations: List[Tuple[str, str, str]]  # (head_text, label, tail_text)

@cache
def nlp_model(threshold: float, entity_types: List[str], device: str = DEVICE):
    custom_spacy_config = {
        "gliner_model": "urchade/gliner_mediumv2.1",
        "chunk_size": 250,
        "labels": entity_types,
        "style": "ent",
        "threshold": threshold,
        "map_location": device,
    }
    spacy.require_gpu()  # type: ignore

    nlp = spacy.blank("en")
    nlp.add_pipe("gliner_spacy", config=custom_spacy_config)
    nlp.add_pipe("glirel", after="gliner_spacy")
    return nlp


def extract_rels(
    text: str, entity_types=MIN_ENTITY_TYPES, relation_types=DEFAULT_RELS_LIST, threshold: float = 0.75
) -> ExtractionResult:
    """
    Extract entities and relations from text using GLiNER and GLiREL.
    """
    nlp = nlp_model(threshold, entity_types)
    docs = list(nlp.pipe([(text, {"glirel_labels": relation_types})], as_tuples=True))
    relations = docs[0][0]._.relations

    sorted_data_desc = sorted(relations, key=lambda x: x["score"], reverse=True)

    # Extract entities
    ents = [(ent.text, ent.label_) for ent in docs[0][0].ents]

    # Extract relations
    rels = [
        (" ".join(item["head_text"]), item["label"], " ".join(item["tail_text"]))
        for item in sorted_data_desc
        if item["score"] >= threshold
    ]

    return ExtractionResult(entities=ents, relations=rels)
