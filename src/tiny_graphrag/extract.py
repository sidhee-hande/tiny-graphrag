import spacy
from functools import cache

from tiny_graphrag.rel_types import DEFAULT_RELS_LIST
from tiny_graphrag.entity_types import MIN_ENTITY_TYPES
from tiny_graphrag.config import DEVICE

@cache
def nlp_model(threshold: float, device: str = DEVICE):
    custom_spacy_config = {
        "gliner_model": "urchade/gliner_mediumv2.1",
        "chunk_size": 250,
        "labels": MIN_ENTITY_TYPES,
        "style": "ent",
        "threshold": threshold,
        "map_location": device,
    }
    spacy.require_gpu()

    nlp = spacy.blank("en")
    nlp.add_pipe("gliner_spacy", config=custom_spacy_config)
    nlp.add_pipe("glirel", after="gliner_spacy")
    return nlp


def extract_rels(text: str, labels = DEFAULT_RELS_LIST, threshold: float = 0.75):
    nlp = nlp_model(threshold)
    docs = list(nlp.pipe([(text, {"glirel_labels": labels})], as_tuples=True))
    relations = docs[0][0]._.relations

    sorted_data_desc = sorted(relations, key=lambda x: x["score"], reverse=True)
    
    ents = [(ent.text, ent.label_) for ent in docs[0][0].ents]
    rels = [
        (" ".join(item["head_text"]), item["label"], " ".join(item["tail_text"]))
        for item in sorted_data_desc
        if item["score"] >= threshold
    ]

    return ents, rels