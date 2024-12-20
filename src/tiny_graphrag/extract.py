from dataclasses import dataclass
from functools import cache
from typing import List, Tuple
import json
import glirel  # noqa: F401 Import time side effect
import spacy

from tiny_graphrag.config import DEVICE


@dataclass
class ExtractionResult:
    """Represents the result of entity and relation extraction from text.

    Contains lists of extracted entities and their relationships.
    """

    entities: List[Tuple[str, str]]  # (text, label)
    relations: List[Tuple[str, str, str]]  # (head_text, label, tail_text)


@cache
def nlp_model(threshold: float, entity_types: tuple[str], device: str = DEVICE):
    """Instantiate a spacy model with GLiNER and GLiREL components."""
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

def use_existing_ents(filepath: str):
    json_file = filepath.replace("/text.txt", "/output.json")

    f = open(json_file)
    output = dict(json.load(f))
    ents = []
    rels = []
    ents.append((output["project"], "Project"))
    ents.append((output["sector"], "Sector"))
    ents.append((output["subsector"], "Subsector"))
    ents.append((output["contract_name"], "ContractName"))
    ents.append((output["contract_date"], "ContractDate"))
    ents.append((output["contract_type"], "ContractType"))
    ents.append((output["country"], "Country"))
    ents.append((output["location"], "Location"))
    for p in output["parties"]:
        ents.append((p["party"], "Party"))
        ents.append((p["role"], "Role"))
        rels.append((p["party"],"has role",p["role"]))
        rels.append((p["party"], "party to", output["project"]))

    for p in output["people"]:
        ents.append((p["person"], "Person"))
        ents.append((p["role"], "Role"))
        rels.append((p["person"],"has role",p["role"]))
        rels.append((p["person"], "mentioned in", output["contract_name"]))
    
    rels.append((output["project"], "belongs to sector", output["sector"]))
    rels.append((output["project"], "belongs to subsector", output["subsector"]))
    rels.append((output["project"], "located in country", output["country"]))
    rels.append((output["project"], "located in", output["location"]))
    rels.append((output["contract_name"], "date is", output["contract_date"]))
    rels.append((output["contract_name"], "type is", output["contract_type"]))
    rels.append((output["contract_name"], "mentions", output["project"]))
    


    return ents, rels


def extract_rels(
    filepath: str,
    text: str,
    entity_types: List[str],
    relation_types: List[str],
    threshold: float = 0.75,
) -> ExtractionResult:
    """Extract entities and relations from text using GLiNER and GLiREL."""

    
    nlp = nlp_model(threshold, tuple(entity_types))
    docs = list(nlp.pipe([(text, {"glirel_labels": relation_types})], as_tuples=True))
    relations = docs[0][0]._.relations

    sorted_data_desc = sorted(relations, key=lambda x: x["score"], reverse=True)

    # Extract entities
    ents = [(ent.text, ent.label_) for ent in docs[0][0].ents]
    print(ents)

    # Extract relations
    rels = [
        (" ".join(item["head_text"]), item["label"], " ".join(item["tail_text"]))
        for item in sorted_data_desc
        if item["score"] >= threshold
    ]

    print(rels)
    #ents, rels = use_existing_ents(filepath)

    return ExtractionResult(entities=ents, relations=rels)
