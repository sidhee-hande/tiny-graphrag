DEFAULT_RELS = {
    "instance of": {"allowed_head": ["Concept"], "allowed_tail": ["Class"]},
    "subclass of": {"allowed_head": ["Class"], "allowed_tail": ["Class"]},
    "related to": {"allowed_head": ["Concept"], "allowed_tail": ["Concept"]},
    "has property": {"allowed_head": ["Concept"], "allowed_tail": ["Concept"]},
    "is a": {"allowed_head": ["Concept"], "allowed_tail": ["Concept"]},
    "sells": {"allowed_head": ["Organization"], "allowed_tail": ["Person"]},
    "created": {"allowed_head": ["Person"], "allowed_tail": ["Organization"]},
    "author": {"allowed_head": ["Person"], "allowed_tail": ["Document"]},
    "location": {"allowed_head": ["Location"], "allowed_tail": ["Location"]},
    "place of birth": {"allowed_head": ["Person"], "allowed_tail": ["Location"]},
    "occupation": {"allowed_head": ["Person"], "allowed_tail": ["Concept"]},
    "won": {"allowed_head": ["Person"], "allowed_tail": ["Award"]},
    "plays for": {"allowed_head": ["Person"], "allowed_tail": ["Organization"]},
    "spouse": {"allowed_head": ["Person"], "allowed_tail": ["Person"]},
    "children": {"allowed_head": ["Person"], "allowed_tail": ["Person"]},
    "employer": {"allowed_head": ["Person"], "allowed_tail": ["Organization"]},
    "education": {"allowed_head": ["Person"], "allowed_tail": ["Organization"]},
}

DEFAULT_RELS_LIST = list(DEFAULT_RELS.keys())
