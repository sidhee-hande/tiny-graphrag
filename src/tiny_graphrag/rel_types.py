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
CUSTOM_ENTITY_TYPES = [
    "ProjectName",
    "Party",
    "Role",
    "Person",
    "ContractDate",
    "Sector",
    "ContractName",
    "ContractType",
    "Location",
    "Country"
]


CUSTOM_RELS = {
  "has role" : {"allowed_head": ["Party"], "allowed_tail": ["Role"]},
  "signed by": {"allowed_head": ["ContractName"], "allowed_tail": ["Person"]},
  "located in": {"allowed_head": ["ProjectName"], "allowed_tail": ["Location"]},
  "in country": {"allowed_head": ["ProjectName"], "allowed_tail": ["Country"]},
  "belongs to sector": {"allowed_head": ["ProjectName"], "allowed_tail": ["Sector"]},
  "contract type is": {"allowed_head": ["ContractName"], "allowed_tail": ["ContractType"]},
  
}

MARKED_ENTITIES = [
    "Project",
    "Sector",
    "Subsector",
    "ContractName",
    "ContractDate",
    "Country",
    "Location",
    "Party",
    "Role",
    "Person",
    "ContractType"
]

MARKED_RELS = {
  "has role" : {"allowed_head": ["Party"], "allowed_tail": ["Role"]},
  "has role" : {"allowed_head": ["Person"], "allowed_tail": ["Role"]},
  "belongs to sector": {"allowed_head": ["ProjectName"], "allowed_tail": ["Sector"]},
  "belongs to subsector": {"allowed_head": ["ProjectName"], "allowed_tail": ["Subsector"]},
  "located in country": {"allowed_head": ["ProjectName"], "allowed_tail": ["Country"]},
  "located in": {"allowed_head": ["ProjectName"], "allowed_tail": ["Location"]},
  "date is": {"allowed_head": ["ContractName"], "allowed_tail": ["ContractDate"]},
  "type is": {"allowed_head": ["ContractName"], "allowed_tail": ["ContractType"]},
  "mentions": {"allowed_head": ["ContractName"], "allowed_tail": ["ProjectName"]},
  
}

MARKED_RELS_LIST = list(MARKED_RELS.keys())

CUSTOM_RELS_LIST = list(CUSTOM_RELS.keys()) 

NEW_RELS = {
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
  "has role" : {"allowed_head": ["Party"], "allowed_tail": ["Role"]},
  "has role" : {"allowed_head": ["Person"], "allowed_tail": ["Role"]},
  "belongs to sector": {"allowed_head": ["ProjectName"], "allowed_tail": ["Sector"]},
  "belongs to subsector": {"allowed_head": ["ProjectName"], "allowed_tail": ["Subsector"]},
  "located in country": {"allowed_head": ["ProjectName"], "allowed_tail": ["Country"]},
  "located in": {"allowed_head": ["ProjectName"], "allowed_tail": ["Location"]},
  "date is": {"allowed_head": ["ContractName"], "allowed_tail": ["ContractDate"]},
  "type is": {"allowed_head": ["ContractName"], "allowed_tail": ["ContractType"]},
  "mentions": {"allowed_head": ["ContractName"], "allowed_tail": ["ProjectName"]},
  

}

NEW_RELS_LIST = list(NEW_RELS.keys()) 