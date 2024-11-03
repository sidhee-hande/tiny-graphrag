import spacy
from glirel import GLiREL
from gliner import GLiNER
from functools import cache

from tiny_graphrag.rel_types import rel_labels_list
from tiny_graphrag.entity_types import MIN_ENTITY_TYPES

# Initialize GLiNER with the base model
model1 = GLiNER.from_pretrained("urchade/gliner_mediumv2.1")
model2 = GLiREL.from_pretrained("jackboyla/glirel_beta")


def extract_ents(text: str, threshold: float = 0.5) -> list[tuple[str, str]]:
    # Labels for entity prediction
    # Most GLiNER models should work best when entity types are in lower case or title case
    labels = MIN_ENTITY_TYPES
    # Perform entity prediction
    entities = model1.predict_entities(text, labels, threshold=threshold)
    # Display predicted entities and their labels
    for entity in entities:
        print(entity["text"], "=>", entity["label"])
    return [(entity["text"], entity["label"]) for entity in entities]


@cache
def fetch_model():
    custom_spacy_config = {
        "gliner_model": "urchade/gliner_mediumv2.1",
        "chunk_size": 250,
        "labels": [
            "Person",
            "Organization",
            "Location",
            "Product",
            "Concept",
            "Property",
            "Class",
        ],
        "style": "ent",
        "threshold": 0.75,
        # "map_location": "cuda",
    }
    # from thinc.api import get_current_ops
    # print(get_current_ops())
    spacy.require_gpu()

    nlp = spacy.blank("en")
    nlp.add_pipe("gliner_spacy", config=custom_spacy_config)
    nlp.add_pipe("glirel", after="gliner_spacy")
    return nlp


def extract_rels(text: str, threshold: float = 0.2):
    # Configure spaCy with GLiNER
    # Add the labels to the pipeline at inference time
    nlp = fetch_model()

    docs = list(nlp.pipe([(text, {"glirel_labels": rel_labels_list})], as_tuples=True))
    relations = docs[0][0]._.relations

    # tokens = [token.text for token in docs[0][0]]
    # relations = model2.predict_relations(tokens, rel_labels_list, threshold=0.0, ner=ner, top_k=1)

    # Get the entities
    # print("Entities:")
    # for ent in docs[0][0].ents:
    #     print(ent.text, ent.label_, ent._.score)

    # print("Number of relations:", len(relations))

    sorted_data_desc = sorted(relations, key=lambda x: x["score"], reverse=True)
    # print("\nDescending Order by Score:")

    # for item in sorted_data_desc:
    #     if item["score"] >= threshold:
    #         print(
    #             f"{item['head_text']} => {item['label']} => {item['tail_text']} | score: {item['score']}"
    #         )

    ents = [(ent.text, ent.label_) for ent in docs[0][0].ents]

    rels = [
        (" ".join(item["head_text"]), item["label"], " ".join(item["tail_text"]))
        for item in sorted_data_desc
        if item["score"] >= threshold
    ]

    return ents, rels


exmaple = """
Cristiano Ronaldo dos Santos Aveiro GOIH ComM (Portuguese pronunciation: born 5 February 1985)
is a Portuguese professional footballer who plays as a forward for and captains both Saudi Pro League club Al Nassr and
the Portugal national team. Widely regarded as one of the greatest players of all time, Ronaldo set numerous records
for individual accolades won throughout his professional footballing career, such as five Ballon d'Or awards, a record
three UEFA Men's Player of the Year Awards, four European Golden Shoes, and was named five times the world's best player
by FIFA,[note 3] the most by a European player. He has won 33 trophies in his career, including seven league titles,
five UEFA Champions Leagues, the UEFA European Championship and the UEFA Nations League. Ronaldo holds the records
for most appearances (183), goals (140) and assists (42) in the Champions League, most appearances (30), assists (8),
goals in the European Championship (14), international appearances (216) and international goals (133). He is one of
the few players to have made over 1,200 professional career appearances, the most by an outfield player, and has
scored over 900 official senior career goals for club and country, making him the top goalscorer of all time
"""

if __name__ == "__main__":
    ents, rels = extract_rels(exmaple)
    for rel in rels:
        print(rel)
