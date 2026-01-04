import re

def extract_entities_and_relations(text):
    triples = []
    sentences = text.split(".")

    for s in sentences:
        words = re.findall(r"[A-Za-z][A-Za-z0-9]+", s)
        if len(words) >= 2:
            triples.append((words[0], "RELATED_TO", words[1]))

    return triples
