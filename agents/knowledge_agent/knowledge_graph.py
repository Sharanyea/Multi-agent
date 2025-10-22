from rdflib import Graph, Literal, RDF, URIRef, Namespace

BC = Namespace("http://example.org/breastcancer#")

def build_graph():
    g = Graph()
    g.add((BC.Microcalcification, RDF.type, BC.Symptom))
    g.add((BC.Microcalcification, BC.indicates, BC.PossibleMalignancy))
    g.add((BC.Lump, BC.indicates, BC.MalignancyRisk))
    g.add((BC.FamilyHistory, BC.indicates, BC.IncreasedRisk))
    return g

def query_knowledge(symptom: str):
    g = build_graph()
    qres = g.query(
        f"""
        SELECT ?disease WHERE {{
            ?symptom <{BC.indicates}> ?disease .
            FILTER regex(str(?symptom), "{symptom}", "i")
        }}
        """
    )
    return [str(row[0]) for row in qres]
