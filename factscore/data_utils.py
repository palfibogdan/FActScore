import json
import pandas as pd

def get_selfcheck_topic(dp):
    concept = ""
    mid_names = ["of", "de", "van", "von", "the", "The"]
    concept_list = dp["wiki_bio_text"].split(" ")
    for string in concept_list:
        if string in mid_names or string.isupper() or string[0].isupper():
            if string[0] != "(":
                concept += string + " "
        else:
            break

    return concept.strip()


def get_selfcheck_data(path):
    data = json.load(open(path, "r"))
    topics, generations, contexts = [], [], []

    for idx, dp in enumerate(data):
        if idx == 10:
            break
        topics.append(get_selfcheck_topic(dp))
        generations.append(dp["gpt3_text"])
        contexts.append(dp["wiki_bio_text"])

    return topics, generations, contexts


def get_scopus_data(path):
    hard = False
    #TODO maybe try to remove the citation in order to not create atomic facts
    hard_questions = ["potential of class activation mapping in cxr",
                      "how does the percentage of agricultural products used for fish feed production in europe compare to other regions?",
                      "what are some notable achievements or publications by roberta fusaro in her career?",
                      "what is the process of water sowing and how does it differ from traditional methods of irrigation?"]

    data = pd.read_feather(path)
    topics, generations, contexts = [], [], []

    if hard:
        for hq in hard_questions:
            dp = data[data["query"] == hq].iloc[0]
            topics.append(dp["query"])
            generations.append(dp["summary"])
            docs = list(dp["retrieved_documents"])
            # "[1] Title:... Abstract:..., [2] Title:... Abstract:..."
            # Maybe also adapt prompt
            # context = ' '.join(f'{idx + 1}')
            contexts.append(' '.join(docs))

    for idx, dp in data.iterrows():

        generations.append(dp["summary"])
        topics.append(dp["query"])

        docs = list(dp["retrieved_documents"])
        # contexts.append(' '.join(docs))
        # "[1] Title:... Abstract:..., [2] Title:... Abstract:..."
        # Add the number of the document to the context
        contexts = ' '.join(f'[{idx_doc + 1}] {doc}' for idx_doc, doc in enumerate(docs))

    return topics, generations, contexts