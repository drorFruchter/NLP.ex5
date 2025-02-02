import spacy
import wikipedia
import random
import google.generativeai as genai


# Initialize spaCy and LLM
nlp = spacy.load("en_core_web_sm")
genai.configure(api_key="****")  # TODO: Replace with your Gemini API key


# ========================
# POS-Based Extractor
# ========================
def get_propn_spans(doc):
    """Find consecutive PROPN sequences in a sentence"""
    spans = []
    current_span = []
    for token in doc:
        if token.pos_ == "PROPN":
            current_span.append(token)
        elif current_span:
            spans.append(current_span)
            current_span = []
    if current_span:
        spans.append(current_span)
    return spans


def pos_extractor(doc):
    triplets = []
    for sent in doc.sents:
        spans = get_propn_spans(sent)
        for i in range(len(spans) - 1):
            subj_span = spans[i]
            obj_span = spans[i + 1]

            # Get tokens between spans
            start = subj_span[-1].i + 1
            end = obj_span[0].i
            between_tokens = sent.doc[start:end]

            # Check conditions
            if any(t.pos_ == "VERB" for t in between_tokens) and \
                    not any(t.pos_ == "PUNCT" for t in between_tokens):
                relation = " ".join([t.text for t in between_tokens if t.pos_ in ("VERB", "ADP")])
                subject = " ".join(t.text for t in subj_span)
                obj = " ".join(t.text for t in obj_span)
                triplets.append((subject, relation, obj))
    return triplets


# ========================
# Dependency-Based Extractor
# ========================
def get_proper_nouns(sent):
    """Return list of (head, compound_span) pairs"""
    proper_nouns = []
    for token in sent:
        if token.pos_ == "PROPN" and token.dep_ != "compound":
            compounds = [token] + [t for t in token.children if t.dep_ == "compound"]
            compounds.sort(key=lambda t: t.i)
            proper_nouns.append((token, compounds))
    return proper_nouns


def dep_extractor(doc):
    triplets = []
    for sent in doc.sents:
        proper_nouns = get_proper_nouns(sent)

        for h1, subj_tokens in proper_nouns:
            for h2, obj_tokens in proper_nouns:
                if h1 == h2:
                    continue

                # Condition 1: Shared head with nsubj/dobj
                if h1.head == h2.head:
                    if h1.head.dep_ == "ROOT" and \
                            h1.dep_ == "nsubj" and \
                            h2.dep_ == "dobj":
                        subject = " ".join(t.text for t in subj_tokens)
                        relation = h1.head.text
                        obj = " ".join(t.text for t in obj_tokens)
                        triplets.append((subject, relation, obj))

                # Condition 2: Prepositional phrase
                elif h1.head == h2.head.head and \
                        h1.dep_ == "nsubj" and \
                        h2.head.dep_ == "pobj" and \
                        h2.head.head.dep_ == "prep":
                    subject = " ".join(t.text for t in subj_tokens)
                    relation = f"{h1.head.text} {h2.head.head.text}"
                    obj = " ".join(t.text for t in obj_tokens)
                    triplets.append((subject, relation,obj))
    return triplets


# ========================
# Evaluation Functions
# ========================
def evaluate_pages(pages):
    results = {}
    for page_title in pages:
        print(f"Processing {page_title}...")
        content = wikipedia.page(page_title).content
        doc = nlp(content)

        pos_triplets = pos_extractor(doc)
        dep_triplets = dep_extractor(doc)

        # Store results
        results[page_title] = {
            'pos': {
                'count': len(pos_triplets),
                'samples': random.sample(pos_triplets, min(5, len(pos_triplets))) if pos_triplets else []
            },
            'dep': {
                'count': len(dep_triplets),
                'samples': random.sample(dep_triplets, min(5, len(dep_triplets))) if dep_triplets else []
            }
        }
    return results


# ========================
# LLM Extractor
# ========================
def llm_extractor(text):
    prompt = f"""
    Extract (Subject, Relation, Object) triplets from the text below using two extractors: POS and dependency extractors.
    
    Follow these rules:
    1. Subject and Object must be proper nouns (names of people/places)
    2. Relation must be a verb phrase
    
    Then do these steps:
    1. Report how many total triplets did you find for each extractor. (number) 
    2. For each extractor, sample 5 random triplets from what you've found.(5 triplets)
    
    Output format: (Subject, Relation, Object)
    
    The Text: {text[:4000]}
    """

    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    return response.text


# ========================
# Main Execution
# ========================
if __name__ == "__main__":
    # Part 3: Evaluate extractors
    wikipedia_pages = ["Donald Trump", "Ruth Bader Ginsburg", "J.K. Rowling"]
    results = evaluate_pages(wikipedia_pages)

    # Print evaluation results
    print("\nEvaluation Results:")
    for page, data in results.items():
        print(f"\n{page}:")
        print(f"POS Triplets: {data['pos']['count']} | Samples: {data['pos']['samples']}")
        print(f"DEP Triplets: {data['dep']['count']} | Samples: {data['dep']['samples']}")

    # Part 4: LLM Comparison (example with one page)
    for page in wikipedia_pages:
        page_content = wikipedia.page(page).content
        llm_output = llm_extractor(page_content)
        print(f"\nLLM Output for page {page}")
        print(llm_output)
