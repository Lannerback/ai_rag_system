from src.ai.document_loaders.legal_section_tagger import LegalSectionTagger


def test_prepends_current_article_to_continuation_chunks():
    texts = [
        "Article 14 Human oversight\n\n1. High-risk AI systems shall be designed to be overseen.",
        "2. Human oversight shall aim to prevent or minimise the risks to health and safety.",
    ]
    tagged = LegalSectionTagger().tag(texts)

    assert tagged[0] == texts[0]
    assert tagged[1].startswith("Article 14 Human oversight")
    assert "Human oversight shall aim" in tagged[1]


def test_does_not_double_tag_chunk_that_names_another_article():
    texts = [
        "Article 5 Prohibited AI practices\n\nThe following practices shall be prohibited.",
        "This is subject to Article 6 conditions and applies broadly.",
    ]
    tagged = LegalSectionTagger().tag(texts)

    # Second chunk already references an Article -> left untouched (no prefix).
    assert tagged[1] == texts[1]


def test_untagged_leading_chunks_pass_through():
    texts = ["Recital text before any article heading appears in the document."]
    assert LegalSectionTagger().tag(texts) == texts


def test_switches_heading_when_new_article_starts():
    texts = [
        "Article 9 Risk management\n\nA risk management system shall be established.",
        "The process shall be iterative and run throughout the lifecycle.",
        "Article 10 Data governance\n\nTraining data shall meet quality criteria.",
        "Data sets shall be relevant and sufficiently representative.",
    ]
    tagged = LegalSectionTagger().tag(texts)

    assert tagged[1].startswith("Article 9 Risk management")
    assert tagged[3].startswith("Article 10 Data governance")
