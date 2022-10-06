import difflib
import edit_distance
import spacy
nlp = spacy.load('en_core_web_sm')
from common_utils.multiwoz_data import remove_ws_before_punctuation

class OPETYPE:
    EQUAL = edit_distance.EQUAL
    INSERTION = edit_distance.INSERT
    DELETION = edit_distance.DELETE
    SUBSTITUTION = edit_distance.REPLACE

def tokenize(text):
    doc = nlp(text)
    tokens = [token.text for token in doc]
    return doc, tokens

def detokenize(tokens):
    text = " ".join(tokens)
    text = remove_ws_before_punctuation(text)
    return text

def get_confusion(text_a, text_b):
    doc_a, words_a = tokenize(text_a)
    doc_b, words_b = tokenize(text_b)
    if not words_b:
        return [], [], [], words_a
    
    equals, substitutions, insertions, deletions = [], [], [], []
    matcher = edit_distance.SequenceMatcher(a=words_a, b=words_b)
    for ope, a_start, a_end, b_start, b_end in matcher.get_opcodes():
        assert a_end - a_start <= 1
        assert b_end - b_start <= 1
        word_a = words_a[a_start]
        word_b = words_b[b_start]
        if ope == OPETYPE.EQUAL:
            equals.append(word_a)
        elif ope == OPETYPE.SUBSTITUTION:
            if word_a.lower() != word_b.lower():
                substitutions.append((word_a, word_b))
            else:
                equals += [word_a, word_b]
        elif ope == OPETYPE.INSERTION:
            insertions.append(word_b)
        elif ope == OPETYPE.DELETION:
            deletions.append(word_a)
    return equals, substitutions, insertions, deletions

def unmatching_ngrams_by_RatcliffAndOberhelp(s_a, s_b):
    """
    Reference: https://stackoverflow.com/questions/22163829/text-alignment-extracting-matching-sequence-using-python
    """
    words_a = s_a.split()
    words_b = s_b.split()
    matcher = difflib.SequenceMatcher(a=words_a, b=words_b)
    matching_blocks = matcher.get_matching_blocks()
    unmatching_ngrams = []
    for i in range(0, len(matching_blocks)-1):
        block_i = matching_blocks[i]
        block_j = matching_blocks[i+1]
        unmatching_start_in_a = block_i.a + block_i.size
        unmatching_end_in_a = block_j.a
        unmatching_start_in_b = block_i.b + block_i.size
        unmatching_end_in_b = block_j.b
        unmatching_ngrams.append((
            ' '.join(words_a[unmatching_start_in_a:unmatching_end_in_a]),
            ' '.join(words_b[unmatching_start_in_b:unmatching_end_in_b])
        ))
    return unmatching_ngrams

def unmatching_ngrams_by_LevenshteinDistance(s_a, s_b):
    raise NotImplementedError