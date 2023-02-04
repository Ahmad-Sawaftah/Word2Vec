from gensim.models.keyedvectors import KeyedVectors
"""
word_vectors=KeyedVectors.load_word2vec_format(
    'model/GoogleNews-vectors-negative300.bin',
    binary=True,
    limit=200000
)

word_vectors.most_similar(positive=['uncle','women'],negative=['man'],topn=1)
"""

from gensim.model.word2vec import Word2Vec
token_list=['This','is','an','example']
model=Word2Vec(
    token_list,
    workers=2,
    size=300,
    min_cout=3,
    window=6,
    sample=1e-3
)

model.init_sims(replace=True)

model.save("my_example_word2vec")
