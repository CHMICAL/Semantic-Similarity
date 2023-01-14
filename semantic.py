import spacy
nlp = spacy.load('en_core_web_md')
word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")
print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))


tokens = nlp('cat apple monkey banana')

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

tokens = nlp('boat bridge relationship ship')

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

# As mentioned in the task document, monkey and banana are more alike compared to monkey and apple. This shows us that the model seems to recognise
# the common trope of monkeys eating bananas. Monkey and cat are 59% alike in semantic similarity, presumably because they are both animals.

# With my own examples, it is interesting that 'boat' and 'bridge' have 45% similarity. This is presumably because boats commonly encounter bridges when travelling.
# It is also interesting that 'relationship' and 'bridge' have a 24% similarity score, which is fairly similar for two words that seem
# disimilar. This could be due to it being common to view relationships as bridges.

#The limitation of the smaller spacy model (en_core_web_sm), is that it does not have word vectors loaded. This doesn't give as accurate of a semantic similarity 
# calculation, compared to the larger spacy model.

# Error when using sm version:The model you're using has no word vectors loaded, so the result of the Doc.similarity method will be based on the tagger, parser and NER,
# which may not give useful similarity judgements. This may happen if you're using one of the small models, e.g. `en_core_web_sm`, which don't ship with word vectors 
# and only use context-sensitive tensors. You can always add your own word vectors, or use one of the larger models instead if available.
# print(token.similarity(token_))