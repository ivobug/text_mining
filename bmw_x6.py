# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 20:29:44 2023

@author: Ivan
"""

import nltk.corpus  
from nltk.text import Text, sent_tokenize
from nltk import word_tokenize, pos_tag, RegexpParser
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import WordNetLemmatizer
from nltk.probability import FreqDist
from collections import Counter
from nltk.collocations import BigramAssocMeasures, TrigramAssocMeasures, TrigramCollocationFinder, BigramCollocationFinder
from nltk.corpus import gutenberg
import re


#nltk.download()

# Učitavanje teksta iz txt file-a!
text = open("2009/2009_bmw_x6", "r", encoding = "UTF-8")
raw = text.read()



# as per recommendation from @freylis, compile once only
CLEANR = re.compile('<.*?>') 

def cleanhtml(raw_html):
  cleantext = re.sub(CLEANR, '', raw_html)
  return cleantext

raw=cleanhtml(raw)

print(raw)

number_of_sentences = sent_tokenize(raw)
print('Ukupno znakova u sirovom tekstu: ',len(raw))
print('Ukupno recenica u tekstu: ',len(number_of_sentences))



#Tokeniziranje danog teksta!
tokens = word_tokenize(raw)
#print(tokens)
print('Ukupan broj tokena za brojeve i riječi: ', len(tokens))

#Normalizacija teskta
#Ostaviti riječi koje imaju samo abecedne znakove
tokens = [word for word in tokens if word.isalpha()]
print("broj riječi:", len(tokens))
#Prebacivanje teksta u mala slova
tokens = [word.lower() for word in tokens]
print("\n", tokens)


# Micanje Stop word-ova iz text-a
stop_words = stopwords.words("english")
tokens = [word for word in tokens if not word in stop_words]

print("broj riječi:", len(tokens))

print("\n", tokens)




# Micanje Stop word-ova iz text-a
stop_words = stopwords.words("english")
tokens = [word for word in tokens if not word in stop_words]

print("broj riječi:", len(tokens))

print("\n", tokens)


#Stemming teksta 
porterStemmer = PorterStemmer()

for token in tokens: 
    print(token, ":", porterStemmer.stem(token)) 




#Lematizacija teksta
lemmatization = WordNetLemmatizer()

for token in tokens: 
    print(token, ":", lemmatization.lemmatize(token))


#Izračunati frekvencije riječi u listi i prikazati njihov graf
fdist = FreqDist(tokens)

#broj jedinstvenih riječi
print(len(fdist))

fdist.most_common(20)

fdist.plot(10)



most=[]

# Najčešće riječi
word_count = Counter(tokens)
for i in word_count.most_common(20):
    most.append(i[0])
    #print('Pojavnost 20 najčešćih riječi u tekstu:\n\n', alo)
print(most)



#Prikazati concordance za 10 najfrekventnije riječi 
textList = Text(tokens)

textList.concordance('steve')
textList.dispersion_plot(['car', 'like', 'drive', 'handling', 'one', 'bmw', 'sport', 'suv', 'get', 'love', 'great', 'road', 'driving', 'fun', 'better', 'cars', 'looks', 'engine', 'best', 'two'])


#Prikazati kolokacije
#Bigram
bigram_measures = BigramAssocMeasures()
finderBigram = BigramCollocationFinder.from_words(tokens)
finderBigram.nbest(bigram_measures.pmi, 20)


trigram_measures = TrigramAssocMeasures()
finder = TrigramCollocationFinder.from_words(tokens)
finder.nbest(trigram_measures.raw_freq, 10)


#Izračunati leksički diverzificitet 
lexical_diversity = len(set(tokens)) / len(tokens)
print(lexical_diversity)


#Primjena korpusa teksta: Gutenberg
gutenberg.fileids()


for fileid in gutenberg.fileids():
        num_chars = len(gutenberg.raw(fileid))
        num_words = len(gutenberg.words(fileid))
        num_sents = len(gutenberg.sents(fileid))
        print(round(num_chars/num_words), round(num_words/num_sents), fileid)
        
        
melville = Text(gutenberg.words('melville-moby_dick.txt'))
len(melville)

melville.concordance("love")


#Tagirati riječi i prikazati lingvističko stablo na temelju vlastitog uzorka teksta

#Tokenizacija teksta po rečenicama
custom_tokenizer = PunktSentenceTokenizer(raw)

tokenized = custom_tokenizer.tokenize(raw)
print(tokenized[:5])

#RB - prilog, VB - glagol, NNP - vlastita imenica,  NN - imenica
for i in tokenized[:5]:
    words = word_tokenize(i)
    words = [word for word in words if word.isalpha()]
    tagged = pos_tag(words)
    chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
    chunkParser = RegexpParser(chunkGram)
    chunked = chunkParser.parse(tagged)
    chunked.draw()
    print(chunked)
