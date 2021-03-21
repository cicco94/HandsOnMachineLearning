from nltk import DefaultTagger, UnigramTagger, BigramTagger
from nltk.corpus import treebank

train_set = treebank.tagged_sents()[:3000]
test_set = treebank.tagged_sents()[3000:]

bitagger = UnigramTagger(train_set)
print(bitagger.evaluate(test_set)) # quanto è buono da 0 a 1?
print(bitagger.tag("I love Alessia too much her since years".split())) # lo provo su una frase che non conosce nessuno

# domanda: e se voglio utilizzare un custom train/test_set invece che quelli del treebank?
# soluzione: devo crearmelo io e al solito splittare in train & test
custom_set = [
    [('Pierre', 'NNP'), ('Vinken', 'NNP'), (',', ','), ('61', 'CD'), ('years', 'NNS'), ('old', 'JJ'), (',', ','), ('will', 'MD'), ('join', 'VB'), ('the', 'DT'), ('board', 'NN'), ('as', 'IN'), ('a', 'DT'), ('nonexecutive', 'JJ'), ('director', 'NN'), ('Nov.', 'NNP'), ('29', 'CD'), ('.', '.')],
    [('Mr.', 'NNP'), ('Vinken', 'NNP'), ('is', 'VBZ'), ('chairman', 'NN'), ('of', 'IN'), ('Elsevier', 'NNP'), ('N.V.', 'NNP'), (',', ','), ('the', 'DT'), ('Dutch', 'NNP'), ('publishing', 'VBG'), ('group', 'NN'), ('.', '.')],
    [('Rudolph', 'NNP'), ('Agnew', 'NNP'), (',', ','), ('55', 'CD'), ('years', 'NNS'), ('old', 'JJ'), ('and', 'CC'), ('former', 'JJ'), ('chairman', 'NN'), ('of', 'IN'), ('Consolidated', 'NNP'), ('Gold', 'NNP'), ('Fields', 'NNP'), ('PLC', 'NNP'), (',', ','), ('was', 'VBD'), ('named', 'VBN'), ('*-1', '-NONE-'), ('a', 'DT'), ('nonexecutive', 'JJ'), ('director', 'NN'), ('of', 'IN'), ('this', 'DT'), ('British', 'JJ'), ('industrial', 'JJ'), ('conglomerate', 'NN'), ('.', '.')],
    [('A', 'DT'), ('form', 'NN'), ('of', 'IN'), ('asbestos', 'NN'), ('once', 'RB'), ('used', 'VBN'), ('*', '-NONE-'), ('*', '-NONE-'), ('to', 'TO'), ('make', 'VB'), ('Kent', 'NNP'), ('cigarette', 'NN'), ('filters', 'NNS'), ('has', 'VBZ'), ('caused', 'VBN'), ('a', 'DT'), ('high', 'JJ'), ('percentage', 'NN'), ('of', 'IN'), ('cancer', 'NN'), ('deaths', 'NNS'), ('among', 'IN'), ('a', 'DT'), ('group', 'NN'), ('of', 'IN'), ('workers', 'NNS'), ('exposed', 'VBN'), ('*', '-NONE-'), ('to', 'TO'), ('it', 'PRP'), ('more', 'RBR'), ('than', 'IN'), ('30', 'CD'), ('years', 'NNS'), ('ago', 'IN'), (',', ','), ('researchers', 'NNS'), ('reported', 'VBD'), ('0', '-NONE-'), ('*T*-1', '-NONE-'), ('.', '.')],
    [('The', 'DT'), ('asbestos', 'NN'), ('fiber', 'NN'), (',', ','), ('crocidolite', 'NN'), (',', ','), ('is', 'VBZ'), ('unusually', 'RB'), ('resilient', 'JJ'), ('once', 'IN'), ('it', 'PRP'), ('enters', 'VBZ'), ('the', 'DT'), ('lungs', 'NNS'), (',', ','), ('with', 'IN'), ('even', 'RB'), ('brief', 'JJ'), ('exposures', 'NNS'), ('to', 'TO'), ('it', 'PRP'), ('causing', 'VBG'), ('symptoms', 'NNS'), ('that', 'WDT'), ('*T*-1', '-NONE-'), ('show', 'VBP'), ('up', 'RP'), ('decades', 'NNS'), ('later', 'JJ'), (',', ','), ('researchers', 'NNS'), ('said', 'VBD'), ('0', '-NONE-'), ('*T*-2', '-NONE-'), ('.', '.')],
    [('Lorillard', 'NNP'), ('Inc.', 'NNP'), (',', ','), ('the', 'DT'), ('unit', 'NN'), ('of', 'IN'), ('New', 'JJ'), ('York-based', 'JJ'), ('Loews', 'NNP'), ('Corp.', 'NNP'), ('that', 'WDT'), ('*T*-2', '-NONE-'), ('makes', 'VBZ'), ('Kent', 'NNP'), ('cigarettes', 'NNS'), (',', ','), ('stopped', 'VBD'), ('using', 'VBG'), ('crocidolite', 'NN'), ('in', 'IN'), ('its', 'PRP$'), ('Micronite', 'NN'), ('cigarette', 'NN'), ('filters', 'NNS'), ('in', 'IN'), ('1956', 'CD'), ('.', '.')],
    [('Although', 'IN'), ('preliminary', 'JJ'), ('findings', 'NNS'), ('were', 'VBD'), ('reported', 'VBN'), ('*-2', '-NONE-'), ('more', 'RBR'), ('than', 'IN'), ('a', 'DT'), ('year', 'NN'), ('ago', 'IN'), (',', ','), ('the', 'DT'), ('latest', 'JJS'), ('results', 'NNS'), ('appear', 'VBP'), ('in', 'IN'), ('today', 'NN'), ("'s", 'POS'), ('New', 'NNP'), ('England', 'NNP'), ('Journal', 'NNP'), ('of', 'IN'), ('Medicine', 'NNP'), (',', ','), ('a', 'DT'), ('forum', 'NN'), ('likely', 'JJ'), ('*', '-NONE-'), ('to', 'TO'), ('bring', 'VB'), ('new', 'JJ'), ('attention', 'NN'), ('to', 'TO'), ('the', 'DT'), ('problem', 'NN'), ('.', '.')],
    [('A', 'DT'), ('Lorillard', 'NNP'), ('spokewoman', 'NN'), ('said', 'VBD'), (',', ','), ('``', '``'), ('This', 'DT'), ('is', 'VBZ'), ('an', 'DT'), ('old', 'JJ'), ('story', 'NN'), ('.', '.')],
    [('We', 'PRP'), ("'re", 'VBP'), ('talking', 'VBG'), ('about', 'IN'), ('years', 'NNS'), ('ago', 'IN'), ('before', 'IN'), ('anyone', 'NN'), ('heard', 'VBD'), ('of', 'IN'), ('asbestos', 'NN'), ('having', 'VBG'), ('any', 'DT'), ('questionable', 'JJ'), ('properties', 'NNS'), ('.', '.')],
    [('There', 'EX'), ('is', 'VBZ'), ('no', 'DT'), ('asbestos', 'NN'), ('in', 'IN'), ('our', 'PRP$'), ('products', 'NNS'), ('now', 'RB'), ('.', '.'), ("''", "''")]
]
# solo 10 frasi, verrà un risultato molto scarso perchè ho pochi dati con cui posso istruire il mio classificatore

# adesso faccio le stesse identiche cose che ho fatto prima ma col mio classificatore
custom_train_set =custom_set[:7]
custom_test_set =custom_set[7:]
custom_bitagger = UnigramTagger(custom_train_set)
print(custom_bitagger.evaluate(custom_test_set)) # quanto è buono da 0 a 1?
print(custom_bitagger.tag("I love Alessia too much her since years".split())) # lo provo su una frase che non conosce nessuno

# il mio custom_bitagger ha indovinato solo "years" perchè era l'unica parola presente nel mio custom_train_set