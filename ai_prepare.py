from ai.MTGDeckBuilderModel import MTGDeckBuilderModel
from ai.Preparer import Preparer

preparer = Preparer(MTGDeckBuilderModel)
#print(trainer_t1.vocab)
#print(trainer_t1.word2idx)
#print(trainer_t1.embeddings)
preparer.save()
