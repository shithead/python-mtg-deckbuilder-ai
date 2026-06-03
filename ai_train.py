from ai.MTGDeckBuilderModel import MTGDeckBuilderModel
from ai.Trainer import Trainer_T1

trainer = Trainer_T1(MTGDeckBuilderModel)
trainer.run(epochs=10)
