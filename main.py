from pre_processamento import preprocess
from classificadores import train

# Paths
docsPath = 'docs/'
rawFileName = 'base_de_treino.json'
preprocessedFileName = 'base_de_treino_pre_processada.json'
tokenizedRawFileName = 'base_de_treino_raw_tokenizada.json'

# Pre-processamento
preprocess(docsPath, rawFileName, preprocessedFileName, tokenizedRawFileName)

# Classificadores
# hold-out
test_size = 0.3 # 70-30

train(docsPath+preprocessedFileName, test_size)