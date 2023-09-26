from pre_processamento import preprocess
from classificadores import execute_classifiers

# Paths
docsPath = 'docs/'
rawFileName = 'base_de_treino.json'
preprocessedFileName = 'base_de_treino_pre_processada.json'
tokenizedRawFileName = 'base_de_treino_raw_tokenizada.json'

# Pre-processamento
# preprocess(docsPath, rawFileName, preprocessedFileName, tokenizedRawFileName)

# Classificadores
execute_classifiers(docsPath+preprocessedFileName)