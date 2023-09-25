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
# K-folds
k_folds = 3
train(docsPath+preprocessedFileName, k_folds)

k_folds = 5
train(docsPath+preprocessedFileName, k_folds)

k_folds = 10
train(docsPath+preprocessedFileName, k_folds)