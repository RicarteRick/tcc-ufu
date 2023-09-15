from pre_processamento import preprocess_documents, tokenize_file
from util_files import openFile, saveJsonFile

docsPath = 'docs/'
rawFileName = 'base_de_treino.json'
preprocessedFileName = 'base_de_treino_pre_processada.json'
tokenizedRawFileName = 'base_de_treino_raw_tokenizada.json'

#region Pre-processamento
# carregando arquivo
rawData = openFile(docsPath+rawFileName)

# pré-processando
preprocessedData = preprocess_documents(rawData)

# salvando novo arquivo pré-processado
saveJsonFile(docsPath+preprocessedFileName, preprocessedData)

# salvando arquivo sem pré-processamento, mas tokenizado
rawData2 = openFile(docsPath+rawFileName)
tokenizedRawData = tokenize_file(rawData2)
saveJsonFile(docsPath+tokenizedRawFileName, tokenizedRawData)

print('Documentos salvos')
#endregion
