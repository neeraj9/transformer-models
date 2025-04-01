function mdl = bertv2(nvp)
% bertv2   Pretrained BERT transformer model
%   mdl = bertv2 loads a pretrained BERT-Base model and downloads the model
%   weights and vocab file if necessary.
%
%   mdl = bertv2('Model', modelName) loads the BERT model specified by
%   modelName.  Supported values for modelName are "base" (default),
%   "multilingual-cased","medium","small","mini", "tiny", "japanese-base", 
%   and "japanese-base-wwm"

% Copyright 2021-2023 The MathWorks, Inc.
arguments
    nvp.Model (1,1) string {mustBeMember(nvp.Model,[
        "base"
        "multilingual-cased"
        "medium"
        "small"
        "mini"
        "tiny"
        "japanese-base"
        "japanese-base-wwm"])} = "base"
end

switch nvp.Model
    case "japanese-base"
        mdl = iJapaneseBERTModel("japanese-base", "bert-base-japanese.zip");
    case "japanese-base-wwm"
        mdl = iJapaneseBERTModel("japanese-base-wwm", "bert-base-japanese-whole-word-masking.zip");
    otherwise
        % Download the license file
        bertv2.internal.getSupportFilePath(nvp.Model,"bert.RIGHTS");
        params = bertv2.load(nvp.Model);
        % Get the IgnoreCase hyperparameter, then remove it, downstream code
        % shouldn't need it.
        ignoreCase = params.Hyperparameters.IgnoreCase;
        % Get vocab file
        vocabFile = bertv2.internal.getSupportFilePath(nvp.Model,"vocab.txt");
        params.Hyperparameters = rmfield(params.Hyperparameters,'IgnoreCase');
        mdl = struct(...
            'Tokenizer',bertv2.tokenizer.BERTTokenizer(vocabFile,'IgnoreCase',ignoreCase),...
            'Parameters',params);
end
end

function mdl = iJapaneseBERTModel(modelName, zipFileName)
zipFilePath = bertv2.internal.getSupportFilePath(modelName, zipFileName);
modelDir = fullfile(fileparts(zipFilePath), replace(zipFileName, ".zip", ""));      
unzip(zipFilePath, modelDir);
% Build the tokenizer
btok = bertv2.tokenizer.internal.TokenizedDocumentTokenizer("Language","ja","TokenizeMethod","mecab",IgnoreCase=false);
vocabFile = fullfile(modelDir, "vocab.txt");
ftok = bertv2.tokenizer.internal.FullTokenizer(vocabFile,BasicTokenizer=btok);
tok = bertv2.tokenizer.BERTTokenizer(vocabFile,FullTokenizer=ftok);
% Build the model
params.Weights = load(fullfile(modelDir, "weights.mat"));
params.Weights = dlupdate(@dlarray,params.Weights);
params.Hyperparameters = struct(...
    NumHeads=12,...
    NumLayers=12,...
    NumContext=512,...
    HiddenSize=768);
mdl = struct(...
    Tokenizer=tok,...
    Parameters=params);
end