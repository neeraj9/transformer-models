function z = classifierHead(x,poolerWeights,classifierWeights)
% classifierHead   The standard classification head for a BERT model.
% 
%   Z = classifierHead(X,poolerWeights,classifierWeights) applies
%   bertv2.layer.pooler and bertv2.layer.classifier to X with poolerWeights and
%   classifierWeights respectively. Both poolerWeights and
%   classifierWeights must be structs with fields 'kernel' and 'bias'.

% Copyright 2021 The MathWorks, Inc.
z = bertv2.layer.pooler(x,poolerWeights);
z = bertv2.layer.classifier(z,classifierWeights);
end