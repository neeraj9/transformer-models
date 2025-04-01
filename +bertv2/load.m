function params = load(modelName)
% load   Load BERT parameters
%
%   parameters = load(modelName) will load the model weights associated to
%   modelName and the hyperparameters associated to that model.

% Copyright 2021 The MathWorks, Inc.
arguments
    modelName (1,1) string
end

% the URLs don't have bertv2- at the start.
paramsStructFile = bertv2.internal.getSupportFilePath(modelName,"parameters.mat");
paramsStruct = load(paramsStructFile);

params = struct(...
    'Hyperparameters',paramsStruct.Hyperparameters,...
    'Weights',bertv2.internal.createParameterStruct(paramsStruct.Weights));
end