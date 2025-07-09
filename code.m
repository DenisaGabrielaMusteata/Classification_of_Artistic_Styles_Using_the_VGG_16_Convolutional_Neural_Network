
%% CNN for image classification
%+++++++++++++++++++++++++++++++++++
 
clear all
close all
clc

%% PARAMETERS
% the name of the file where the trained CNN is saved
nameFileRez='rezCNN.mat';

% the size of the input images
inputSize = [224 224 3]; 

% training parameters
MBS=32;% mini batch size
NEP=5; % number of epochs

%% TRAINING AND VALIDATION DATASETS

% indicate the path to the trainingaand validation images
pathImages='C:\Users\muste\Desktop\DatasetArts4';
 
% create the datastore with the training and validation images
imds = imageDatastore(pathImages, ...  
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); 

%resize the images to be compatible with the network
resizeImagesImds(imds, 'C:\Users\muste\Desktop\DatasetArts4', inputSize(1:2));

% split the dataset into training and validation datasets
[imdsTrain, imdsTest] = splitEachLabel(imds,0.7,'randomized');
[imdsTest, imdsValidation] = splitEachLabel(imdsTest, 0.7, 'randomized');

% obtain information about the training dataset
numTrainImages = numel(imdsTrain); % the number of trainig images 
numClasses = numel(categories(imdsTrain.Labels)); %the number of classes  

%% DESIGN THE ARCHITECTURE

% load the pretrained model 
net = vgg16; 

% take the kayers for transfer of learning
layersTransfer = net.Layers(1:end-3); 

% create the new architecture: the last fully connected layer is configured for the necessary number of classes
layersNew = [
    layersTransfer    
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20) 
    softmaxLayer 
    classificationLayer];
 
%% TRAIN THE CNN

% indicate the training parameters
options = trainingOptions('adam', ...
    'MiniBatchSize',MBS,...            
    'MaxEpochs',NEP, ...      
    'InitialLearnRate',1e-4, ...  
    'LearnRateDropFactor', 1e-5,...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',3, ...
    'ValidationPatience',Inf, ...
    'Verbose',false, ...
    'Plots','training-progress');
                  
% train the model
netTransfer = trainNetwork(imdsTrain,layersNew,options);

% save the trained model
feval(@save,nameFileRez,'netTransfer'); 


%% VERIFY THE RESULTS 

% validation - responses and accuracy
[YPredValidation,scoresValidation] = classify(netTransfer,imdsValidation); 
accuracyValidation = mean(YPredValidation == imdsValidation.Labels)  

% training - responses and accuracy
[YPredTrain,scoresTrain] = classify(netTransfer,imdsTrain);  
accuracyTrain = mean(YPredTrain == imdsTrain.Labels)  

% testing- responses and accuracy
[YPredTest,scoresTest] = classify(netTransfer,imdsTest);  
accuracyTest = mean(YPredTest == imdsTest.Labels)  


%% Confussion matrix
  figure;
  confusionchart(imdsValidation.Labels, YPredValidation);
  title('Matricea de Confuzie - Set Validare');


  %% SECOND TRAIN
  clear all;
  close all;
  clc
nameFileRez='rezCNN2.mat';
%load the old network
load('rezCNN.mat', 'netTransfer');

inputSize = [224 224 3]; 

% training parameters
MBS=32;
NEP=5; 

%% TRAINING AND VALIDATION DATASETS

% indicate the path to the trainingaand validation images
pathImages='C:\Users\muste\Desktop\DatasetArts5';

% create the datastore with the training and validation images
imds = imageDatastore(pathImages, ...  
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); 

%resize the images to be compatible with the network
resizeImagesImds(imds, 'C:\Users\muste\Desktop\DatasetArts5', inputSize(1:2));

% split the dataset into training and validation datasets
[imdsTrain, imdsTest] = splitEachLabel(imds,0.7,'randomized');
[imdsTest, imdsValidation] = splitEachLabel(imdsTest, 0.7, 'randomized');

% obtain information about the training dataset
numTrainImages = numel(imdsTrain); % the number of trainig images 
numClasses = numel(categories(imdsTrain.Labels)); %the number of classes
    

%% DESIGN THE ARCHITECTURE

% load the pretrained model 
net = vgg16; 

% take the kayers for transfer of learning
layersTransfer = net.Layers(1:end-3); 

% create the new architecture: the last fully connected layer is configured for the necessary number of classes
layersNew = [
    layersTransfer    
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20) 
    softmaxLayer 
    classificationLayer];

% freeze the weights of the base layers
for i = 1:numel(layersTransfer)
    if isa(layersTransfer(i), 'nnet.cnn.layer.Convolution2DLayer') || ...
       isa(layersTransfer(i), 'nnet.cnn.layer.FullyConnectedLayer')
        layersTransfer(i).WeightLearnRateFactor = 0;  
        layersTransfer(i).BiasLearnRateFactor = 0;    
    end
end


layersNew = [
    layersTransfer
    fullyConnectedLayer(numClasses, 'WeightLearnRateFactor', 20, 'BiasLearnRateFactor', 20)
    softmaxLayer
    classificationLayer];

%% TRAIN THE CNN

% indicate the training parameters
options = trainingOptions('adam', ...
    'MiniBatchSize',MBS,...            
    'MaxEpochs',NEP, ...      
    'InitialLearnRate',1e-4, ...  
    'LearnRateDropFactor', 1e-5,...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',3, ...
    'ValidationPatience',Inf, ...
    'Verbose',false, ...
    'Plots','training-progress');
                  
% train the model
netTransfer = trainNetwork(imdsTrain,layersNew,options);

% save the trained model
save('rezCNN2.mat', 'netTransfer');

%% VERIFY THE RESULTS 

% validation - responses and accuracy
[YPredValidation,scoresValidation] = classify(netTransfer,imdsValidation); 
accuracyValidation = mean(YPredValidation == imdsValidation.Labels)  

% training - responses and accuracy
[YPredTrain,scoresTrain] = classify(netTransfer,imdsTrain);  
accuracyTrain = mean(YPredTrain == imdsTrain.Labels)  

% testing- responses and accuracy
[YPredTest,scoresTest] = classify(netTransfer,imdsTest);  
accuracyTest = mean(YPredTest == imdsTest.Labels)  

% Confussion Matrix
  figure;
  confusionchart(imdsValidation.Labels, YPredValidation);
  title('Matricea de Confuzie - Set Validare');

  %% tests on new network
  % Load the network
load('rezCNN2.mat', 'netTransfer');  %network from the last train

imagePath = "C:\Users\muste\Desktop\DatasetArts4\Ukiyo_e\hiroshige_chrysanthemums.jpg";
img = imread(imagePath);

imgResized = imresize(img, [224 224]);  

[label, score] = classify(netTransfer, imgResized);

disp(['Predicția rețelei: ', char(label)]);

figure;
imshow(img);
title(['Predicția: ', char(label)]);

  %% tests on old network

load('rezCNN.mat', 'netTransfer');  %network from the first train


imagePath = "C:\Users\muste\Desktop\DatasetArts4\Ukiyo_e\hiroshige_chrysanthemums.jpg";
img = imread(imagePath);

imgResized = imresize(img, [224 224]);  

[label, score] = classify(netTransfer, imgResized);

disp(['Predicția rețelei: ', char(label)]);

figure;
imshow(img);
title(['Predicția: ', char(label)]);


