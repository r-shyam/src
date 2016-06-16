% JA_HCRF_SVM.m - Predict child's engagement level in adult-child
% interaction. Execute this program by setting the working directory to
% where this file exists. It is assumed that 'data' and 'output'
% directories are present in the current folder. 

% This program computes the output from Stage-2 referenced in the following paper:
% Shyam Sundar Rajagopalan, O.V. Ramana Murthy, R. Goecke and Agata Rozga, Play with
% Me Measuring a Childs Engagement in a Social Interaction. Proceedings of the IEEE International
% Conference on Automatic Face and Gesture Recognition (FG 2015), Ljubljana, Slovenia, 4-8 May 2015
%
% Dataset: Multimodal Dyadic Behavior Dataset
% (http://www.cbi.gatech.edu/mmdb/#)
%
% Tools: Support Vector Machines (SVM)
%
% Input: Hidden state probabilities for each class, behaviour word and the
% hidden state. A cell array of this 3-dim input for each video. This
% sample program assumes that this matrix is pre-computed and stored in the
% data folder.
%
% Output: Child's engagement level recognition accuracy.
%
% Dependencies: LIBSVM
%
% Description: Feature vector is constructed using hidden state marginals
% and SVM is used to train the model.
%
% Author: Shyam Sundar Rajagopalan, Shyam.Rajagopalan@canberra.edu.au
function JA_HCRF_SVM()
clearvars ; fclose all ; close all ;

addpath(genpath('./libsvm-3.17/libsvm-3.17/windows/')) ;
strRoot = '.';
nVideos = 59 ; 
nStates = 4;  % hidden states used in stage-1
NUMCENTERS = 6 ; % behaviour words used in stage-1
strFeatName = 'stip' ; % feature type used in stage-1
randstate = 500 ; % random seed used in stage-1

% Load the ground truth labels
strOutFileName = sprintf('%d_R%d',NUMCENTERS,randstate) ;
strLabF = sprintf('vLabels_%s_C%d_R%d.mat',strFeatName,NUMCENTERS,randstate) ;
strLabelsPath = [strRoot '/data/' strLabF];
load (strLabelsPath,'vLabels') ;
for i = 1 : length(vLabels)
    vGTLabels(i) = vLabels{i}(2) ; % book label is 2
end

 % Leave-One-Out-Cross-Validation
for i = 1 : nVideos
    % It is assumed that these files are generated from Stage-1. For this
    % sample program, they are copied in the 'data' folder.
    strVHSFileTrain = sprintf('vHS_Prob_%s_%d_C%s_%d_Train.mat',strFeatName,nStates,strOutFileName,i) ; % Video, GT Label, Feats
    load([strRoot '/data/' strVHSFileTrain],'vHS_Prob') ;
    vTrainVideoNos = setdiff(1:nVideos,i) ;
    vLabels = vGTLabels(vTrainVideoNos) ;
    % This function forms a new feature vector using the method described
    % in the paper.
    features_scaled_file = ScaleFeatMat(vHS_Prob,1,vLabels') ;
    
    
    % Read features and labels from scaled file
    [vTrainLabels, vTrainFeats] = libsvmread(features_scaled_file);
    
   % Train model - this can be replaced with a grid search
    bestc = 0.005 ; bestg  = 0.005;
    params = sprintf('-c %2.4f -g %2.4f -b 1 -q ', bestc, bestg) ; % -t 0 -- linear svm
    model = svmtrain(vTrainLabels, vTrainFeats, params); % -v 5 is 5 fold cross validation % c 2 g 0.07
    
    % Test
    vHS_Prob = [] ;
    strVHSFileTest = sprintf('vHS_Prob_%s_%d_C%s_%d_Test.mat',strFeatName,nStates,strOutFileName,i) ; % Video, GT Label, Feats
    load([strRoot '/data/' strVHSFileTest],'vHS_Prob') ;
    vTestVideoNos = i ;
    vLabels = vGTLabels(vTestVideoNos) ;
    features_scaled_file = ScaleFeatMat(vHS_Prob,0,vLabels') ;

    % Read features and labels from scaled file
    [vTestLabels, vTestFeats] = libsvmread(features_scaled_file);

    % Predict using the test data
    params = '-b 1' ;
    [predict_label, accuracy, dec_values] = svmpredict(vTestLabels, vTestFeats, model, params); % test the training data
 
    
    fprintf('Instance %d : True = %d, Predicted = %d \n',i, vTestLabels(1), predict_label) ;
    
    if (predict_label == vTestLabels(1))
        acc(i) = 100 ;
    else
        acc(i) = 0 ;
    end

    
end
fprintf ('Mean Accuracy = %2.1f%%\n', mean(acc)) ;
disp ('End') ;

function features_scaled_file = ScaleFeatMat(vHS_Prob,bTrain,gtlabels) 
    % Compute features
     features = [] ;
     for r = 1 : size(vHS_Prob,2)
        row = vHS_Prob{r} ; 
        feat =[] ;
        % contact vertically - every state across all frames
        for s = 1 : size(row,3)
          % Working Version
          col_sum = 0 ;
          for c = 1 : size(row,1)
            col_dist = row(c,:,s)' ;
            col_sum = col_sum + sum(col_dist)  ; % Computing marginal value for each state
          end
          feat = [feat ; col_sum] ;
        end
        features = [features ;[(feat') ] ] ;
     end
    
    % Create sparse representation of features
    features_sparse = sparse(features); % features must be in a sparse matrix
    outfile = './output/features.sparse' ;
    libsvmwrite(outfile, gtlabels, features_sparse);

    % Scale the features from sparse representation
    features_sparse_file = './output/features.sparse';
    features_scaled_file = './output/features.scale' ;
    scaling_parameters = './output/scaling_params.params' ;
    if (bTrain)
        cmd = sprintf ('svm-scale -s %s %s > %s', scaling_parameters, features_sparse_file, features_scaled_file) ;
    else
        cmd = sprintf ('svm-scale -r %s %s > %s', scaling_parameters, features_sparse_file, features_scaled_file) ;
    end
    system (cmd) ;

