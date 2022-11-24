% make simulated data for decoding
% Author: Zhibing Xiao,2022/11/24

clear

sampleSize   = 200 ; % number of trials
NumOfChannel = 62  ;
NumOfPoints  = 256 ; % number of points per trial
NumOfClass   = 4   ;
TargetRang   = [0.3 0.6]  ;
 
signalRate   = 0.2;
nNullExamples=0;




tarPointRang = round( NumOfPoints* TargetRang);
numPerStim = sampleSize/NumOfClass;
point_time = linspace(-100,1000,NumOfPoints);

%% add noise 
signalRateSeq = zeros(NumOfPoints,1);
signalRateSeq(tarPointRang(1):tarPointRang(2))= signalRate;
% signalRateSeq = signalRateSeq+ rand(NumOfPoints,1)/2;
signalRateSeq = smooth(signalRateSeq,5);
subplot(2,2,1)
plot(signalRateSeq)
%%
realPattern = randn(NumOfClass,NumOfChannel);
allTrialData ={};
for it = 1:numPerStim
    for ic = 1:NumOfClass
        A = randn(NumOfChannel);
        [U,~] = eig((A+A')/2);
        covMat = U*diag(abs(randn(NumOfChannel,1)))*U';

        X(1,:) = randn([1 NumOfChannel]);
        for iT=2:NumOfPoints
            X(iT,:) = (~signalRateSeq(it))*0.10*(X(iT-1,:) + mvnrnd(zeros(1,NumOfChannel), covMat)) ;
            if signalRateSeq(iT)
                signal = realPattern(ic,:)*(signalRateSeq(iT));
                signal = signal/max(signal);
            else
                signal = realPattern(ic,:)*(signalRateSeq(iT));
            end
            X(iT,:) = X(iT,:)/max(X(iT,:)) +  signal + randn(1,NumOfChannel) ;% add dependence of the sensors
        end
        Xs(:,:,ic) = X';
    end
    allTrialData{it} =Xs;
end
allTrialData = cat(3,allTrialData{:});


%%
trainingLabels = [zeros(nNullExamples,1); repmat((1:NumOfClass)', [numPerStim 1])];


X = squeeze(allTrialData(:,:,1));
rg = max(X(:,1))-min(X(:,1));
subplot(222)
plot(point_time, X + repmat([1:rg:rg*NumOfChannel]',1,NumOfPoints))

subplot(223)
autocorr(X(1,:))

 

trainingData = squeeze(allTrialData(:,tarPointRang(1),:));

marker_type = trainingLabels;

[~, validationAccuracy] = CVtrainClassifier( squeeze(allTrialData(:,tarPointRang(1),:)),marker_type)

validationAccuracy =[];
for ii = 1:5:NumOfPoints
    [~, validationAccuracy(end+1)] = CVtrainClassifier( squeeze(allTrialData(:,ii,:)),marker_type);
end
subplot(224)
plot(point_time(1:5:end),validationAccuracy)
title decoding
% save sub00_test_data_varifyCode.mat stim_data marker_type point_time fixzation tarPointRang


