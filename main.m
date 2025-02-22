clear;clc;close all;

%% TOOLBOX EKLEME

addpath('miditoolbox')

%% MIDI DOSYALARI YÜKLEME (DATASET ÇALIŞMIYORSA ÇALIŞTIRIN UZUN SÜRÜYOR)

input_folder = 'GiantMIDI-PIano\midis_v1.2\midis';
midi_files = dir(fullfile(input_folder, '*.mid'));
for k = 1:length(midi_files)
    midi_file_path = fullfile(midi_files(k).folder, midi_files(k).name);
    nmat = readmidi(midi_file_path);
   disp(k)
end

%%
%%DATASETİ EKLEME

load mididata.mat;

%% TÜM MODELLERİN TRAIN EDİLMİŞ HALİNİ EKLEMEK İÇİN

load final.mat
%% KISA OLAN MIDI DOSYALARINI AYIKLAMA

nmatcellnew = {};
idx = 1;
for k = 1:size(nmatcell,2)
    if(size(nmatcell{k},1) >= 500)
        nmatcellnew{idx} = nmatcell{k};
        idx = idx + 1;
    end
end
templength = numel(nmatcellnew);
sizes = zeros(length(nmatcellnew),1);
%% MIDI DOSYALARININ İLK 500 TIMESTEPLERINI ALMA

maxsize = 500;
for i = 1:templength
    %%nmatcellnew{i} = quantize(nmatcellnew{i},1/64,1/64,1/64);
    nmatcellnew{i} = quantize(nmatcellnew{i},1/32,1/32,1/32);
end
nmatcellnormalize = {};
for i = 1:templength
    if(length(nmatcellnew{i}) < maxsize)
        nmattempnorm = zeros(maxsize,7);
        nmatnorm = zeros(maxsize,7);
        nmattemp = nmatcellnew{i};
        for j = 1:length(nmatcellnew{i})
            nmatnorm(j,:) = nmattemp(j,:) + nmattempnorm(j,:);
        end
    else
    nmattemp = nmatcellnew{i};
    nmatnorm = nmattemp(1:maxsize,:);
    end
    nmatcellnormalize{i} = nmatnorm;
end

%% OUTLIER OLAN MIDILERI AYIKLAMA

nmatcellsame = {};
idx = 1;
for k = 1:size(nmatcellnormalize,2)
    if(nmatcellnormalize{k}(1) <= 8  && (nmatcellnormalize{k}(end,1) <= 100  && nmatcellnormalize{k}(end,1) >= 80))
        nmatcellsame{idx} = nmatcellnormalize{k};
        idx = idx + 1;
    end
end
lengthnmat = numel(nmatcellsame);

%% TÜM ŞARKILARI AYNI AKORA GÖRE NORMALİZE ETME

for i = 1:lengthnmat
    nmatcellstd{i} = transpose2c(nmatcellsame{i});
end

%% NORMALİZASYON

for i = 1:lengthnmat
    nmatcellstd{i}(:,4) = nmatcellstd{i}(:,4)/103;  %PITCH
    nmatcellstd{i}(:,1) = nmatcellstd{i}(:,1)/100; %ONSET
    nmatcellstd{i}(:,2) = nmatcellstd{i}(:,2)/12;  %DURATION
end


%% FEATURE EXTRACTION

onsets = {};
durations = {};
pitch = {};
for i= 1:lengthnmat
    onsets{i} = nmatcellstd{i}(:,1);
    durations{i} = nmatcellstd{i}(:,2);   %% GEREKLİ OLAN FEATURELARI ALIYOR
    pitch{i} = nmatcellstd{i}(:,4);
end
features = {};
for i= 1:lengthnmat
    feature(:,1) = onsets{i};
    feature(:,2) = durations{i};
    feature(:,3) = pitch{i};
    features{i} = feature;
end
%%
sequenceLength = 50; % Length of the input sequence
numFeatures = 3;     % Onset, Duration, Pitch, Velocity
%% TRAININ INPUT VE OUTPUTLARI AYARLAMA SEKANS

X = {};
Y = {};

for i = 1:length(features)
    song = features{i};
    for j = 1:(size(song, 1) - 2*sequenceLength)
        X{end+1} = song(j:j+sequenceLength-1,:); % 50x4
        Y{end+1} = song(j+sequenceLength:j+2*sequenceLength-1,:);   % 1x4 %% SEKANSLARA AYIRMA
    end
end

X = X';
Y = Y';
%% TRAININ INPUT VE OUTPUTLARI AYARLAMA TEK TEK

X = {};
Y = {};

for i = 1:length(features)
    song = features{i};
    for j = 1:(size(song, 1) - sequenceLength)
        X{end+1} = song(j:j+sequenceLength-1,:); % 50x4
        Y{end+1} = song(j+1:j+sequenceLength,:);   % 1x4 %% SEKANSLARA AYIRMA
    end
end

X = X';
Y = Y';
%% TRAIN TEST VE VALIDATION AYIRMA

XTrain = {};
YTrain = {};
XVal = {};
YVal = {};
XTest = {};
YTest = {};

numSamples = numel(X);
numTrain = floor(0.9 * numSamples);
numTest = floor(0.075 * numSamples);
numValidation = numSamples - (numTrain + numTest);

shuffledIdx = randperm(numSamples);

XTrain = X(shuffledIdx(1:numTrain));
YTrain = Y(shuffledIdx(1:numTrain));
XTest = X(shuffledIdx(numTrain+1:numTrain+numTest));
YTest = Y(shuffledIdx(numTrain+1:numTrain+numTest));
XVal = X(shuffledIdx(numTrain+numTest+1:end));
YVal = Y(shuffledIdx(numTrain+numTest+1:end));

%% NETWORK TANIMLARI
layers = [
    sequenceInputLayer(3);
    bilstmLayer(512, 'OutputMode', 'sequence');   
    dropoutLayer(0.3);
    fullyConnectedLayer(3);
];
%%

options = trainingOptions('adam', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 200, ...
    'InitialLearnRate', 0.0001, ...
    'Plots', 'training-progress', ...
    'ValidationData', {XVal, YVal}, ...
    'ValidationFrequency', 200, ...
    'Verbose', true);


%%
[binet50_2,info_binet50_2] = trainnet(XTrain,YTrain,layers,"mse",options);

%% SEED SEKANS AYARLAMA

seedSequence = [];
seedSequence = features{3};  %% 1 3 10
seedSequence = seedSequence(1:sequenceLength, :);
generatedSequence = seedSequence;
%%  TEK TEK NOTA ALMAK İÇİN
for i = 1:150 % Generate 150 notes
    nextNote = predict(dunet50_2, seedSequence);
    generatedSequence = [generatedSequence; nextNote(end,:)];
    seedSequence = [seedSequence(2:end,:); nextNote(end,:)]; 
    disp(i)
end
%% SEKANS ALMAK İÇİN
for i = 1:4
    nextSequence = predict(dubinet50_1, seedSequence);
    generatedSequence = [generatedSequence; nextSequence];
    seedSequence = nextSequence;
    disp(i)
end
%% MIDI FORMATINA DÖNÜŞTÜREBİLMEK İÇİN DÜZENLEME

generatedSequence(:,5) = 60;
generatedSequence(:,4) = generatedSequence(:,3)*103;
generatedSequence(:,1) = generatedSequence(:,1)*100;
generatedSequence(:,2) = generatedSequence(:,2)*12;
generatedSequence(:,3) = 0;

for i= 1:size(generatedSequence,1)
    generatedSequence(i,5) = round(generatedSequence(i,5),0);
    generatedSequence(i,4) = round(generatedSequence(i,4),0);
end

generatedSequence(:,6) = generatedSequence(:,1)./2;
generatedSequence(:,7) = generatedSequence(:,2)./2;

generatedSequence = quantize(generatedSequence,1/32,1/32,1/32);
%%
playsound(generatedSequence);
%%
writemidi(generatedSequence,"dunet50_2_3.mid",120);
%%
info_binet50_2.TrainingHistory


%%

info_binet50_2.ValidationHistory