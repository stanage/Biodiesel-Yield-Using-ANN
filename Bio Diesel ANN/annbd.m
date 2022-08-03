% This is a practice project working with experimental data gotten from a
% research paper' on Elsevier journal. In this project, I aimed to train a
% model to predict biodiesel yield given a set of input features;
% temperature, pressure, reaction time, and methanol/oil (M/O) ratio. I
% went about this by dividing my dataset (with 42 examples), into a training
% set, a validation set, and a test set in a ratio of 3:1:1. This project
% was done using MATLAB R2021a

% Source of data: Obie F, Nur H, Yukihiko M. Artificial neural network
% modeling to predict biodiesel production in supercritical methanol and
% ethanol using spiral reactor. 2015 p221



%=============INITIALIZATION=============================
clear; close all; clc

%===================LOAD DATA============================
data = load('dataset.txt');
X = data(:, 1:4); %This file contains 42 training examples with 4 features
y = data(:, 5); % This file contains 42 output values
m = length(y);

%==================VISUALIZING THE DATA==================
% yield
histogram(y,10); 
title('yield frequency');
xlabel('value');
ylabel('frequency');

figure;
% temperature
histogram(X(:, 1), 10);
title('Temperature frequency');
xlabel('value');
ylabel('frequency');

figure;
% pressure
histogram(X(:, 2), 10);
title('pressure frequency');
xlabel('value');
ylabel('frequency');

figure;
% reaction time
histogram(X(:, 3), 10);
title('reaction time frequency');
xlabel('value');
ylabel('frequency');

figure;
% M/O ratio
histogram(X(:, 4 ), 10);
title('M/O frequency');
xlabel('value');
ylabel('frequency');

%===================NORMALIZATION=========================
for i = 1:4
    X2(:, i) = (X(:, i)-min(X(:, i)))/(max(X(:, i))-min(X(:, i)));
end
X = X2;
y = log(1 + y);

%===================VISUALIZATION OF NORMALIZED DATA====================
figure;
% input - yield
Xaxis = [X(:, 1), X(:, 2), X(:, 3), X(:, 4)];
scatter(Xaxis, y, 'o');
title('Input-yield Visualization');
xlabel('input value');
ylabel('yield');
legend('Temperature', 'Pressure', 'Reaction Time', 'M/O Ratio'); 


%% 

%================= TRAINING THE ANN =====================
Xt = X';
yt = y';
hiddenLayerSize = 25;
net = fitnet(hiddenLayerSize);
net.divideParam.trainRatio = 60/100;
net.divideParam.valRatio = 20/100;
net.divideParam.testRatio = 20/100;
[net, tr] = train(net, Xt, yt);


% =========== PERFORMANCE OF THE ANN ========================

%---------TRAINING SET------------
yTrain = exp(net(Xt(:, tr.trainInd)))-1;
yTrainTrue = exp(yt(tr.trainInd))-1;
rms_train = sqrt(mean((yTrain - yTrainTrue).^2));

%--------VALIDATION SET-----------
yVal = exp(net(Xt(:, tr.valInd)))-1;
yValTrue = exp(yt(tr.valInd))-1;
rms_val = sqrt(mean((yVal - yValTrue).^2));

%--------TEST SET-----------
yTest = exp(net(Xt(:, tr.testInd)))-1;
yTestTrue = exp(yt(tr.testInd))-1;
rms_test = sqrt(mean((yTest - yTestTrue).^2));

%---------- CHECK -------------
comp = [rms_train; rms_val; rms_test]

%figure;
plot(yTrainTrue, yTrain, 'x'); 
title('Model Visualization of Training, Validation, and Test Data');
xlabel('Input Values');
ylabel('yield');
hold on;
plot(yValTrue, yVal, 'go'); hold on;
plot(yTestTrue, yTest, 'rx'); hold on;
plot(0:1,0:1);
hold off;
legend('training set', 'validation set', 'test set', 'Location', 'NorthWest');

%% 

%-----------OPTIMIZE THE NUMBER OF NEURONS IN THE HIDDEN LAYER
for i = 1:60
    % defining the architecture of the ann
    hiddenLayerSize = i;
    net = fitnet(hiddenLayerSize);
    net.divideParam.trainRatio = 60/100;
    net.divideParam.valRatio = 20/100;
    net.divideParam.testRatio = 20/100;
    
    %training the ann
    [net, tr] = train(net, Xt, yt);
    
    %determin the error of the ann
  
    yTrain = exp(net(Xt(:, tr.trainInd)))-1;
    yValTrue = exp(yt(tr.valInd))-1;
    yTrainTrue = exp(yt(tr.trainInd))-1;
    yVal = exp(net(Xt(:, tr.valInd)))-1;

    rmse_train(i) = sqrt(mean((yTrain - yTrainTrue).^2));
    rmse_val(i) = sqrt(mean((yVal - yValTrue).^2));


end
%% 

figure;
% select the optimal number of neurons in the hidden layer
plot(1:60, rmse_train); 
hold on;
plot(1:60, rmse_val); hold off;



% For this project I used the fitnet function to train my model. I have
% also tried writing code to train this model using feed forward
% back propagation, but with little success. I am yet to get a full grasp
% on the concept of using FFBPNN to train models of linear systems, but I
% will update this repo when I have successfully done that. I will also
% appreciate any assistance I can get to help me better understand the BPNN
% for linear systems.