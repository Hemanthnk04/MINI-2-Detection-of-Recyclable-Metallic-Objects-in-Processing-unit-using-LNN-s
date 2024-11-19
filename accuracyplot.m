clc;
close all;
info1=readtable('batch_logs (5).csv');
plot(100*info1.accuracy)
hold on;
x=1:171:85500;
x=x';
y=info1.ValidationAccuracy;
y=y(~isnan(y));
plot(x,100*y)
grid on;
xlabel("Iterations")
ylabel("Accuracy")
legend("Training Accuracy","Validation Accuracy")
title("LNN Accuracy Trained with ADAM")
% figure(2)
% plot(info2.TrainingAccuracy)
% hold on;
% plot(info2.ValidationAccuracy)
% xlabel("Iterations")
% ylabel("Accuracy")
% legend("Training Accuracy","Validation Accuracy")
% title("AlexNet Accuracy Trained with SGDM")