clc;
close all;
info1=readtable('batch_logs (5).csv');
plot(info1.loss)
hold on;
x=1:171:85500;
x=x';
y=info1.ValidationLoss;
y=y(~isnan(y));
plot(x,y)
grid on;
xlabel("Iterations")
ylabel("Loss")
legend("Training Loss","Validation Loss")
title("LNN Loss Trained with ADAM")
% figure(2)
% plot(info2.TrainingLoss)
% hold on;
% plot(info2.ValidationLoss)
% xlabel("Iterations")
% ylabel("Loss")
% legend("Training Loss","Validation Loss")
% title("AlexNet Loss Trained with ADAM")