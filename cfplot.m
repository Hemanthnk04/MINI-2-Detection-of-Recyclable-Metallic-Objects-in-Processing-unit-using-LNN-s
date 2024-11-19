% Define the confusion matrix based on the provided values
confMat = [
    56  2   5   2   0  15  12  0   0;
    1   51  0   0   1  27  0   2   0;
    7   2   29  0   12 6   28  0   0;
    6   0   0   77  0  2   0   1   0;
    0   7   1   1   56 5   3   25  1;
    4   1   2   0   2  82  3   6   0;
    5   0   0   0   38 17  14  10  0;
    1   1   1   0   24 13  6   12  6;
    0   19  1   0   15 0   1   0   51
];

% Create a confusion chart
figure;
confusionchart(confMat, {'Cardboard', 'Food Organics', 'Glass', 'Metal Reuseable', ...
                         'Miscellaneous Trash', 'Paper', 'Plastic', 'Textile Trash', 'Vegetation'});

% Customize chart appearance
title('LNN Confusion Matrix');
% colormap(summer); % Adjust color scheme as desired