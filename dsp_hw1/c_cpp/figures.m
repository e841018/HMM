fileID = fopen('log.txt', 'r');
acc = fscanf(fileID, '%d %f', [2, Inf]);
fclose(fileID);
plot(acc(1,:), acc(2,:));
title('testing set'); xlabel('Iterations'); ylabel('Accuracy');

figure(); hold;
title('training set'); xlabel('Iterations'); ylabel('Accuracy');
for i = 1:5
    fileID = fopen(sprintf('log%d.txt', i), 'r');
    acc = fscanf(fileID, '%d %f', [2, Inf]);
    fclose(fileID);
    plot(acc(1,:), acc(2,:));
end
legend(...
    'model\_01.txt',...
    'model\_02.txt',...
    'model\_03.txt',...
    'model\_04.txt',...
    'model\_05.txt'...
);