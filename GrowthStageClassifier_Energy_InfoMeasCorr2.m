dataset = readtable('Book1.xlsx');
X = dataset(:,2:3);
Y = dataset(:,1);
y = table2array(Y);
y = categorical(y);
x = table2array(X);
labels = categories(y);
gscatter(x(:,1),x(:,2),y,'rgb','osd')
xlabel('Energy');
ylabel('Information Measure Correlation 2');

classifier_name = {'Naive Bayes','Discriminant Analysis','Classification Tree','Nearest Neighbor'};
classifier{1} = fitcnb(x,y);
classifier{2} = fitcdiscr(x,y);
classifier{3} = fitctree(x,y);
classifier{4} = fitcknn(x,y);

x1range = min(x(:,1)):.01:max(x(:,1));
x2range = min(x(:,2)):.01:max(x(:,2));
[xx1, xx2] = meshgrid(x1range,x2range);
XGrid = [xx1(:) xx2(:)];


for i = 1:numel(classifier)
   predictedGrowthStage = predict(classifier{i},XGrid);

   subplot(2,2,i);
   gscatter(xx1(:), xx2(:), predictedGrowthStage,'rgb');

   title(classifier_name{i})
   legend off, axis tight
end

legend(labels,'Location',[0.35,0.01,0.35,0.05],'Orientation','Horizontal')