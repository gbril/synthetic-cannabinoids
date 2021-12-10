##############################################################
#                                                            #
# Topic: Detect synthetic cannabinoids with machine learning #
# Author: Gabriel Streun (gabriel.streun@irm.uzh.ch)         #
# Date: 30.11.2021                                           #
#                                                            #
##############################################################

#The data conversion step after 
#Progenesis QI export is described in 
#DOI:10.1515/cclm-2021-0010

#Total data set: 474 samples x 18'656 features

#0. Loading libraries
library(ggplot2)
library(randomForest)
library(dplyr)
library(corrr)
library(caret)

#1. Feature engineering

#1.1 Correlation analysis
lab<-c(rep(1,176),rep(0,193))
data<-cbind(lab,data) #data: 369 training set samples x 18'656 features
corrr_analysis<-correlate(data) %>%
  focus(lab)

#A ROC curve analysis was done with the MetaboAnalyst online tool
#The overlap of the features with a ROC curve threshold >0.7 and a Pearson correlation >=0.25 or <=-0.25 were read out in Excel
#This overlap (n=106) was selected as important features

#1.2 Initial random forest (RF) model to get the feature importance
model<-randomForest(data,labels) #data: 369 training set samples x 106 important features, labels: factor with 176 1's and 193 0's
importance(model)

#1.3 Iterate RF models with the features in descending importance
sensitivity<-c()
specificity<-c()
for(i in 1:106){ #for(i in 2:i)
  model <- randomForest(data[,1:i], #data: 369 training set samples x 106 important features
                        labels,
                        mtry=sqrt(i))
  sensitivity<-c(sensitivity,model$confusion[4]/sum(model$confusion[4],model$confusion[2]))
  specificity<-c(specificity,model$confusion[1]/sum(model$confusion[1],model$confusion[3]))
  print(i)
}

#The best sensitivity and specificity is obtained with 49 features

#2. Machine learning model

#2.1 The used data frames are the following:
#training set: train_data (369 samples x 49 features)
#test set: positive (25 samples x 49 features)
#test set: negative (25 samples x 49 features)
#verification set: thc (15 samples x 49 features)
#verification set: spiked (20 samples x 49 features)
#verification set: amph (5 samples x 49 features)
#verification set: coc (5 samples x 49 features)
#verification set: opi_benz (5 samples x 49 features)
#verification set: mixed (5 samples x 49 features)

#2.2 Scaling and normalization
data<-rbind(positive,thc,train_set,negative,spiked,amph,coc,opi_benz,mixed) #data: 474 samples x 49 features
for(i in 1:nrow(data)){
  data[i,]<-data[i,]/sum(data[i,])
  print(i)
}
data<-scale(data)
data<-as.data.frame(data[,1:49])

#2.3 PCA plot
labels_pca<-as.factor(c(rep("testPos",25),rep("verifTHC",15),rep("trainPos",176),rep("trainNeg",193),rep("testNeg",25),rep("spiked",20),rep("otherDrugs",20)))
data_pca<-cbind(labels_pca,data) #data: 474 samples x 49 features
colnames(data_pca)[1]<-"y"
df<-data_pca[,2:ncol(data_pca)]
pca_res <- prcomp(df)
PCi<-data.frame(pca_res$x,label=data_pca$y)
PCA<-ggplot(PCi,aes(x=PC1,y=PC2,col=label))+
  geom_point()+
  scale_color_manual(breaks = c("testPos","verifTHC","trainPos","trainNeg","testNeg","spiked","otherDrugs"),
                          values=c("cyan","green4","blue","red","magenta3","hotpink1","darkorange"))

PCA<-PCA+theme_bw()

ggsave(PCA,file="PCA_final.eps")

#2.4 k-means clustering
rownames(data)<-c(paste("testPos",1:25,sep=""),paste("verifTHC",1:15,sep=""),paste("trainPos",1:176,sep=""),paste("trainNeg",1:193,sep=""),paste("testNeg",1:25,sep=""),paste("spiked",1:20,sep=""),paste("otherDrugs",1:20,sep=""))
colnames(data)<-c(1:49)
res<-kmeans(data[,2:ncol(data)], 2)
names(which(res$cluster==1))
names(which(res$cluster==2))

#2.5 Optimize RF model parameters
#2.5.1 ntree parameter
labels<-as.factor(c(rep("trainPos",176),rep("trainNeg",193)))
model<-randomForest(data[41:409,],labels,ntree=500) #only training set
model

oob.error.data <- data.frame(
  Trees=rep(1:nrow(model$err.rate), times=3),
  Type=rep(c("OOB", "trainNeg", "trainPos"), each=nrow(model$err.rate)),
  Error=c(model$err.rate[,"OOB"], 
          model$err.rate[,"trainNeg"], 
          model$err.rate[,"trainPos"]))

trees<-ggplot(data=oob.error.data, aes(x=Trees, y=Error)) +
  geom_line(aes(color=Type))+
  scale_color_manual(values = c("black","red","blue"))
trees<-trees+theme_bw()

ggsave(trees,file="trees_final.eps")

#2.5.2 mtry parameter
oob.values <- vector(length=14)
for(i in 1:14) {
  temp.model<-randomForest(data[41:409,],labels,mtry=i,ntree=200) #only training set
  oob.values[i] <- temp.model$err.rate[nrow(temp.model$err.rate),1]
}
oob.values
min(oob.values)
which(oob.values == min(oob.values))

#2.6 10-fold cross validation
labels<-as.factor(c(rep(1,176),rep(0,193)))
x_train<-data[41:409,] #only training set
y_train<-labels

k<-10
indices<-sample(1:nrow(x_train))
folds<-cut(indices,breaks=k,lables=FALSE)

acc_scores<-c()
sens_scores<-c()
spez_scores<-c()
ppv_scores<-c()
npv_scores<-c()
tables_list<-list()

for(i in 1:k){
  cat("processing fold #", i, "\n")
  
  val_indices<-which(folds == levels(folds)[i], arr.ind=TRUE)
  val_data<-x_train[val_indices,]
  val_targets<-y_train[val_indices]
  partial_train_data<-x_train[-val_indices,]
  partial_train_targets<-y_train[-val_indices]
  
  forest<-randomForest(partial_train_data,partial_train_targets,ntree=200,mtry=7)
  pred<-predict(forest,newdata=val_data,type="class")
  table1 <- table(Predicted=pred, Actual=val_targets)[2:1,2:1]
  
  acc_scores<-c(acc_scores,(table1[1,1]+table1[2,2])/length(val_targets))
  sens_scores<-c(sens_scores,sensitivity(table1))
  spez_scores<-c(spez_scores,specificity(table1))
  ppv_scores<-c(ppv_scores,posPredValue(table1))
  npv_scores<-c(npv_scores,negPredValue(table1))
  tables_list<-c(tables_list,table1)
}

x<-mean(acc_scores)
paste("After 10-fold CV the mean accuracy is ", round(x*100,2), "% ", "(Min = ", min(acc_scores)*100, "%, ", "Max = ", max(acc_scores)*100, "%)", sep="")
x<-mean(sens_scores)
paste("After 10-fold CV the mean sensitivity is ", round(x*100,2), "% ", "(Min = ", min(sens_scores)*100, "%, ", "Max = ", max(sens_scores)*100, "%)", sep="")
x<-mean(spez_scores)
paste("After 10-fold CV the mean specificity is ", round(x*100,2), "% ", "(Min = ", min(spez_scores)*100, "%, ", "Max = ", max(spez_scores)*100, "%)", sep="")
x<-mean(ppv_scores)
paste("After 10-fold CV the mean ppv is ", round(x*100,2), "% ", "(Min = ", min(ppv_scores)*100, "%, ", "Max = ", max(ppv_scores)*100, "%)", sep="")
x<-mean(npv_scores)
paste("After 10-fold CV the mean npv is ", round(x*100,2), "% ", "(Min = ", min(npv_scores)*100, "%, ", "Max = ", max(npv_scores)*100, "%)", sep="")
tables_list

#2.7 Final model (500 iterations and average value)
testpos<-c()
verifTHC<-c()
testneg<-c()
spiked<-c()
otherdrugs<-c()
tl<-c()
bl<-c()
tr<-c()
br<-c()
labels<-as.factor(c(rep(1,176),rep(0,193)))
for(i in 1:500){
  model <- randomForest(data[41:409,],
                        labels,
                        ntree=200, 
                        proximity=TRUE, 
                        mtry=7)
  tl<-c(tl,model$confusion[4])
  bl<-c(bl,model$confusion[3])
  tr<-c(tr,model$confusion[2])
  br<-c(br,model$confusion[1])
  testpos<-c(testpos,sum(as.numeric(as.vector(predict(model,data[1:25,]))))/25)
  verifTHC<-c(verifTHC,sum(as.numeric(as.vector(predict(model,data[26:40,]))))/15)
  testneg<-c(testneg,length(which(as.numeric(as.vector(predict(model,data[410:434,])))==0))/25)
  spiked<-c(spiked,length(which(as.numeric(as.vector(predict(model,data[435:454,])))==0))/20)
  otherdrugs<-c(otherdrugs,length(which(as.numeric(as.vector(predict(model,data[455:474,])))==0))/20)
  print(i)
}
all_49<-data.frame(tl,bl,tr,br,testpos,verifTHC,testneg,spiked,otherdrugs,iteration=1:500)
maxs<-data.frame(max=sapply(all_49[,-10],max))
mins<-data.frame(max=sapply(all_49[,-10],min))
means<-data.frame(max=sapply(all_49[,-10],mean))
meds<-data.frame(max=sapply(all_49[,-10],median))

#MDS plot
labels<-as.factor(c(rep("allPositive",176),rep("allNegative",193)))
model <- randomForest(data[41:409,],
                      labels,
                      ntree=200, 
                      proximity=TRUE, 
                      mtry=7)

distance.matrix <- as.dist(1-model$proximity)

mds.stuff <- cmdscale(distance.matrix, eig=TRUE, x.ret=TRUE)

mds.var.per <- round(mds.stuff$eig/sum(mds.stuff$eig)*100, 1)

mds.values <- mds.stuff$points

mds.data <- data.frame(Sample=rownames(mds.values),
                       X=mds.values[,1],
                       Y=mds.values[,2],
                       Class=labels)

p<-ggplot(data=mds.data, aes(x=X, y=Y, label=Sample)) + 
  geom_point(aes(color=Class)) +
  theme_bw() +
  xlab(paste("MDS1 - ", mds.var.per[1], "%", sep="")) +
  ylab(paste("MDS2 - ", mds.var.per[2], "%", sep="")) +
  ggtitle("MDS plot using (1 - Random Forest Proximities)")
MDS<-p+scale_color_manual(breaks = c("allPositive","allNegative"),
                        values=c("blue","red"))
MDS<-MDS+theme_bw()

ggsave(MDS,file="MDS_final.eps")
