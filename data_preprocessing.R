##############################################################
#                                                            #
# Topic: Detect synthetic cannabinoids with machine learning #
# Author: Gabriel Streun (gabriel.streun@irm.uzh.ch)         #
# Date: November 2021                                        #
#                                                            #
##############################################################


#import the csv file with the abundance table exported from Progenesis QI
data<-as.matrix(read.csv(file.choose(), header=TRUE, sep=";"))

#inspect the data
dim(data)

#set the amount of samples
samples<-474

#display the normalized features and then store them in rawdata
data[2,16:(16+samples-1)]
rawdata<-data[2:nrow(data),16:(16+samples-1)]

#clean_PG_data function to obtain a matrix with columns=features and rows=samples
clean_PG_data<-function(rawdata){
  rawdata<-t(rawdata)
  rownames(rawdata)<-rawdata[,1]
  rawdata<-rawdata[,-1]
  colnames(rawdata)<-data[3:nrow(data),1]
  procdata<-sapply(rawdata, FUN=as.numeric)
  procdata<-matrix(data=procdata, nrow=dim(rawdata)[1], ncol=dim(rawdata)[2])
  colnames(procdata)<-data[3:nrow(data),1]
  rownames(procdata)<-NULL
  data<<-procdata
}
clean_PG_data(rawdata)

#inspect the data
dim(data)
