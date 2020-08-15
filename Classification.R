
#####################################################################
#####################################################################
#  
#  CLASSIFICATION & Discriminant Analysis
#       LDA, QUADRATIC
#  with dfferent types of error rates

#  1. PLUG-IN
#  2. CROSS-VALIDATION
#  3. K-FOLD
#     and KNN Classification (Nonparametric Classification)
#####################################################################
#####################################################################

# We can read the SAT scores (with graduation information) data from the Internet:
require(MASS)
#setwd("~/auburn/spring18/stat7840/codes")
satgradu <- read.table("satgradu.txt", header=T)

attach(satgradu)
# The first three columns are the SAT scores on the 3 tests,
# the last column (called gradu) is an indicator of whether the student successfully graduated
# (1 = graduated, 0 = did not graduate)
#####################################################################
################         LDA    ######################################
#####################################################################
# Using the built-in lda function in the MASS package
# for linear discriminant analysis:

# assuming equal prior probabilities of graduating or not:

dis <- lda(gradu ~ math + reading + writing, data=satgradu, prior=c(0.5, 0.5))

## THis is another LDA function that exists in R package.

dis11=lda(satgradu[,1:3], grouping=satgradu[,4], prior=c(0.5, 0.5))

# a1, a2, a3 are given as "Coefficients of linear discriminants".

# Let's predict whether a new applicant with 
# SAT scores of: math = 550, reading = 610, writing = 480
# will graduate:

newobs <- rbind( c(550,610,480) )
dimnames(newobs) <- list(NULL,c('math','reading', 'writing'))
newobs <- data.frame(newobs)
Out1=predict(dis,newdata=newobs)
Out1
# Posterior probabilities of this applicant being in each group:

predict(dis,newdata=newobs)$posterior

# Making predictions for several new individuals at once:

newobs <- rbind( c(300,420,280), c(510,480,470), c(780,760,710) )
dimnames(newobs) <- list(NULL,c('math','reading', 'writing'))
newobs <- data.frame(newobs)
predict(dis,newdata=newobs)


# assuming prior probabilities of graduating is about twice as large 
# as probability of not graduating:

# dis <- lda(gradu ~ math + reading + writing, data=satgradu, prior=c(0.33, 0.67))

# If we do not specify any prior probabilities, it will by default use the proportions
# of the sampled individuals that are in each group as the prior probabilities.

dis2 <- lda(gradu ~ math + reading + writing, data=satgradu)

dis2


##### Misclassification rate of LDA rule:

# Simple plug-in misclassification rate:

group<-predict(dis, satgradu, method='plug-in')$class
out2=table(group,gradu)
outm=as.matrix(out2)
dim(outm)
pER=(outm[1,2]+outm[2,1])/(outm[1,2]+outm[2,1]+outm[1,1]+outm[2,2])
pER
# The plug-in misclassification rate for LDA here is (11+4)/40 = 0.375.

# cross-validation rate of LDA rule:

########
correct<-rep(0,times=nrow(satgradu) )
for (j in 1:nrow(satgradu) ) {
mydis<-lda(grouping=gradu[-j], x=satgradu[-j,1:3], prior=c(0.5,0.5))
mypred<-predict(mydis,newdata=satgradu[j,1:3])$class
correct[j] <- (mypred==gradu[j])
}
cv.misclass <- 1-mean(correct)
cv.misclass
#########

# The cross-validation misclassification rate for LDA here is 0.425.

####################################################################################
####################################################################################

####### Quadratic discriminant analysis (QDA) can be implemented with the qda function:

####################################################################################
####################################################################################

disquad <- qda(gradu ~ math + reading + writing, data=satgradu, prior=c(0.5, 0.5))

##### Misclassification rate of QDA rule:

# Simple plug-in misclassification rate:

group<-predict(disquad, satgradu, method='plug-in')$class
table(group,gradu)
out3=table(group,gradu)
outm3=as.matrix(out3)
dim(outm3)
pQER=(outm3[1,2]+outm3[2,1])/(outm3[1,2]+outm3[2,1]+outm3[1,1]+outm3[2,2])
pQER
# The plug-in misclassification rate for QDA here is (10+2)/40 = 0.3.

# cross-validation rate of QDA rule:

########
correct<-rep(0,times=nrow(satgradu) )
for (j in 1:nrow(satgradu) ) {
mydisquad<-qda(grouping=gradu[-j], x=satgradu[-j,1:3], prior=c(0.5,0.5))
mypred<-predict(mydisquad,newdata=satgradu[j,1:3])$class
correct[j] <- (mypred==gradu[j])
}
cv.misclass <- 1-mean(correct)
cv.misclass
#########


# The cross-validation misclassification rate for QDA here is still 0.425.

####################################################################################
##                   K-Fold
####################################################################################
library(crossval)
c1 = satgradu[,4]
X1 = as.matrix(satgradu[c1!=2,-4])
Y1 = as.factor(c1[c1!=2])
dim(X1) # 130 13
levels(Y1)
sum(Y1==1)

# set up lda prediction function
predfun.lda = function(train.x, train.y, test.x, test.y, negative)
{
  require("MASS") # for lda function
  lda.fit = lda(train.x, grouping=train.y, prior=c(0.5,0.5))
  ynew = predict(lda.fit, test.x)$class
  # count TP, FP etc.
  out = confusionMatrix(test.y, ynew, negative=negative)
  return( out )
}
set.seed(12345)
#crossval performs K-fold cross validation with B repetitions. If Y is a factor then balanced sampling
#is used (i.e. in each fold each category is represented in appropriate proportions).

cv.out = crossval(predfun.lda, X1, Y1, K=10, B=10, negative="1")

#diagnosticErrors computes various diagnostic errors useful for evaluating the performance of a
#diagnostic test or a classifier: accuracy (acc), sensitivity (sens), specificity (spec), positive predictive
#value (ppv), negative predictive value (npv), and log-odds ratio (lor).

a=diagnosticErrors(cv.out$stat)
a1=as.matrix(a)

###MISCLASSIFICATION ERROR RATE

ER=1-a1[1,]
require(kLaR)
####################################################################################
## knn for a IRIS data
# Do this for SATGRADU data.
#####################################################################################
library(class)
data(iris3)

train1 <- rbind(iris3[1:25,,1], iris3[1:25,,2], iris3[1:25,,3])
test1 <- rbind(iris3[26:50,,1], iris3[26:50,,2], iris3[26:50,,3])
cl <- factor(c(rep("s",25), rep("c",25), rep("v",25)))
class<-knn(train1, test1, cl, k = 3, prob=TRUE)
mcr <- 1 - sum(class == cl) / length(cl)
mcr
######
#### one method to choose k
##############
k=c(1:10)
p=rep(0,10)
summary=cbind(k,p)
colnames(summary)=c("k","Percent misclassified")
for(i in 1:10){
  result1=knn(train1, test1, cl, k=i)
  summary[i,2]=(nrow(test1)-sum(diag(table(result1,cl))))/nrow(test1)
}
dim(summary)
plot(summary[,1], summary[,2])
min(summary[,2])