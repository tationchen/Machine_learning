---
title: "Final_project"
author: "Tianxiang Chen, Xuanchen Ren, Yuqiao Liu"
date: "2/28/2022"
output:
  html_document:
    toc: yes
    toc_float: yes
    code_folding: hide
    number_sections: yes
  pdf_document:
    toc: yes
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(warning = FALSE, message = FALSE,eval=TRUE) 
```

The dataset source:
https://www.kaggle.com/vardhansiramdasu/income
Kaggle, Income Classification, by Vardhan Siramdasu




```{r}
library(tidyverse)
library(ISLR)
library(glmnet)
library(tree)
library(maptree)
library(randomForest)
library(gbm)
library(ROCR)
library(caret)
library(leaps)
```


# Introduction

What is subsidies?  

A subsidy is a benefit given to a person, business or institution, usually by the government. It can be direct (such as a cash payment) or indirect (such as a tax break). A subsidy is usually intended to eliminate some type of burden, it is usually considered to be in the overall public interest, and it is intended to promote a social good or economic policy.  

Why it's important to have subsidies nowadays?  

Subsidies are incentives given by the government to individuals or businesses in the form of cash, grants or tax breaks to improve the availability of certain goods and services and the financial crisis of individuals. With subsidies, consumers are able to address their own personal financial difficulties. With the impact of the epidemic on the world economy in the last two years, more and more people have lost their jobs. That's why government subsidies are especially important in this situation.  

Why it is important to issue the correct subsidy to those who are eligible?

For an employee, the amount of salary determines the quality of his life. For those who are not well paid, it is necessary for the government to issue subsidies. But money is not unlimited, and the government needs to predict how many people will need the subsidy. Therefore, this project is to predict whether an employee needs a subsidy or not.


# Import Data
*Import the csv to R*
```{r cars}
income <- read_csv("C:/Users/Tation'chan/Desktop/pstat131/Final project/income.csv")
#View(income)
```


# Data Cleaning 
## Compute and Delete Missing Value 
show how many missing value in csv*
and delete  remove all rows with NA values*
```{r}
sum(is.na(income))
income<-na.omit(income)
```
Here total 3625 missing value in the csv.
Here we known that total 3625 missing value in the csv.
We then create New Colume need_subsidy- "Yes" if a person's salary is less than or equal to 50,000, and "No" otherwise.


## Create New Colume need_subsidy
New define a new factor response variable need_subsidy which is "Yes" if
a person's salary is less than or equal to 50,000, and "No" otherwise.
```{r}
income<-income %>% mutate(need_subsidy=as.factor(ifelse(SalStat=="less than or equal to 50,000","YES", "NO")))
```



## Removing SalStat
Before we start to fit model we have to remove the SalStat since this variable is overlap with need_subsidy variable.
```{r}
income <-income %>% select(-c(SalStat))
```




## Creating Training and Test Data Set
Now We have to split the data into the training and test set
```{r}
train = sample(nrow(income), nrow(income)/2)
dat.train = income[train, ]
# The rest as test data
dat.test = income[-train, ]
```


# Exploratory Data Analysis
## Data Summery
```{r}
summary(income)
```

Due to the random distribution of the data, and the fact that many of the variables are not numerical variables. It is difficult for us to find useful information from the summary.  

## Barplot for Numeric Variables
Here we create some barplot for some numeric variable from the dataset.
```{r}
barplot(table(income$age))
```
Combined with the histogram of ages we see that this dataset, although ignoring people under 17 years old, still fits the characteristics of a normal-normal distribution.
```{r}
barplot(table(income$capitalgain))
barplot(table(income$capitalloss))
```
Capital gains and losses are hard to tell anything from the graph, so we have to combine it with the correlation plot that follows.


## Correlation Summary and Plot 
We select the numeric variables to fit in correlation summary and plot which are age, capitalgain, capitalloss,working hours.
```{r}
income.1 <- income[, c(1,9,10,11)]

income.cor <- cor(income.1)
library(corrplot)
corrplot(income.cor, type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45)
```
In the Corplot, positive correlations are displayed in blue and negative correlations in red color.We can conclude that 
```{r}
library("PerformanceAnalytics")
chart.Correlation(income.1, histogram=TRUE, pch=19)

```
And in the chartplot of each variable. The bottom shows that the bivariate scatter plots with a fitted line, On the top of the diagonal,the value of the correlation plus the significance level as stars. Each significance level is associated to a symbol : p-values(0, 0.001, 0.01, 0.05, 0.1, 1) <=> symbols(“***”, “**”, “*”, “.”, " “)  
We can conclude that the curve of working hours tends to be smooth and the effect of other variables on working hours is not significant. Combined with the previous summary, we can conclude that the working hours are stable at about 40 hours and are not affected by other factors.

# Build Models 
## Decision Tree
Now we will Construct a single decision tree to predict need_subsidy using all other predictors in the training data.
```{r}
#fit the decision tree
tree.income<-tree(need_subsidy~.,data = dat.train)
#Visualize the tree
plot(tree.income)
text(tree.income, pretty = 0, cex = .4, col = "red")
title("decision tree on income", cex = 0.8)
```


### Using Cross Validation to Obtain the Optimize Size for Tree 
Now this is the decision tree without prune
let find the best size of tree by cross validation
```{r}
#use 5-th cv to determine the best size tree.
# Set random seed
set.seed(3)
# K-Fold cross validation
cv = cv.tree(tree.income, FUN=prune.misclass, K=5)
#best size
best.cv = min(cv$size[cv$dev == min(cv$dev)])
best.cv
```
Here the result show that the best size from cross validation is four




### Now we will prume the original tree to size 4
```{r}
# Prune tree.subdrug
pt.cv = prune.misclass (tree.income, best=best.cv)
# Plot pruned tree
plot(pt.cv)
text(pt.cv, pretty=0, col = "blue", cex = .5)
title("Pruned tree of size 4")
```


### get the tpr from decision tree
```{r}
# Predict on tpr
pred.pt.cv = predict(pt.cv,dat.test, type="class")
# Obtain confusion matrix
err.pt.cv = table(pred.pt.cv,dat.test$need_subsidy)
err.pt.cv
#The tpr
err.pt.cv[1,1]/sum(err.pt.cv[,1])
```





## Using the Regularized Regression Methods
Now we use the regularized regression methods to predict the income and get the test error.
For this part we will use ridge regression model & lasso regression model
First we have to find the  the optimal value of tuning parameter lambda using cross validation
```{r}
#split the data
set.seed(123)
x=model.matrix(need_subsidy~.,data =income)[,-1]
y=income$need_subsidy
train = sample(nrow(income), nrow(income)/2)
test=(-train)
x.train=x[train,]
y.train=y[train]
x.test=x[test,]
y.test=y[test]
```





### Fit the Ridge Model
```{r}
lambda.list.ridge = 1000 * exp(seq(0, log(1e-5), length = 100))
ridge_mod = glmnet(x.train, y.train, alpha = 0, lambda = lambda.list.ridge,family = "binomial")
cv.out.ridge=cv.glmnet(x.train, y.train, alpha = 0,folds=5,family = "binomial")
bestlam = cv.out.ridge$lambda.min
bestlam
ridge.pred=predict(ridge_mod,s=bestlam ,newx=x.test,type = "response")
# Save the predicted labels using 0.5 as a threshold
predsubsidy=as.factor(ifelse(ridge.pred<=0.5, "No", "Yes"))

``` 




#### Confusion matrix (training error/accuracy)
```{r}
error<-table(pred=predsubsidy, true=dat.test$need_subsidy)
#test error
1-sum(diag(error))/(sum(error))
x=model.matrix(need_subsidy~.,data =income)[,-1]
y=income$need_subsidy
out = glmnet(x,y,alpha=0,family = "binomial")
predict(out,type="coefficients",s=bestlam)
```
The test error of ridge model is 0.34

From the table of coefficient of ridge we can some result list below.  
1.age don't really effect that people need subsidy or not.  
2.People who work without pay more likely need subsidy.  
3.People have lower graducated level more likely need subsidy.  
4.People never married before more likely need subsidy.  
5.People who are work for Private house server industry more likely need subsidy.  
6.race except Asian, black white arer more likely need subsidy.  
7.People who own a child more likely need subsidy.  
8.Gender don't have positive relation with need_subsidy or not.  
9.people who from developing country are more likely need subsidy.
10.summary the variable above we find the the jobtype-without pay have more closed relation with need_subsidy or not



### Fit the Lasso Regession
```{r}
lambda.list.lasso = 2 * exp(seq(0, log(1e-4), length = 100))
lasso.mod <- glmnet(x.train, y.train, alpha=1, lambda = lambda.list.ridge,family = "binomial")
set.seed(1)
cv.out.lasso = cv.glmnet(x.train, y.train, alpha = 1,family = "binomial")
bestlam = cv.out.lasso$lambda.min
lasso.pred = predict(lasso.mod, s = bestlam, newx =x.test,type = "response")
predsubsidy=as.factor(ifelse(lasso.pred<=0.5, "No", "Yes"))
# Confusion matrix (training error/accuracy)
lasso_error<-table(pred=predsubsidy, true=dat.test$need_subsidy)
#test error
1-sum(diag(lasso_error))/(sum(lasso_error))

out=glmnet(x,y,alpha=1,lambda=lambda.list.ridge,family = "binomial")
lasso.coef=predict(out,type="coefficients",s=bestlam)
lasso.coef
```
The test error of lasso regession is 0.331

From the table of coefficent of lasso regression, we found that no coefficient is shown for some predictor show - because the lasso regression shrunk the coefficient all the way to zero. This means it was completely dropped from the model because it wasn’t influential enough.


## construct a random forest model
### fit the random forest model
Now we will construct a random forest model to predict if subsidy is needed.
Firstly we use training data to drain the model.
```{r}
#fit the random forest model
#View(dat.train) 
rf.train=randomForest(need_subsidy~.,data = dat.train,importance=TRUE)
rf.train
```





#### predict the result from the model above
After training the model, we will predict on the test data.
```{r}
#After training the model, predict on the test data.
rf.predict = predict(rf.train, newdata = dat.test)
```




#### Get the confusion matrix 
The prediction with the test data is finished, then we are able to build a confusion matrix, to see directly the result. Finally we compute the test error rate, using the data of confusion matrix.

```{r}
# Confusion matrix
rf.err = table(pred = rf.predict, truth = dat.test$need_subsidy )
#Calculating the test error
test.rf.err = 1 - sum(diag(rf.err))/sum(rf.err)
test.rf.err
```
the test error rate is 0.138, which indicates that random forest yielded an lowest test error rate so far.



## log regresssion model 
### Fit logistic regression model
```{r}
# Fit logistic regression model 
glm.fit = glm(need_subsidy~.,data = dat.train, family=binomial)
# Summarize the logistic regression model
summary(glm.fit)
```
Here we can set up a hypothesis for each predictor Xi that
H0:The predictor Xi are related to response 
H1:The predictor xi are not related to response where i in 0 to n

So the from the table above we get the p-value for each predictor xi if the p-value is less than 0.05 then we reject the null hypothesis
and this predictor wasn’t influential enough for the response.The predictor list below are less than 0.05

For the variable jobtype we find that who work for government or have their own business are not need subsidy since those variable are less
than 0.05.

For the variable Edtype we find that people who have the high level education are not need subsidy since those variable are less than 0.05

For the marital status we find that people were married are not need subsidy since those variable less than 0.05.

For the occupation we find that people who work in the farming or service industry are not need subsidy since those variable less than 0.05.

For the race we find that white and Asian are not need subsidy since those variable less than 0.05.

For the country we find that the person who from east asian, west euro,North and middle amrican are not need subsidy since those variable 
less than 0.05.



### Construct confusion matrix for the training data 
```{r,results= FALSE}
prob.training = predict(glm.fit, type="response")
#round(prob.training, digits=2)
# Save the predicted labels using 0.5 as a threshold
dat.train = dat.train %>%
  mutate(pred.need_subsidy=as.factor(ifelse(prob.training<=0.5, "No", "Yes")))
# Confusion matrix (training error/accuracy)
log_error = table(pred=dat.train$pred.need_subsidy, true=dat.train$need_subsidy)
log_error
```


### Construct ROC curve and compute AUC to determind how the model fits.
```{r}
library(ROCR)
# First arument is the prob.training, second is true labels
pred = prediction(prob.training, dat.train$need_subsidy)
# TPR on the y axis and FPR on the x axis
perf = performance(pred, measure="tpr", x.measure="fpr")
plot(perf, col=2, lwd=3, main="ROC curve")
abline(0,1)

# Calculate AUC
auc = performance(pred, "auc")@y.values
auc
```
Since the AUC value is 0.905499, which is a optimal result. The model fits the case well. 





# Compare the performance of each models
To choose the best model, we will calculate the TPR and FPR for each model.
TPR=TP/(TP+FN), and FPR=FN/(FP+TN)

## copy the confusion matrix and test error for random forest
We firstly copy the confusion matrix and test error for random forest
```{r}

rf.err = table(pred = rf.predict, truth = dat.test$need_subsidy )
#Calculating the test error
test.rf.err = 1 - sum(diag(rf.err))/sum(rf.err)
test.rf.err

#TPR=94%
rf.err[2,2]/sum(rf.err[,2])
#FPR=18%
(rf.err[1,2])/(sum(rf.err[,1]))
```
Then we found out that the TPR of random forest is 94%, and the FPR is 18%



## Confusion matrix and test error for logistic regression
Confusion matrix and test error for logistic regression
```{r}
log_erorr = table(pred=dat.train$pred.need_subsidy, true=dat.train$need_subsidy)
log_error
#test error
1-1-sum(diag(log_error))/(sum(log_error))

#TPR=93%
log_error[2,2]/sum(log_error[,2])
#FPR=21%
(log_error[1,2])/(sum(log_error[,1]))
```
Then we found out the TPR is about 93%, and the FPR is 21%



## confusion matrix and test error for lasso regression
```{r}
lasso_error = table(pred=predsubsidy, true=dat.test$need_subsidy)
lasso_error
#test error
1-sum(diag(lasso_error))/(sum(lasso_error))
#TPR=82%
lasso_error[2,2]/sum(lasso_error[,2])
#FPR=53%
(lasso_error[1,2])/(sum(lasso_error[,1]))
```
Then we found out that the TPR is 82%, the FPR is 53%




## confusion matrix and test error for ridge model
```{r}
error = table(pred=predsubsidy, true=dat.test$need_subsidy)
error
#test error
1-sum(diag(error))/(sum(error))

#TPR=82%
error[2,2]/sum(error[,2])
#FPR=53%
(error[1,2])/(sum(error[,1]))
```
Then we found out that the TPR is 82%, the FPR is 53%.




## Confusion matrix and test error for decision tree model
```{r}
err.pt.cv = table(pred.pt.cv,dat.test$need_subsidy)
err.pt.cv
#The test error
test_error<-1-sum(diag(err.pt.cv))/sum(err.pt.cv)
test_error

#TPR=99%
err.pt.cv[2,2]/sum(err.pt.cv[,2])
#FPR=3%
(err.pt.cv[1,2])/(sum(err.pt.cv[,1]))
```
Then we found out the TPR is 99%, and the FPR is 3%.
According to the TPR and FPR value of all the model we use, we get the highest TPR and lowest FPR in Decision Tree model. So the Decision Tree model is the best model for our project.



# Conlusion
In this final project, we also experienced some difficulties and made progress

## Key Finding

### key Finding 1
1.After we builed models associated with data set, we compare the test error initially. However, It is hard to gain the comparable test error for log regreesion model. Then after reaserching, we decided to compare TPR and FPR of each models. Then we get the final result. 

### key Finding 2:
2.TPR is important because it is about how credible the model we choose is. If the TPR is high enough, the government will need less money as reserve funds. And the FPR is important because it is about how much reserve funds the government can expect to set aside when adopting our model. This will allow the government to plan its finances more accurately.

### key Finding 3:
3.After building log regression model's AUC and ROC, we thought the model fits the data set perfectly since it's RUC is high. However, after we compared TPR and FPR of each models. We find out the decision tree model is better even thought log regression has high AUC We then conclued that decision tree model is the best fitting model among four models we tried.


## The final model we chose
In this final project, we tried four statistical models to predict and analyze the existing dataset. By comparing the TPR and FPR of these four models, we found that the tree decision model is the best model. Because it has the highest TPR and the lowest FPR. 

## what we find from the decison tree model
From part 5 we concluded a few main findings from decision tree model.We found that the capitalgain has the priority relation for the response, in our dataset if the capitalgain larger than 7073.5 then we predict that the person who has the capitalgain larger than 7073.5 most likely can not gain the subsidy.If the capitalgain less than 7073.5,we saw that the age has the secondly priority relation for the response, if the age is less than 28.5 then most likely can gain the subsidy, but if the age is larger than the 28.5 we will saw that the capitalloss have third priority.If the capitalloss is less than 1820.5 then most likely we can get the subsidy but if larger than 1820.5 then we will not get the subsidy.
