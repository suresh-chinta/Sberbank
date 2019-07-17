library(RJSONIO)
library(data.table)
library(stringi)
library(xgboost)
library(readr)
library(dummies)
library(text2vec)
library(SnowballC)
library(caret)
library(rBayesianOptimization)
library(Matrix)
library(zoo)
library(readr)
library(lubridate)
library(funModeling)
library(sqldf)
library(plyr)
library(dplyr)
setwd("G:\\R\\sberbank")
detach("package:plyr", unload=TRUE)
train <- fread('train.csv',stringsAsFactors = FALSE)
test <- fread('test.csv',stringsAsFactors = FALSE)

########### Under sampling to match the LB and CV #######################
re_investment = 
  train %>% 
  filter(product_type=='Investment',timestamp>='2011-10-01') %>% 
  group_by(ts=substring(timestamp,1,7)) %>% 
  summarise(n=n(),
            n1M=sum(ifelse(price_doc<=1000000,1,0))/n(),
            n2M=sum(ifelse(price_doc==2000000,1,0))/n(),
            n3M=sum(ifelse(price_doc==3000000,1,0))/n())

m1=floor(mean(re_investment$n1M[re_investment$ts>='2015-01'])/10*nrow(train)) #undersampling by magic numbers
m2=floor(mean(re_investment$n2M[re_investment$ts>='2015-01'])/3*nrow(train)) #undersampling by magic numbers
m3=floor(mean(re_investment$n3M[re_investment$ts>='2015-01'])/2*nrow(train)) #undersampling by magic numbers

set.seed(1)
i1 = train %>% filter(price_doc<=1000000,product_type=='Investment') %>% sample_n(m1)
i2 = train %>% filter(price_doc==2000000,product_type=='Investment') %>% sample_n(m2)
i3 = train %>% filter(price_doc==3000000,product_type=='Investment') %>% sample_n(m3)

train = train %>% filter(!(price_doc<=1000000 & product_type=='Investment'))
train = train %>% filter(!(price_doc==2000000 & product_type=='Investment'))
train = train %>% filter(!(price_doc==3000000 & product_type=='Investment'))

train = rbind(train,i1,i2,i3) %>% arrange(id)
setDT(train) 
###########################GDA BEGIN####################
test$price_doc = -99999
train_ids = train$id
train_test = rbind(train, test)

#### features based on population ######################## 
train_test$work_male_percent = round((train_test$work_male*100)/train_test$work_all)
train_test$work_female_percent = round((train_test$work_female*100)/train_test$work_all)

train_test$total_pop_m_per = round((train_test$male_f*100)/train_test$full_all)
train_test$total_pop_f_per = round((train_test$female_f*100)/train_test$full_all)

train = subset(train_test, train_test$id %in% train$id)
test = subset(train_test, !(train_test$id %in% train$id))
test$price_doc = NULL

################### PICK UP ONLY IMPORTANT FEATURES FROM MACRO ###################################

vifcols = c("balance_trade",                      
            "balance_trade_growth",               
            "eurrub",                             
            "average_provision_of_build_contract",
            "micex_rgbi_tr",
            "timestamp",
            "micex_cbi_tr",                       
            "deposits_rate",              
            "mortgage_value",                     
            "mortgage_rate",                      
            "income_per_cap",                     
            "museum_visitis_per_100_cap",         
            "apartment_build")

macro <- fread('macro.csv',stringsAsFactors = FALSE)
macro = macro[,vifcols, with=FALSE]

setkey(train,timestamp)
setkey(test,timestamp)
setkey(macro,timestamp)

train <- merge(train,macro,all.x=TRUE)
test <- merge(test,macro,all.x=TRUE)

###############################FEATURES BASED ON HOUSE PROPERTIES ###################

train <- train[,ratio_life_sq_full_sq:=life_sq/full_sq]
train <- train[,ratio_kitch_sq_life_sq:=kitch_sq/life_sq]
train <- train[,ratio_floor_max_floor:=floor/max_floor]
train <- train[,floor_from_top:=(max_floor-floor)]
train <- train[,extra_sq:=(full_sq - life_sq)]
train <- train[,age_of_building:=(build_year-year(timestamp))]

test <- test[,ratio_life_sq_full_sq:=life_sq/full_sq]
test <- test[,ratio_kitch_sq_life_sq:=kitch_sq/life_sq]
test <- test[,ratio_floor_max_floor:=floor/max_floor]
test <- test[,floor_from_top:=(max_floor-floor)]
test <- test[,extra_sq:=(full_sq - life_sq)]
test <- test[,age_of_building:=(build_year-year(timestamp))]

names(train) <- make.names(names(train))
names(test) <- names(train)[-292]
temp <- df_status(train)
features <- temp$variable[temp$type%in%c('character','integer')][-c(1,2,201)]

train$year <- year(train$timestamp)
train$month <- month(train$timestamp)
test$year <- year(test$timestamp)
test$month <- month(test$timestamp)

train[train=="yes"] = 1
train[train=="no"] = 0
train[train==TRUE] = 1
train[train==FALSE] = 0
train[is.na(train)] = -999

test[test=="yes"] = 1
test[test=="no"] = 0
test[test==TRUE] = 1
test[test==FALSE] = 0
test[is.na(test)] = -999

###################### NZV AND HIGH CORRELATED ###########################
# Remove near zero var columns, only 2 coulms are getting removed
#if (!(length(nzv_cols) > 0))
nzv_cols <- nearZeroVar(train,100)
train = train[,-nzv_cols,with=FALSE]
test  = test[,-nzv_cols,with=FALSE]

# Remove highly co-related variables, about 30 coulumns 
numericols = colnames(train)[sapply(train, is.numeric)]
numericCor <- cor(train[,numericols,with=FALSE])
numericCor[is.na(numericCor)] = 0.5 # some correlations are NA not sure why
highCorrs <- findCorrelation(numericCor, 0.88, names=T)

####################OUTLIERS##########################################

########## building year ####################################
### Building year has 0, 1, 2 values both in train and test
### assume 0, 1, 2 be number of years since construction

train[train$build_year==20052009, "build_year"] = 2005

train[build_year == 0 & year==2013, "build_year"] = 2013
train[build_year == 0 & year==2014, "build_year"] = 2014
train[build_year == 0 & year==2015, "build_year"] = 2015

train[build_year == 1 & year==2013, "build_year"] = 2012
train[build_year == 1 & year==2014, "build_year"] = 2013
train[build_year == 1 & year==2015, "build_year"] = 2014

train[build_year == 2 & year==2013, "build_year"] = 2011
train[build_year == 2 & year==2014, "build_year"] = 2012
train[build_year == 2 & year==2015, "build_year"] = 2013
train[build_year == 215, "build_year"] = 2015

### recalculate age of building
train$age_of_building <- train$year - train$build_year

test[build_year == 0 & year==2013, "build_year"] = 2013
test[build_year == 0 & year==2014, "build_year"] = 2014
test[build_year == 0 & year==2015, "build_year"] = 2015

test[build_year == 1 & year==2013, "build_year"] = 2012
test[build_year == 1 & year==2014, "build_year"] = 2013
test[build_year == 1 & year==2015, "build_year"] = 2014

test[build_year == 2 & year==2013, "build_year"] = 2011
test[build_year == 2 & year==2014, "build_year"] = 2012
test[build_year == 2 & year==2015, "build_year"] = 2013
test[build_year == 215, "build_year"] = 2015

test$age_of_building <- test$year - test$build_year

setdiff(colnames(test), colnames(train))

##############################################################
# round minutes and km to nearst integer
train$metro_min_avto = round(train$metro_min_avto)
train$nuclear_reactor_km = round(train$nuclear_reactor_km)

test$metro_min_avto = round(test$metro_min_avto)
test$nuclear_reactor_km = round(test$nuclear_reactor_km)
######### Correct Outliers In Number of Rooms #####################################
train[train$num_room>=5, "num_room"] = 5
train[train$sport_count_2000>=42, "sport_count_2000"] = 42

test[test$num_room>=5, "num_room"] = 5
test[test$sport_count_2000>=42, "sport_count_2000"] = 42

############## maximum floors is less than floor (floor_from_top < 0) #####################
### Assume max floors in incorrect and set it to NA

train[floor_from_top < 0, "max_floor"] = NA
train[floor_from_top < 0, "floor_from_top"] = NA

test[floor_from_top < 0, "max_floor"] = NA
test[floor_from_top < 0, "floor_from_top"] = NA

################# Zero values in area #####################
### Assume them to be incorrect values and set them to NAs

train[life_sq == 0, "life_sq"] = NA
train[full_sq == 0, "full_sq"] = NA

train[is.na(full_sq), "extra_sq"] = NA
train[is.na(life_sq), "extra_sq"] = NA

test[life_sq == 0, "life_sq"] = NA
test[full_sq == 0, "full_sq"] = NA

test[is.na(full_sq), "extra_sq"] = NA
test[is.na(life_sq), "extra_sq"] = NA

################# life_sq > full_sq ######################
#### Assume these values r swapped by mistake and reswap them

train_ids_error = train$id[train$life_sq > train$full_sq]
for (i in 1:length(train_ids_error))
{   
  
  temp = train[train$id == train_ids_error[i], "life_sq"]
  train[train$id == train_ids_error[i], "life_sq"] = train[train$id == train_ids_error[i], "full_sq"]
  train[train$id == train_ids_error[i],"full_sq"] = temp
}

train$extra_sq = train$full_sq - train$life_sq

test_ids_error = test$id[test$life_sq > test$full_sq]
for (i in 1:length(test_ids_error))
{   
  
  temp = test[test$id == test_ids_error[i], "life_sq"]
  test[test$id == test_ids_error[i], "life_sq"] = test[test$id == test_ids_error[i], "full_sq"]
  test[test$id == test_ids_error[i],"full_sq"] = temp
}

test$extra_sq = test$full_sq - test$life_sq

#### There are incorrect values in life_sq ##########################
train[train$life_sq=="1.0" & train$num_room >=2, "extra_sq"] = NA
train[train$life_sq=="1.0" & train$num_room >=2, "life_sq"] = NA
train[train$kitch_sq=="1.0", "kitch_sq"] = NA

test[test$life_sq=="1.0" & test$num_room >=2, "extra_sq"] = NA
test[test$life_sq=="1.0" & test$num_room >=2, "life_sq"] = NA
test[test$kitch_sq=="1.0", "kitch_sq"] = NA

train[is.na(train$life_sq), "ratio_life_sq_full_sq"] = NA
train[is.na(train$kitch_sq), "ratio_kitch_sq_life_sq"] = NA

test[is.na(test$life_sq), "ratio_life_sq_full_sq"] = NA
test[is.na(test$kitch_sq), "ratio_kitch_sq_life_sq"] = NA


################### Only one record with state of house as 33, others have 1,2,3, 4##############
train[train$state==33, "state"] = 3
#################desity of square km ######################
train$area_km = train$area_m/1000000
train$density = train$raion_popul/train$area_km
train$density = round(train$density)

test$area_km = test$area_m/1000000
test$density = test$raion_popul/test$area_km
test$density = round(test$density)
#############################################################
# Product type is NA (-999) for some rows in test, seeting it to Investment type
test[test$product_type==-999, "product_type"] = "Investment"
##############################################################
#### Create ratio variables for distance variables, found to be important in xgb
x = ncol(train)
for (j in 1:x)
{
  if(colnames(train[,j,with=FALSE]) %like% "_km" )
  {
    if (is.numeric(train[,j]))
    {
      colname = colnames(train[,j,with=FALSE])
      train[, paste0(colname, "_", "median_ratio") := train[[j]]/median(train[[j]],na.rm = TRUE)]
      test[, paste0(colname, "_", "median_ratio") := test[[j]]/median(test[[j]],na.rm = TRUE)]
    }
  }
}

setdiff(colnames(test), colnames(train))
setdiff(colnames(train), colnames(test))
##########################################################################
### Remove highly correlated columns, check for importance in xgb, before removing ########
impmat_new = read.csv("feat_imp_679_ration_counts_feat.csv")
highCorrs_x = setdiff(highCorrs, impmat_new$Feature)
highCorrs_x = setdiff(highCorrs_x, "id")


train <- train[, !names(train) %in% highCorrs_x, with=FALSE]
test <- test[, !names(test) %in% highCorrs_x, with=FALSE]

#### One hot encoding of categorical variables ##################
train <- dummy.data.frame(train, names=c("product_type", "ecology", "state"), sep="_")
test <- dummy.data.frame(test, names=c("product_type", "ecology", "state"), sep="_")

train <- dummy.data.frame(train, names=c("sub_area"), sep="_")
test <- dummy.data.frame(test, names=c("sub_area"), sep="_")

colnames(train)[which(!(colnames(train) %in% colnames(test)))]
setdiff(colnames(test), colnames(train))

##### How much the Euro Ruble exchange rate w.r.t median exchange rate
train$eurrub_x = train$eurrub/median(c(train$eurrub))
test$eurrub_x = test$eurrub/median(c(train$eurrub))

##### How much the mortgae lending rate w.r.t median exchange rate
train$rate_ratio_lend = train$mortgage_rate/median(c(train$mortgage_rate))
test$rate_ratio_lend = test$mortgage_rate/median(c(train$mortgage_rate))

#### Last 3 months average of couple of macro indicators,
######since impact of macro may not be immediate on house prices
#### only eurorub 3 months avaeage is found to be important, 
##### others are not important 

library(plyr)

macro_mortgage_rate = sqldf("select max(eurrub) eurrub_3months, year, month from macro_new group by year, month")
macro_mortgage_rate$average_eurrub_3months = 0
for (i in 1:nrow(macro_mortgage_rate))
{
  if (i == 1)
    macro_mortgage_rate$average_eurrub_3months[i] = macro_mortgage_rate$eurrub_3months[i]
  
  if (i == 2)
    macro_mortgage_rate$average_eurrub_3months[i] = (macro_mortgage_rate$eurrub_3months[i] + macro_mortgage_rate$eurrub_3months[i-1])/2
  
  if (i > 2)
    macro_mortgage_rate$average_eurrub_3months[i] = (macro_mortgage_rate$eurrub_3months[i] + macro_mortgage_rate$eurrub_3months[i-1] + macro_mortgage_rate$eurrub_3months[i-2])/3
  
  macro_mortgage_rate$average_eurrub_3months[i] = round(macro_mortgage_rate$average_eurrub_3months[i],2)
}
macro_mortgage_rate$eurrub_3months = NULL

trainbcp = train
train = join(train, macro_mortgage_rate)

testbcp = test
test = join(test, macro_mortgage_rate)

### Consumer price index median ratio
macro_new =fread('macro.csv',stringsAsFactors = FALSE)
macro_new$year = year(macro_new$timestamp)
macro_new$month = month(macro_new$timestamp)

macro_cpi = sqldf("select max(cpi) index_CPI,  year, month from macro_new group by year, month")
train = join(train, macro_cpi)
test = join(test, macro_cpi)
rm(macro_new)
gc()

###################### Add district Feature, and number of days since
######Ukraine crisis, which started in March 2014 ################
trainx <- fread('train.csv',stringsAsFactors = FALSE, select=c("id", "sub_area", "timestamp"))
testx <- fread('test.csv',stringsAsFactors = FALSE, select=c("id", "sub_area", "timestamp"))

trainx$timestamp = as.Date(trainx$timestamp, format = "%Y-%m-%d")
testx$timestamp = as.Date(testx$timestamp, format = "%Y-%m-%d")
cremiea_crisis_start = as.Date('3/14/2014',format='%m/%d/%Y')
trainx$time_since_CC = as.numeric(trainx$timestamp - cremiea_crisis_start)
testx$time_since_CC = as.numeric(testx$timestamp - cremiea_crisis_start)
train_testx = rbind(trainx,testx)
train_testx$sub_area = NULL
train_testx$timestamp = NULL
train = join(train, train_testx, by = "id", type="left")
test = join(test, train_testx, by = "id", type="left")

#### District features found to be not useful
#train <- dummy.data.frame(train, names=c("district"), sep="_")
#test <- dummy.data.frame(test, names=c("district"), sep="_")

rm(trainx)
rm(testx)
gc()
##################################################################
#### Euro exchange rate w.r.t the starting period i.e. 2011, not useful
#euroexratein2011 = 40.6651
#train$euroexratio = (train$eurrub/40.6651)
#test$euroexratio = (test$eurrub/40.6651)
########################## Convert characters to numerics ####################
for (i in 1:ncol(test))
{ 
  if (is.character(train[1,i]))
  {
    print(colnames(train)[i])
    test[,i] = as.numeric(test[,i])
    train[,i] = as.numeric(train[,i])
  }
}
###################################################################

train[is.na(train)] = -999
test[is.na(test)] = -999

train[train==NaN] = -999
train[is.na(train)] = -999
train[train==Inf] = -999

test[test==NaN] = -999
test[is.na(test)] = -999
test[test==Inf] = -999

y_train = log(train$price_doc  + 1)
train$price_doc = NULL
train$timestamp = NULL
test$timestamp = NULL

####################### PCA of cafe counts #################################
cafecols = c("cafe_count_1000_price_high",
             "cafe_count_500_price_500",
             "cafe_count_500_price_1000", 
             "cafe_count_500_price_1500", 
             "cafe_count_500_price_2500",
             "cafe_count_500_price_4000", 
             "cafe_count_500_price_high",
             "cafe_count_1000_na_price",
             "cafe_count_1000_price_1000",
             "cafe_count_1000_price_2500",
             "cafe_count_1000_price_4000",
             "cafe_count_1500_na_price", 
             "cafe_count_1500_price_500",
             "cafe_count_1500_price_1500",
             "cafe_count_1500_price_2500",
             "cafe_count_1500_price_4000",
             "cafe_count_1500_price_high",
             "cafe_count_2000_na_price", 
             "cafe_count_2000_price_500",
             "cafe_count_2000_price_1000",
             "cafe_count_2000_price_1500",
             "cafe_count_2000_price_2500",
             "cafe_count_2000_price_4000",
             "cafe_count_2000_price_high",
             "cafe_count_3000_na_price", 
             "cafe_count_3000_price_1000",
             "cafe_count_3000_price_1500",
             "cafe_count_3000_price_2500",
             "cafe_count_3000_price_high",
             "cafe_count_5000_na_price", 
             "cafe_count_5000_price_500",
             "cafe_count_5000_price_1000",
             "cafe_count_5000_price_1500",
             "cafe_count_5000_price_2500",
             "cafe_count_5000_price_4000",
             "cafe_count_5000_price_high")


train_test = rbind(train, test)
cafecolsx = intersect(cafecols, colnames(train_test))
train_test_coffee = train_test[,cafecolsx]
prc = prcomp(train_test_coffee, scale = TRUE)
xComponents = predict(prc, newdata = train_test_coffee)[,1:12]
colnames(xComponents) = paste(colnames(xComponents),"-1", sep = "")
cafecolsx = setdiff(colnames(train_test), cafecols)
train_test_rdi = train_test[,cafecolsx]
train_test_rdi = cbind(train_test_rdi, xComponents)
train = subset(train_test_rdi, train_test_rdi$id %in% train_ids)
test = subset(train_test_rdi, train_test_rdi$id %in% test_ids)
######################## PCA of other counts  #############################
cafecols <- c("big_church_count_5000",
              "church_count_5000",
              "leisure_count_5000",
              "mosque_count_5000",
              "sport_count_5000",
              "market_count_5000",
              "big_church_count_3000",
              "church_count_3000",
              "leisure_count_3000",
              "mosque_count_3000",
              "sport_count_3000",
              "market_count_3000",
              "big_church_count_2000",
              "church_count_2000",
              "leisure_count_2000",
              "mosque_count_2000",
              "sport_count_2000",
              "market_count_2000",
              "big_church_count_1000",
              "church_count_1000",
              "leisure_count_1000",
              "mosque_count_1000",
              "sport_count_1000",
              "market_count_1000",
              "big_church_count_500",
              "leisure_count_500",
              "mosque_count_500",
              "sport_count_500",
              "market_count_500",
              "big_church_count_1500",
              "church_count_1500",
              "leisure_count_1500",
              "mosque_count_1500",
              "sport_count_1500",
              "market_count_1500")


train_test = rbind(train, test)
cafecolsx = intersect(cafecols, colnames(train_test))
train_test_coffee = train_test[,cafecolsx]
prc = prcomp(train_test_coffee, scale = TRUE)
xComponents = predict(prc, newdata = train_test_coffee)[,1:21]
colnames(xComponents) = paste(colnames(xComponents),"-2", sep = "")
cafecolsx = setdiff(colnames(train_test), cafecols)
train_test_rdi = train_test[,cafecolsx]
train_test_rdi = cbind(train_test_rdi, xComponents)
train = subset(train_test_rdi, train_test_rdi$id %in% train_ids)
test = subset(train_test_rdi, train_test_rdi$id %in% test_ids)
##############################  Average CPI percentage over the last 3 months         ############################
macro_cpi = sqldf("select avg(cpi) index_avg, year, month from macro_new group by year, month")
macro_cpi$inflation_over_last3M = 0
for (i in 4:nrow(macro_cpi))
{
  print(i)
  macro_cpi$inflation_over_last3M[i] = (macro_cpi$index_avg[i] - macro_cpi$index_avg[i-3]) * 100/macro_cpi$index_avg[i-3]
}
macro_cpi[is.na(macro_cpi$inflation_over_last3M),"inflation_over_last3M"] = 6/4
macro_cpi$inflation_over_last3M[1:3] = 1.5
macro_cpi$inflation_over_last3M = round(macro_cpi$inflation_over_last3M, 2)
macro_cpi = macro_cpi[,c("year", "month", "inflation_over_last3M")]
train = join(train, macro_cpi)
test = join(test, macro_cpi)
##################################################################################
dtrain <- xgb.DMatrix(data=data.matrix(train), label=y_train)
dtest <- xgb.DMatrix(data=data.matrix(test))

train_ids = train$id
test_ids =  test$id

train$id = NULL
test$id = NULL



setdiff(colnames(test), colnames(train))
setdiff(colnames(train), colnames(test))
dim(dtrain) == dim(dtest)
dim(dtrain) == dim(train)
dim(test)   == dim(dtest)

gc()

# Params for xgboost
param <- list(objective="reg:linear",
              eval_metric = "rmse",
              eta = 0.01,
              #gamma = 1,
              alpha = 4,
              lambda = 4,
              max_depth = 4,
              min_child_weight = 1,
              subsample = .7,
              colsample_bytree = .7
)

kfolds<- 5
set.seed(2019)
folds<-createFolds(y_train, k = kfolds, list = TRUE, returnTrain = FALSE)
train$timestamp = NULL
test$timestamp = NULL
trainc$id = NULL
test$id = NULL
train$district = NULL
test$district = NULL
dtest <- xgb.DMatrix(data.matrix(test))
score  = c()
for(i in 1:5){
  fold <- as.numeric(unlist(folds[i]))
  x_train<-train[-fold,] #Train set
  x_val<-train[fold,] #Out of fold validation set
  y<-y_train[-fold]
  yv<-y_train[fold]
  dtrain = xgb.DMatrix(as.matrix(x_train), label=y)
  dval = xgb.DMatrix(as.matrix(x_val), label=yv)
  set.seed(2016)
  gbdt = xgb.train(params = param,
                   data = dtrain,
                   nrounds =8575,
                   watchlist = list(train = dtrain, val=dval),
                   print_every_n = 200,
                   early_stopping_rounds=150)
  score <- c(score,gbdt$best_score)
  if(i==1) allpredictions <- predict(gbdt,dtest) else allpredictions <- allpredictions + predict(gbdt,dtest)
  #print(length(allpredictions)/3)
  print(i)
  print(mean(score))
}


allpredictions_top_ = allpredictions
######################
##Generate Submission
allpredictions =  (as.data.frame(matrix(allpredictions_top_/5, nrow=dim(test), byrow=TRUE)))
preds <- exp(allpredictions) - 1
write.table(data.table(id=test_ids, price_doc=preds), "G:\\R\\sberbank\\model3.csv", sep=",", dec=".", quote=FALSE, row.names=FALSE)


#### Check importance of variables
nround = 256
bst = xgboost(param=param, print_every_n = 50, data = dtrain, nrounds=nround, importance=TRUE)
names <- dimnames(train)[[2]]
importance_matrix_new <- xgb.importance(names, model = bst)
write.csv(importance_matrix_new, row.names = FALSE, file="G:\\R\\sberbank\\feat_imp__484.csv")
#######################################################################
#### THINGS DID NOT WORK ##############################################

##### Adjust the prices for the years 2014, 2015
##### Target inflation rate (CPI) is 6%, but in the years 2014 it is 8.5
##### In the year 2015 it is 12.5, in the year 2016, inflation is back to
##### normal i.e around 6%, may be housing prices have increased in 2014, 2015
##### hence adjusting those prices to 6% inflation.

train$price_doc = NULL
for (i in 1:nrow(train))
{
  if (train[i,"year"] == 2014)
  {
    y_train[i] = y_train[i] - y_train[i]*0.025
  }
  
  if (train[i,"year"] == 2015)
  {
    y_train[i] = y_train[i] - y_train[i]*0.065
  }
}

# Adjusting the prices of houses in 2011, 2012, 2013, to what would be the price
# if they were sold in 2016, similar to Net Present Value
# if a house was sold for x rubles in 2012, it would be sold for x * 0.24 in 2016
# as the inflation over 4 years from 2012 to 2016 will be 24%
for (i in 1:nrow(train))
{
  if (train[i,"year"] == 2011)
  {
    y_train[i] = y_train[i] + y_train[i]*0.3
  }
  
  if (train[i,"year"] == 2012)
  {
    y_train[i] = y_train[i] + y_train[i]*0.24
  }
  
  if (train[i,"year"] == 2013)
  {
    y_train[i] = y_train[i] + y_train[i]*0.18
  }
}

#macro_mortgage_rate = sqldf("select max(mortgage_rate) rate_mortgage, year, month from macro_new group by year, month")
#macro_mortgage_rate$average_rate_3months = 0
#for (i in 1:nrow(macro_mortgage_rate))
#{
#  if (i == 1)
#    macro_mortgage_rate$average_rate_3months[i] = macro_mortgage_rate$rate_mortgage[i]

#  if (i == 2)
#    macro_mortgage_rate$average_rate_3months[i] = (macro_mortgage_rate$rate_mortgage[i] + macro_mortgage_rate$rate_mortgage[i-1])/2

#  if (i > 2)
#    macro_mortgage_rate$average_rate_3months[i] = (macro_mortgage_rate$rate_mortgage[i] + macro_mortgage_rate$rate_mortgage[i-1] + macro_mortgage_rate$rate_mortgage[i-2])/3

#  macro_mortgage_rate$average_rate_3months[i] = round(macro_mortgage_rate$average_rate_3months[i],2)
#}
#macro_mortgage_rate$rate_mortgage = NULL

#macro_mortgage_rate = sqldf("select median(micex_rgbi_tr) micex_rgbi_tr_3months, year, month from macro_new group by year, month")
#macro_mortgage_rate$average_rgbi_3months = 0
#for (i in 1:nrow(macro_mortgage_rate))
#{
#  if (i == 1)
#    macro_mortgage_rate$average_rgbi_3months[i] = macro_mortgage_rate$micex_rgbi_tr_3months[i]

#  if (i == 2)
#    macro_mortgage_rate$average_rgbi_3months[i] = (macro_mortgage_rate$micex_rgbi_tr_3months[i] + macro_mortgage_rate$micex_rgbi_tr_3months[i-1])/2

#  if (i > 2)
#    macro_mortgage_rate$average_rgbi_3months[i] = (macro_mortgage_rate$micex_rgbi_tr_3months[i] + macro_mortgage_rate$micex_rgbi_tr_3months[i-1] + macro_mortgage_rate$micex_rgbi_tr_3months[i-2])/3

#  macro_mortgage_rate$average_rgbi_3months[i] = round(macro_mortgage_rate$average_rgbi_3months[i],2)
#}
#write.csv(macro_mortgage_rate, "G:\\R\\sberbank\\micex_rgbi_tr.csv")
#macro_mortgage_rate$micex_rgbi_tr_3months = NULL

#trainbcp = train
#train = join(train, macro_mortgage_rate)

#testbcp = test
#test = join(test, macro_mortgage_rate)

#macro_mortgage_rate = sqldf("select median(deposits_rate) deposits_rate_avg, year, month from macro_new group by year, month")
#macro_mortgage_rate$deposits_rate_3months = 0
#for (i in 1:nrow(macro_mortgage_rate))
#{
#if (i == 1)
#macro_mortgage_rate$deposits_rate_3months[i] = macro_mortgage_rate$deposits_rate_avg[i]

#if (i == 2)
#macro_mortgage_rate$deposits_rate_3months[i] = (macro_mortgage_rate$deposits_rate_avg[i] + macro_mortgage_rate$deposits_rate_avg[i-1])/2

#if (i > 2)
#macro_mortgage_rate$deposits_rate_3months[i] = (macro_mortgage_rate$deposits_rate_avg[i] + macro_mortgage_rate$deposits_rate_avg[i-1] + macro_mortgage_rate$deposits_rate_avg[i-2])/3

#macro_mortgage_rate$deposits_rate_3months[i] = round(macro_mortgage_rate$deposits_rate_3months[i],2)
#}
#write.csv(macro_mortgage_rate, "G:\\R\\sberbank\\deposits_rate_avg.csv")
#macro_mortgage_rate$deposits_rate_avg = NULL

#trainbcp = train
#train = join(train, macro_mortgage_rate)

#testbcp = test
#test = join(test, macro_mortgage_rate)

#median_cpi = median(c(macro_cpi$index_CPI), na.rm = TRUE)
#mean_cpi = mean(c(macro_cpi$index_CPI), na.rm = TRUE)
#mean_since_1993 = 205

#train$cpi_median_ratio = train$index_CPI/median_cpi
#train$cpi_mean_ratio = train$index_CPI/mean_cpi
#train$cpi_mean_19_ratio = train$index_CPI/mean_since_1993

#test$cpi_median_ratio = test$index_CPI/median_cpi
#test$cpi_mean_ratio = test$index_CPI/mean_cpi
#test$cpi_mean_19_ratio = test$index_CPI/mean_since_1993


###### Things to try##################################################

#1) remove data of 2011 as it has lot of NA
#2) develop different models with
 #    a) Full data, with out filtering to match CV LB
#     b) Without macro features a model
 #    c) Seperate models for Investment type and Owner occupied type
#     d) Create combination of categorical variables
#     e) models with data of only 2014, 2015
#     f) H2o models
#     g) GLMNET models
#     e) handling of Missing values
     
##### KEY EVENTS ########################################
#1) Market is stable until end of 2013,
#2) Depreciation of ruble in 2014 JAN, because of fall of oil prices
#3) Russia occupation of UKRAINE in 2014 March and subsequent sanctions by EU and US
#4) Ruble devaluation, govt allowed devaluation, to get more money 
    # for oil exports as oil prices are dropping
#5) High inflation from 2014 Jan to end 2015 about 12%
#6) Russian Government identifies Real estate as the only driver 
##7) for economic growth, hence subsidizing housing and encouraging real estate





