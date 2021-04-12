library(fdapace)
load("data_new.Rdata")
source("model_func.R")

# train-test split
ttsplit = function(data, train_size=0.7, rand_seed=1, byharp=F){
  set.seed(rand_seed)
  if(!byharp){
    M_idx = which(data$fclass=="M")
    B_idx = which(data$fclass=="B")
    train_idx = c(sample(M_idx,
                         size = floor(train_size*length(M_idx)),replace = F),
                  sample(B_idx,
                         size = floor(train_size*length(B_idx)),replace = F)
    )
    D_train = data[train_idx,]
    D_test = data[-train_idx,]
  }else{
    harps = unique(data$HARP)
    train = which(runif(n=length(harps)) < train_size)
    train_harp = harps[train] # train set
    test_harp = harps[-train] # test set
    D_train = data[data$HARP %in% train_harp,]
    D_test = data[data$HARP %in% test_harp,]
  }
  return(list("train"=D_train,"test"=D_test))
}

# some data for testing code
subdata = data[data$hour==1,]
tt = ttsplit(subdata)
h = 1
grp = 0

# data-preprocessing
fpca_score_gen = function(tt, grp=0, K=5){
  train_data = tt$train
  test_data = tt$test
  time_point = seq(1,100,5)
  feature = paste("Ripley",grp,"_",seq(1,100,5),sep="")
  dmat = train_data[,feature]
  test_dmat = test_data[,feature]
  L3 = MakeFPCAInputs(IDs = rep(1:nrow(train_data), each=20),
                      tVec = rep(time_point, nrow(train_data)), yVec = as.matrix(dmat))
  test_L3 = MakeFPCAInputs(IDs = rep(1:nrow(test_data), each=20),
                           tVec = rep(time_point, nrow(test_data)), as.matrix(test_dmat))
  FPCAdense = FPCA(L3$Ly, L3$Lt)
  pc_score = as.data.frame(FPCAdense$xiEst)[,1:5]
  colnames(pc_score) = paste("Ripley",grp,"_PC",1:5,sep="")
  pc_score$Intensity = train_data$Intensity
  pc_score$HARP = train_data$HARP
  pc_score$hour = train_data$hour
  pc_score$Time = train_data$Time
  tt$train = inner_join(tt$train,pc_score,by=c("Intensity","HARP",
                                               "hour","Time"))
  
  pc_score_test = predict(object = FPCAdense, newLy = test_L3$Ly, newLt = test_L3$Lt, K=5)
  pc_score_test = as.data.frame(pc_score_test$scores)
  colnames(pc_score_test) = paste("Ripley",grp,"_PC",1:5,sep="")
  pc_score_test$Intensity = test_data$Intensity
  pc_score_test$HARP = test_data$HARP
  pc_score_test$hour = test_data$hour
  pc_score_test$Time = test_data$Time
  tt$test = inner_join(tt$test,pc_score_test,by=c("Intensity","HARP",
                                              "hour","Time"))
  return(tt)
}

d_preprocess = function(tt, t_pc=5, g_pc=5, sp_pc=5){
  # do fpca on ripley's K function
  for (i in c(".1",0:9)){
    tt = fpca_score_gen(tt, grp = i, K = sp_pc)
  }
  all_feature = c(s_feature, g_feature, t_feature, sp_feature, o_feature, spil_feature)
  train = tt$train
  test = tt$test
  train_feature = train[,all_feature]
  test_feature = test[,all_feature]
  
  # log-transform of train/test geometry features
  train_feature[,c(g_feature)] = log10(train_feature[,c(g_feature)]+1e-2)
  test_feature[,c(g_feature)] = log10(test_feature[,c(g_feature)]+1e-2)
  
  # normalize train and test set
  train_means = sapply(train_feature,mean)
  train_sd = sapply(train_feature, sd)
  train_feature = as.data.frame(t(apply(train_feature,1,function(x) (x-train_means)/train_sd)))
  test_feature = as.data.frame(t(apply(test_feature,1,function(x) (x-train_means)/train_sd)))
  
  channels = c("Br", 'MEANGAM', 'MEANGBT', 'MEANGBH', 'MEANGBZ', 'TOTUSJZ', 'TOTUSJH', 'MEANPOT', 'MEANSHR')
  t_train_PCA = NULL
  t_test_PCA = NULL
  t_pc_cumvar = c()
  t_pc_loading = NULL
  
  for (ch in channels){
    ch_t_feature = t_feature[grepl(ch,t_feature)]
    
    # PCA for the SHARP parameter's topology feature
    t_feature_pca = prcomp(x=train_feature[,ch_t_feature], center = FALSE, scale = FALSE)
    t_train_pca = t_feature_pca$x[,1:t_pc]
    t_test_pca = as.matrix(test_feature[,ch_t_feature]) %*% as.matrix(t_feature_pca$rotation)[,1:t_pc]
    colnames(t_train_pca) = sapply(1:t_pc,function(x) paste(ch,"_t_PC",x,sep=""))
    # t_test_pca = as.data.frame(t_test_pca)
    colnames(t_test_pca) = sapply(1:t_pc,function(x) paste(ch,"_t_PC",x,sep=""))
    t_train_PCA = cbind(t_train_PCA, t_train_pca)
    t_test_PCA = cbind(t_test_PCA, t_test_pca)
    # t_pc_cumvar = c(t_pc_cumvar, summary(t_feature_pca)$importance[3,t_pc])
    # loading = t(t_feature_pca$rotation[,1:t_pc])
    # colnames(loading) = ch_t_feature
    # t_pc_loading = cbind(t_pc_loading,loading)
  }
  
  # get the geometry feature
  g_train_PCA = NULL
  g_test_PCA = NULL
  g_pc_cumvar = c()
  g_pc_loading = NULL
  
  for (ch in channels){
    ch_g_feature = g_feature[grepl(ch,g_feature)]
    
    # PCA for the SHARP parameter's geometry feature
    g_feature_pca = prcomp(x=train_feature[,ch_g_feature],center = FALSE, scale = FALSE)
    g_train_pca = g_feature_pca$x[,1:g_pc]
    g_test_pca = as.matrix(test_feature[,ch_g_feature]) %*% as.matrix(g_feature_pca$rotation)[,1:g_pc]
    colnames(g_train_pca) = sapply(1:g_pc,function(x) paste(ch,"_g_PC",x,sep=""))
    # g_test_pca = as.data.frame(g_test_pca)
    colnames(g_test_pca) = sapply(1:g_pc,function(x) paste(ch,"_g_PC",x,sep=""))
    g_train_PCA = cbind(g_train_PCA, g_train_pca)
    g_test_PCA = cbind(g_test_PCA, g_test_pca)
    # g_pc_cumvar = c(g_pc_cumvar, summary(g_feature_pca)$importance[3,g_pc])
    # loading = t(g_feature_pca$rotation[,1:g_pc])
    # colnames(loading) = ch_g_feature
    # g_pc_loading = cbind(g_pc_loading,loading)
  }
  t_train_PCA = data.frame(t_train_PCA)
  g_train_PCA = data.frame(g_train_PCA)
  t_test_PCA = data.frame(t_test_PCA)
  g_test_PCA = data.frame(g_test_PCA)
  
  tt$train = cbind(tt$train[,info_feature],train_feature, t_train_PCA, g_train_PCA, tt$train[,sp_pc_feature[1:55]])
  tt$test = cbind(tt$test[,info_feature],test_feature, t_test_PCA, g_test_PCA, tt$test[,sp_pc_feature[1:55]])
  # colnames(train_data)[1] = "fclass"
  # colnames(test_data)[1] = "fclass"
  return(tt)
}

# test code
tt = d_preprocess(tt)

# create a list of model
model_list = list("S"=s_feature, "T"=t_feature, "G"=g_feature, "P"=sp_feature, "SPIL"=spil_feature,
                               "S+T"=c(s_feature,t_feature),
                               "S+G"=c(s_feature,g_feature),
                               "S+P"=c(s_feature,sp_feature),
                               "SPIL+T"=c(spil_feature,t_feature),
                               "SPIL+G"=c(spil_feature,g_feature),
                               "SPIL+P"=c(spil_feature,sp_feature),
                               "full" = c(s_feature,t_feature,sp_feature,o_feature),
                               "full_PC" = c(s_feature, t_pc_feature, sp_pc_feature, o_feature),
                               "full_PIL" = c(spil_feature,t_feature, sp_feature,o_feature),
                               "full_PIL_PC" = c(spil_feature, t_pc_feature, sp_pc_feature, o_feature),
                               "full_PIL_noO" = c(spil_feature,t_feature, sp_feature),
                               "full_PIL_PC_noO" = c(spil_feature, t_pc_feature, sp_pc_feature))

# model fitting
# clean the data
D = data %>% group_by(Intensity, HARP, Time) %>%
  add_tally() %>%
  filter(n==4)
data = D
data$fclass = as.factor(data$fclass)

performance_random = NULL
split_harp = T
for (h in c(1,6,12,24)){
  sub_data = data[data$hour==h,]
  
  # run 20 iterations
  for (i in 1:20){
    print(paste("hour:",h,"iteration:",i))
    tt = ttsplit(sub_data, rand_seed = i, byharp = split_harp)
    tt = d_preprocess(tt)
    
    for (feature_set in names(model_list)){
      cm_XGB = GBoost(tt, feature = model_list[[feature_set]])
      xgmodel = cm_XGB$xgmodel
      df = cm_XGB$cm_stat
      df$hour = h
      df$fset = feature_set
      df$idx = i
      performance_random = rbind(performance_random,df)
    }
  }
}

save(performance_random,file = "performance_random_new.Rdata")
