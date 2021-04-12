# performance analysis
load("performance_random_new.Rdata")
library(xtable)
library(dplyr)
library(ggplot2)
performance = performance_random

# result by iteration
plotdata_iter = performance %>%
  mutate(P=FN+TP,N=FP+TN) %>%
  mutate(precision=(TP/(TP+FP)), recall=(TP/(TP+FN)), 
         TSS=(TP/(TP+FN)-FP/(FP+TN)),
         HSS=(2*(TP*TN-FN*FP)/(P*(FN+TN)+N*(TP+FP))))

# test model differences
compare_model = NULL
compare_set = c("T","P","SPIL+T","SPIL+P",
                "full_PIL_noO","full_PIL_PC_noO","full_PIL","full_PIL_PC")

for (h in c(1,6,12,24)){
  hour_result = c()
  benchmark = plotdata_iter %>% filter(hour==h, fset=="SPIL") %>% select(TSS) %>% unlist()
  for (features in compare_set){
    to_compare = plotdata_iter %>% filter(hour==h, fset==features) %>% select(TSS) %>% unlist()
    tst = t.test(benchmark, to_compare, paired = T, alternative = "less")
    hour_result = c(hour_result, tst$p.value)
  }
  compare_model = rbind(compare_model, hour_result)
}
rownames(compare_model) = c("1","6","12","24")
colnames(compare_model) = compare_set
# xtable(compare_model,digits=3)

perf_all = performance %>%
  mutate(P=FN+TP,N=FP+TN) %>%
  mutate(precision=(TP/(TP+FP)), recall=(TP/(TP+FN)),
         TSS=(TP/(TP+FN)-FP/(FP+TN)),
         HSS=(2*(TP*TN-FN*FP)/(P*(FN+TN)+N*(TP+FP)))) %>%
  select(hour,fset,TSS,HSS) %>%
  group_by(hour,fset) %>%
  summarise_all(list(mean,sd))


# format the table for output
plotdata = perf_all %>% select(fset, hour, TSS_fn1) %>% rename(TSS = TSS_fn1)
plotdata$TSS = round(plotdata$TSS, digits = 3)
plotdata = spread(plotdata, key = hour, value = TSS)
print(xtable(plotdata,digits=3))

# variable selection
f_score = function(col_num,data){
  b = data[data$fclass=="B",col_num]
  m = data[data$fclass=="M",col_num]
  bmean = mean(b)
  mmean = mean(m)
  allmean = mean(data[,col_num])
  nb = sum(data$fclass=="B")
  nm = sum(data$fclass=="M")
  num = (bmean-allmean)^2 + (mmean-allmean)^2
  denom = sum((b-allmean)^2)/(nb-1) + sum((m-allmean)^2)/(nm-1)
  if(denom==0){
    return(0)
  }
  return(num/denom)
}

score = NULL
cordata = list()

for (h in c(1,6,12,24)){
  print(h)
  sub_data = data[data$hour==h,]
  tt = ttsplit(sub_data,train_size = 0.995)
  tt = d_preprocess(tt)
  sub_data = tt$train
  D = data.frame(cbind(sub_data$fclass,sub_data[,c(spil_feature,t_feature,sp_feature,t_pc_feature,sp_pc_feature,o_feature)]))
  colnames(D)[1] = "fclass"
  D[is.na(D)] = 0
  # D_score = fscore(Data=D,classCol = 1, featureCol = 2:ncol(D),silent = T)
  D_score = sapply(2:ncol(D),f_score,data=D)
  D_score = D_score/max(D_score)
  score = rbind(score,D_score)
}
score = data.frame(score)
names(score) = c(spil_feature,t_feature,sp_feature,t_pc_feature,sp_pc_feature,o_feature)
write.csv(score, "Fscore_new.csv")

score = read.csv(file="Fscore_new.csv")
score$X = NULL
ns = names(score)
top_score = NULL
hour = c(1,6,12,24)
feature_compare = c(spil_feature,t_pc_feature,sp_pc_feature,o_feature)
score = score[,feature_compare]
score = apply(score,1,function(x) x/max(x))
score = t(score)
for (i in 1:nrow(score)){
  s = score[i,]
  s = sort(s, decreasing = T)
  for (j in 1:15){
    n = names(s)[j]
    if (grepl("SHARP",n)){
      type = "S"
    }else if(grepl("t_PC",n)){
      type = "T_PC"
    }else if(grepl("Ripley",n)){
      type = "Ripley_K_PC"
    }else if(grepl("Vario",n)){
      type = "V-gram"
    }else{
      type = "A"
    }
    entry = c(n,type,as.character(hour[i]),
              as.character(s[j]),as.character(j))
    top_score = rbind(top_score, entry)
  }
}

top_score = data.frame(top_score)
colnames(top_score) = c("Variable","Type","Hour","Score","Rank")
top_score$Type = factor(top_score$Type)
top_score$Hour = factor(top_score$Hour,
                        levels=c("1","6","12","24"),ordered = T)
top_score$Score = as.numeric(top_score$Score)
top_score$Rank = as.numeric(top_score$Rank)

f_plot = ggplot(data = top_score) +
  geom_point(aes(x=Rank,y=Score,color=Type)) + 
  theme_bw() + 
  ylab("F-score") + 
  xlab("Rank") +
  facet_wrap(~Hour) +
  labs(color="Feature")

f_plot
ggsave("F_score_new.png",width=12,height=8,units="cm")


# scatter plot
ndata = NULL
for (h in c(1,6,12,24)){
  subdata = data[data$hour==h,]
  tt = ttsplit(subdata,train_size=0.995)
  subdata = d_preprocess(tt)$train
  if (h==1){
    ndata = subdata
  }else{
    ndata = rbind(ndata,subdata)
  }
}

p = ggplot(data = ndata, aes(x = Ripley4_PC1, y = Ripley5_PC1, color = fclass)) +
  geom_point(alpha = 0.3, size = 0.5) + 
  facet_wrap(~hour)

p
ggsave("PC_2d_new.png",width=12,height=8,units="cm")

# visualize the functional PC
efunc = c()

for (h in c(1,6,12,24)){
  subdata = data[data$hour==h,]
  feature = paste("Ripley",4,"_",seq(1,100,5),sep="")
  dmat = subdata[,feature]
  L3 = MakeFPCAInputs(IDs = rep(1:nrow(subdata), each=20),
                      tVec = rep(seq(1,100,5), nrow(subdata)), yVec = as.matrix(dmat))
  FPCAdense = FPCA(L3$Ly, L3$Lt)
  ef = FPCAdense$phi[,1]
  efunc = c(efunc, ef)
}

x1 = data %>% 
  drop_na() %>%
  group_by(fclass,hour) %>% 
  summarise_at(vars(starts_with("Ripley9")),mean)

x = data.frame(R4 = as.numeric(t(x1[,3:102])), hour = rep(c(1,6,12,24,1,6,12,24),each=100), flare = rep(c("B","M"),each=400))
px = ggplot(data = x) +
  geom_line(aes(x = rep(1:100,8), y = R4, color = flare)) + 
  facet_wrap(~hour) + 
  xlab("Distance") +
  ylab("Ripley's K at 2000 G") +
  theme_bw() +
  labs(color = "flare type") +
  scale_color_discrete(label=c("B","M/X"))
px
ggsave("ripK_2000_new.png",width=12,height=8,units="cm")

# mean()
# subdata[subdata$fclass=="B","Ripley4_100"])
D = data %>% 
  drop_na()
x1 = D %>%
  select_at(vars(starts_with("Vario") & ends_with("_2")))
x1$fclass = D$fclass
x1$hour = D$hour

ggplot(data=x1,aes(color=fclass,fill=fclass)) +
  geom_density(aes(x=Vario9_2),alpha=0.2) +
  facet_wrap(~hour)
  
x1 = D %>% group_by(fclass,hour) %>% 
  summarise_at(vars(starts_with("Vario") & ends_with("_2")),mean)

x1[,3:13] = round(x1[,3:13],digits = 3)
print(xtable(x1[,c(1,2,11:13)], digits = 3))




px = ggplot(data = data) +
  geom_point(aes(x = Vario4_2, y = Vario7_2, color = fclass)) + 
  facet_wrap(~hour) + 
  xlab("Variogram Sill at 800 G") +
  ylab("Variogram Sill at 1800 G")
px
