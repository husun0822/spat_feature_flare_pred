## combine spatial statistics features for all data

path = "../spat_feature/ten_iteration_data/new_feature/"
full_data = NULL

# combine data
for (h in c(1,6,12,24)){
  for (flare in c("B","M")){
    df = read.csv(paste(paste(path,flare,h,".csv",sep="")))
    df$fclass = flare
    df$hour = h
    full_data = rbind(full_data,df)
  }
}

full_data[full_data==-1] = 0
write.csv(full_data,"spat_feature_new.csv")


## combine all features into one dataset
library(tidyverse)
library(dplyr)

sdata = read.csv("spat_feature_new.csv")
tdata = read.csv("tgs_feature.csv")

# names of features
# Get variable names
feature = c("Br", 'MEANGAM', 'MEANGBT', 'MEANGBH', 'MEANGBZ', 
            'TOTUSJZ', 'TOTUSJH', 'MEANPOT', 'MEANSHR')

sharp = c('USFLUX', 'MEANGAM', 'MEANGBT', 'MEANGBH', 'MEANGBZ',
          'MEANJZD', 'TOTUSJZ', 'MEANALP', 'TOTUSJH','SAVNCPP',
          'MEANPOT', 'MEANSHR')

t_feature = c()
g_feature = c()
s_feature = c()
sp_feature = c()
spil_feature = c()

# other features
info_feature = c("HARP","Intensity","Time","fclass","hour")
o_feature = c("NPIL","areaPIL","width","height",paste("Npts",c(".1",0:9),sep=""))

for (ch in feature){
  for (level in seq(5,95,5)){
    t_feature = c(t_feature, paste(ch,level,sep=""))
  }
  for (level in c(50,90)){
    for (clust in 0:4){
      g_feature = c(g_feature,paste(ch,level,"_G",clust,sep=""))
    }
    g_feature = c(g_feature,paste(ch,level,"_MAXSIZE",sep=""))
    g_feature = c(g_feature,paste(ch,level,"_MEANSIZE",sep=""))
  }
}

for (ch in sharp){
  s_feature = c(s_feature,paste("SHARP_",ch,sep=""))
  spil_feature = c(spil_feature, paste("SHARP_",ch,"_PIL",sep=""))
}

ripley = c()
vario = c()
for (i in c(".1",0:9)){
  feature = paste("Ripley",i,"_",1:100,sep="")
  ripley = c(ripley, feature)
  feature = paste("Vario",i,"_",1:2,sep="")
  vario = c(vario, feature)
}
sp_feature = c(ripley,vario)

sp_pc_feature = c()
t_pc_feature = c()
g_pc_feature = c()

for (i in c(".1",0:9)){
  sp_pc_feature = c(sp_pc_feature, paste("Ripley",i,"_PC",1:5,sep=""))
}

for (i in c(".1",0:9)){
  sp_pc_feature = c(sp_pc_feature, paste("Vario",i,"_",1:2,sep=""))
}

channels = c("Br", 'MEANGAM', 'MEANGBT', 'MEANGBH', 'MEANGBZ', 'TOTUSJZ', 'TOTUSJH', 'MEANPOT', 'MEANSHR')
for (ch in channels){
  t_pc_feature = c(t_pc_feature, paste(ch,"_t_PC",1:5,sep=""))
  g_pc_feature = c(g_pc_feature, paste(ch,"_g_PC",1:5,sep=""))
}

names(sdata)[3] = "Intensity"
tdata$HARP = paste("HARP",tdata$HARP,sep="")
sdata_select = sdata[,c(info_feature,s_feature,sp_feature,o_feature)]
tdata_select = tdata[,c(info_feature,t_feature,g_feature)]

# load the SHARP PIL data
pdata = NULL
for (flare in c("B","M")){
  for (h in c(1,6,12,24)){
    df = read.csv(paste("../spat_feature/",flare,h,"_SHARP_PIL.csv",sep=""))
    df$hour = as.integer(h)
    df$intensity = as.integer(df$intensity)
    pdata = rbind(pdata,df)
  }
}
pdata$X = NULL
colnames(pdata) = c("Intensity","fclass","HARP","Time",spil_feature,"hour")
pdata_select = pdata[,c(info_feature,spil_feature)]

sdata_select = sdata_select %>%
  arrange(hour, HARP, Intensity, Time)
pdata_select = pdata_select %>%
  arrange(hour, HARP, Intensity, Time)

data = cbind(sdata_select,pdata_select[,spil_feature])
data = inner_join(data, tdata_select, by=info_feature)
data[,ripley] = data[,ripley]/data$NPIL
data = drop_na(data)

save(data,t_feature,g_feature,s_feature,sp_feature,info_feature,o_feature, spil_feature,
      sp_pc_feature,t_pc_feature,g_pc_feature,file = "data_new.Rdata")
write.csv(data,"data_new.csv")
