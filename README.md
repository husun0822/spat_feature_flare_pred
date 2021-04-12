##  Solar Flare Prediction with Features based on TDA and Spatial Statistics

### Structure of files:

     .
     ├── Bflare_data                       
     │   └── B_flare_1h.hdf5               # an example file of a B flare from HARP 377, used for paper figures
     ├── Mflare_data                       
     │   └── M_flare_1h.hdf5               # an example file of an M flare from HARP 377, used for paper figures
     ├── code
     │   ├── Analysis_Code (R)             
     │       ├── combine_data.R            # combine spatial, topological, SHARP parameters data into a single .RData and .csv file
     │       ├── model_func.R              # define the XGboost classification model
     │       ├── model_compare.R           # fit XGboost on all combinations of features
     │       └── spat_feature_compare.R    # calculate TSS, plot average Ripley's K, average Variogram parameters
     │   ├── Data_Construct                
     │       ├── PIL_GEN.py                # python functions for detecting PIL regions, based on method by Schri
     
     
     │   ├── Feature_Construct             # construct topology features, spatial statistics features, calculate threshold
     │   ├── Paper_Figure                  # plot figures used in paper
     ├── data                              # data of the derived features, GOES dataset and quantile thresholds
     
