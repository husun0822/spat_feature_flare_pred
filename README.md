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
     │       ├── Data_Download.py          # python code for querying the snapshot datasets from JSOC (large dataset, can take long to run)
     │       ├── PIL_GEN.py                # python functions for detecting PIL regions, based on method by Schri
     │       ├── Potential_Field_Calculation.py  # python functions for calculating potential field
     │       └── SHARP_map.py              # calculate the 2D SHARP maps
     │   ├── Feature_Construct             # construct topology features, spatial statistics features, calculate threshold
     │       ├── Cubic_Complex_Feature.py  # derive topological features
     │       ├── Spat_Feature.py           # derive spatial statistics features
     │       ├── SHARP threshold.py        # calculating thresholds of SHARP parameters, at pixel level
     │       └── SHARP_PIL.py              # calculating SHARP parameters at PIL
     │   ├── Paper_Figure                  # jupyter notebooks plotting figures used in paper
     ├── data                              # data of the derived features, GOES dataset and quantile thresholds
     │   ├── SHARP_PIL                     # data of SHARP parameters along PIL
     │   ├── Spatial_Feature               # data of spatial statistics features
     │   ├── Topology_Feature              # data of topological features
     │   └── data_new.Rdata                # data ready to be used for analysis
     
