##  Solar Flare Prediction with Features based on TDA and Spatial Statistics

### Structure of files:

     .
     ├── Bflare_data                       
     │   ├── B_flare_1h.hdf5               # an example file of a B flare from HARP 377, used for paper figures
     ├── code
     │   ├── Analysis_Code (R)             # fit prediction model, and post-hoc analysis
     │   │   ├── combine_data.R
     
     
     │   ├── Data_Construct                # query data from JSOC, detect PIL and derive SHARP maps
     │   ├── Feature_Construct             # construct topology features, spatial statistics features, calculate threshold
     │   ├── Paper_Figure                  # plot figures used in paper
     ├── data                              # data of the derived features, GOES dataset and quantile thresholds
     ├── Mflare_data                       
     └── └── M_flare_1h.hdf5               # an example file of an M flare from HARP 377, used for paper figures
