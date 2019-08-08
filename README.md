# modelProducers
> Please do not directly develop in the master branch.
> We should create a branch, for example, sean/add_xxxPattern,
> for developing. We than discuss and merge back to the master branch.

### labeling, featuring, ML-modeling 的初步嘗試
使用技術指標([talib](https://github.com/mrjbq7/ta-lib))其中的23種來做featuring。 以`pct_change(period).sign()`來做labeling。使用簡單的PCA+random forest 來做ML-Model。其輸出可直接於stgyProducer/sean/master 做策略(dictionary形式)產出使用。
