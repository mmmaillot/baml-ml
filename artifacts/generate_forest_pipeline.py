"""Independent sklearn random forest pipeline parity fixture (imputation/scaling)."""
import json
from pathlib import Path
import sys
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from export_sklearn import write_artifact

out = Path(sys.argv[1])
x = np.array([[-3.,0.],[-2.,1.],[-1.,2.],[0.,np.nan],[1.,4.],[2.,5.],[3.,6.]])
y = np.array([-3,-3,2,2,7,7,7])
p = make_pipeline(SimpleImputer(),StandardScaler(),RandomForestClassifier(n_estimators=5,max_depth=3,random_state=42)).fit(x,y)
features = ["a","b"]
rows = [[None,1.],[-3.,0.],[-0.5,2.],[1.,None],[3.,6.]]
test = np.array(rows,dtype=np.float64)
write_artifact(p,features,out/"forest-pipeline.json")
(out/"forest-pipeline-oracle.json").write_text(json.dumps(dict(input_json=json.dumps(dict(features=features,rows=rows)),labels=p.predict(test).tolist(),probs=p.predict_proba(test).tolist()),allow_nan=False))
