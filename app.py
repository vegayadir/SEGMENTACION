from flask import Flask, jsonify,request
import pandas as pd
from os import * 
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min 
from kneed import KneeLocator

class segmentacion:

    def get_clusters_numbers_SSD(self,x_var):
        Sum_of_squared_distances = []
        K = range(1,15)
        for k in K:
            km = KMeans(n_clusters=k)
            km = km.fit(x_var)
            Sum_of_squared_distances.append(km.inertia_)
        x = range(1, len(Sum_of_squared_distances)+1)
        kn = KneeLocator(x, Sum_of_squared_distances, curve='convex', direction='decreasing')
        return kn.knee

    def kmens(self, data,var_x,var_y):
        X = np.array(data[var_x])
        y = np.array(data[var_y])
        k = self.get_clusters_numbers_SSD(X)
        kmeans = KMeans(n_clusters=k).fit(X)
        x_dr = kmeans.transform(X)
        kmeans.fit(x_dr)
        y = kmeans.predict(x_dr)
        data['label'] = y + 1
        return data

# -------------------------- Flask APP --------------------------#
app = Flask(__name__)
@app.route('/ping')
def ping():
    return jsonify({"mensaje":"From server: Hola Yadir"})

@app.route('/segmentacion', methods = ['GET'])
def segmentar():
    app = segmentacion()
    datos = pd.DataFrame.from_dict(request.json)
    output = app.kmens(datos,['RangoDias',	'RangoMeses',	'Recency',	'Frecuency',	'Latencia',	'Cantidad_Facturas'	,'Ultimo_monto'],'Idkeycliente')
    return jsonify(output.to_json())