### Active Learning Algorithm (ALA) for optimizing adsorption energy ($E_{ads}$) predictions

Each iteration contains an unfiltered 1000 new images, 

yet each new iteration also contains the previous iteration's images.

Hence in total we have $\sum_{i=1}^{i=10} 1000i = 55,000$ images.

While only 10,000 of these images are unique ($1000$ images $\times$ $10$ iterations), 

Each iteration's neural network (NN) will infer distinct values for its images.

This is important because for plotting errors over 10 iterations,

We want to see how $NN_i$ is doing on the new set of 1000 images.

Yet for plotting parity, we want all the images it was trained on.

Hence we've split the data into: ```DFT-all.xyz``` and ```DFT-unique.xyz```

DATALINK via [Zenodo](https://zenodo.org/records/21075847?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjAzNDRiOWZiLTY3ZTAtNGI4MC04ZjMzLTk4ZWE2ZmU0NjBmZiIsImRhdGEiOnt9LCJyYW5kb20iOiI0MTY3YTRjODMyOWNlMzRlN2Y3ZDQ5NzU5YjhmM2ZjZiJ9.Gij8sVzS285iw_dslwBkv8WbQDrjcNjArJvFd1ubd313MdTcTQ6wAJQe8qonaYOD9bsr6WYRz0VpE8IP5U4d_A)

```python
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read, write
```


```python
unique_images = read('DFT-unique.xyz', index=':')
len(unique_images)
```




    10000




```python
unique_iter = [[i for i in unique_images if i.info['Iteration']==it] for it in range(10)]
[len(i) for i in unique_iter]
```




    [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]



### Each image contains key-value dictionaries in its ```image.info``` and ```image.arrays``` attributes.


```python
image = unique_images[0]
image.info, image.arrays
```




    ({'DFT_Energy': np.float64(-10237.52451889),
      'NN_Energy': np.float64(-10238.068439812365),
      'Adsorbate': 'NH3',
      'DFT_E_ads': np.float64(0.010404089998765187),
      'NN_E_ads': np.float64(0.043619790573384165),
      'E_ads_Error': np.float64(0.03321570057461898),
      'Iteration': np.int64(0),
      'Split': 'train',
      'Nearest_Neighbor_Distance': np.float64(2.926958444477817),
      'Nearest_Neighbor_Element': 'Pt',
      'Adsorbate_Site': 'bridge',
      'Entropy': np.float64(6.0),
      'Cluster_Size': np.int64(30),
      'DFT_Cluster_Energy': np.float64(-10217.3795636),
      'NN_Cluster_Energy': np.float64(-10217.961312188647),
      'DFT_Adsorbate_Energy': np.float64(-20.15535938),
      'NN_Adsorbate_Energy': np.float64(-20.06350783314466),
      'Energy_Error': np.float64(0.5439209223641228)},
     {'numbers': array([47, 47, 47, 79, 79, 79, 79, 79, 29, 29, 29, 29, 28, 28, 28, 28, 28,
             28, 28, 46, 46, 46, 46, 46, 46, 78, 78, 78, 78, 78,  7,  1,  1,  1]),
      'positions': array([[13.87241, 11.25914,  9.34217],
             [ 9.35818, 11.57369, 14.2329 ],
             [ 9.96287,  7.37957, 13.52763],
             [ 6.66897,  7.88486, 10.18796],
             [10.6917 , 13.45646,  7.81281],
             [ 7.07233, 10.82432,  8.05106],
             [ 7.32961, 12.93118, 10.14289],
             [ 7.97284,  9.32322, 13.26326],
             [ 8.13825, 13.13796,  7.69295],
             [ 9.00778,  8.03114, 11.19215],
             [ 9.48141, 11.98055, 11.64731],
             [10.43288,  9.82016, 12.67604],
             [12.18902,  7.29596,  7.38943],
             [11.47927, 11.05428,  8.26935],
             [10.62915,  9.03158,  7.44155],
             [ 9.34969, 11.69607,  9.18185],
             [ 8.69669,  8.93229,  8.88357],
             [10.40572,  9.92854, 10.20609],
             [12.42205, 10.28514, 11.27735],
             [10.41884,  7.09285,  9.11838],
             [ 9.98137,  6.78386,  6.3862 ],
             [ 7.93625,  6.71749,  8.11607],
             [ 7.11621, 11.91906, 12.68756],
             [11.56982, 12.49246, 10.28789],
             [11.71974, 12.00855, 12.92873],
             [ 8.33688,  8.91034,  6.51081],
             [12.50228,  8.93253,  9.11963],
             [ 9.54315, 11.20205,  6.72025],
             [11.53598,  7.94343, 11.38603],
             [ 7.80399, 10.28847, 10.73252],
             [ 8.67845,  9.50374,  3.55003],
             [ 8.87498,  9.71327,  2.57467],
             [ 7.84968,  8.91468,  3.55462],
             [ 8.40672, 10.38304,  3.9823 ]]),
      'DFT_Forces': array([[ 4.99020e-02,  2.40130e-02,  1.90030e-02],
             [-8.07700e-03,  2.04280e-02, -1.22210e-02],
             [ 2.65950e-02,  6.50800e-03,  9.50300e-03],
             [ 6.09290e-02, -5.02620e-02, -4.22490e-02],
             [ 1.80090e-02,  1.65850e-02,  1.78200e-02],
             [ 2.12160e-02,  2.37000e-03, -2.38070e-02],
             [ 2.80550e-02, -1.80200e-02, -3.53300e-02],
             [-4.45250e-02,  1.24710e-02,  4.76900e-03],
             [ 2.85900e-02, -4.86400e-03,  3.65540e-02],
             [-2.30800e-02, -8.22700e-03, -3.71900e-03],
             [-6.41300e-03,  4.77800e-03,  1.37310e-02],
             [-2.63560e-02, -6.92900e-03,  3.03820e-02],
             [ 1.57902e-01, -4.21300e-02, -8.49020e-02],
             [-8.31680e-02,  1.10925e-01,  1.50540e-02],
             [-5.09970e-02, -3.31300e-02,  2.09195e-01],
             [ 1.49123e-01,  8.02960e-02,  2.22470e-02],
             [-2.38313e-01, -2.81060e-02, -7.38730e-02],
             [ 6.39560e-02, -6.26600e-02,  5.39950e-02],
             [-8.88690e-02,  1.05620e-02, -4.98550e-02],
             [ 8.95030e-02, -2.50040e-02, -4.57920e-02],
             [-2.87980e-02,  1.25260e-02,  7.18110e-02],
             [ 1.32000e-04, -3.03230e-02, -2.68720e-02],
             [ 1.67650e-02, -2.98100e-03, -1.68000e-03],
             [-6.10650e-02,  8.32700e-03, -2.74270e-02],
             [-4.95200e-03,  2.21300e-02, -1.44370e-02],
             [ 9.74040e-02,  3.41950e-02,  2.94070e-02],
             [-5.89070e-02, -1.02281e-01, -7.66400e-02],
             [ 4.07330e-02, -2.16740e-02,  2.79760e-02],
             [-5.78820e-02,  1.02191e-01, -8.54460e-02],
             [ 5.65290e-02, -2.27470e-02,  8.87800e-03],
             [ 1.36637e-01, -1.82083e-01,  2.59872e-01],
             [ 1.70562e-01,  4.85400e-03, -2.96210e-01],
             [-3.02887e-01, -1.21827e-01, -5.07980e-02],
             [-1.28252e-01,  2.90090e-01,  1.21061e-01]]),
      'NN_Forces': array([[ 0.0059221 ,  0.04135514, -0.05167064],
             [ 0.0467096 , -0.09498422, -0.01905941],
             [-0.00553376,  0.02500307, -0.00713956],
             [ 0.15636387,  0.16561317,  0.02930397],
             [-0.01741614,  0.06221935,  0.07533005],
             [-0.00049816,  0.10948104, -0.06125241],
             [ 0.00117937, -0.10983181,  0.05284934],
             [-0.04658363, -0.08380078, -0.1259526 ],
             [-0.02107841, -0.01186814, -0.04110607],
             [ 0.02726342, -0.05178385,  0.00864184],
             [-0.02616892,  0.03192379,  0.01488475],
             [-0.05414302, -0.01876464, -0.02310763],
             [ 0.02263803,  0.11465039,  0.01457618],
             [ 0.08677219, -0.01545947, -0.10038903],
             [-0.003997  ,  0.03205162,  0.03662957],
             [ 0.06932122, -0.12857739, -0.16494631],
             [-0.06404946, -0.08938812, -0.04236434],
             [ 0.0928116 , -0.14648665,  0.06831493],
             [-0.03855226,  0.00148274, -0.00951272],
             [ 0.02147717, -0.07897707,  0.17661063],
             [-0.03711707, -0.27627471, -0.40337602],
             [-0.18691526, -0.19563743,  0.03809798],
             [-0.42616593,  0.24715014,  0.19965593],
             [ 0.0548712 ,  0.18634833,  0.06431742],
             [ 0.26819597,  0.20526579,  0.24221298],
             [-0.03083972, -0.05481295,  0.17424368],
             [ 0.13396011,  0.01890927,  0.05157984],
             [ 0.05474372,  0.10466714, -0.04792387],
             [ 0.05053136, -0.11189248,  0.02374112],
             [-0.0272646 ,  0.09827408,  0.05483338],
             [ 0.16421907, -0.23477766,  0.13004349],
             [ 0.17128632,  0.04283443, -0.32337962],
             [-0.31218771, -0.15432462, -0.11083758],
             [-0.12975526,  0.37041249,  0.07615071]]),
      'Forces_Error': array([[0.0439799 , 0.01734214, 0.07067364],
             [0.0547866 , 0.11541222, 0.00683841],
             [0.03212876, 0.01849507, 0.01664256],
             [0.09543487, 0.21587517, 0.07155297],
             [0.03542514, 0.04563435, 0.05751005],
             [0.02171416, 0.10711104, 0.03744541],
             [0.02687563, 0.09181181, 0.08817934],
             [0.00205863, 0.09627178, 0.1307216 ],
             [0.04966841, 0.00700414, 0.07766007],
             [0.05034342, 0.04355685, 0.01236084],
             [0.01975592, 0.02714579, 0.00115375],
             [0.02778702, 0.01183564, 0.05348963],
             [0.13526397, 0.15678039, 0.09947818],
             [0.16994019, 0.12638447, 0.11544303],
             [0.047     , 0.06518162, 0.17256543],
             [0.07980178, 0.20887339, 0.18719331],
             [0.17426354, 0.06128212, 0.03150866],
             [0.0288556 , 0.08382665, 0.01431993],
             [0.05031674, 0.00907926, 0.04034228],
             [0.06802583, 0.05397307, 0.22240263],
             [0.00831907, 0.28880071, 0.47518702],
             [0.18704726, 0.16531443, 0.06496998],
             [0.44293093, 0.25013114, 0.20133593],
             [0.1159362 , 0.17802133, 0.09174442],
             [0.27314797, 0.18313579, 0.25664998],
             [0.12824372, 0.08900795, 0.14483668],
             [0.19286711, 0.12119027, 0.12821984],
             [0.01401072, 0.12634114, 0.07589987],
             [0.10841336, 0.21408348, 0.10918712],
             [0.0837936 , 0.12102108, 0.04595538],
             [0.02758207, 0.05269466, 0.12982851],
             [0.00072432, 0.03798043, 0.02716962],
             [0.00930071, 0.03249762, 0.06003958],
             [0.00150326, 0.08032249, 0.04491029]])})



### With the help of a parsing function, we can extract useful values for our analysis.


```python
import pandas as pd

def Parse_Iteration(images, iteration, outlier_thresh=10.0, verbose=False):
    '''Parses a list of images to calculate core error metrics for NN predictions.'''
    
    splits = ['train', 'valid', 'test']
    targets = ['E', 'F', 'E_ads']
    
    # Initialize the storage architecture
    data = {s: {t: {'ref': [], 'pred': []} for t in targets} for s in splits}
    invalid_count = 0
    
    for atoms in images:
        
        # Fails fast with KeyError if 'split' is missing
        split = atoms.info['Split']
        if split not in data:
            raise ValueError(f'Unexpected split value: {split}. Expected train, valid, or test.')

        # Estract values
        n_atoms = len(atoms)
        ref_e = atoms.info['DFT_Energy'] / n_atoms
        pred_e = atoms.info['NN_Energy'] / n_atoms
        ref_f = atoms.arrays['DFT_Forces'].ravel()
        pred_f = atoms.arrays['NN_Forces'].ravel()        
        ref_e_ads = atoms.info['DFT_E_ads']
        pred_e_ads = atoms.info['NN_E_ads']
        
        # Outlier filter
        if abs(ref_e - pred_e) >= outlier_thresh or np.max(np.abs(ref_f - pred_f)) >= outlier_thresh:
            invalid_count += 1
            continue
            
        # Store values
        data[split]['E']['ref'].append(ref_e)
        data[split]['E']['pred'].append(pred_e)
        data[split]['F']['ref'].extend(ref_f)
        data[split]['F']['pred'].extend(pred_f)
        data[split]['E_ads']['ref'].append(ref_e_ads)
        data[split]['E_ads']['pred'].append(pred_e_ads)
            
    if verbose and invalid_count > 0:
        print(f'Iteration {iteration}: Dropped {invalid_count} outliers.')
        
    row = {'Iteration': iteration}
    
    # Vectorized Metric Calculation
    for split in splits:
        prefix = split.capitalize()
        
        for target in targets:
            col_base = f'{prefix}_{target}'
            
            ref = np.array(data[split][target]['ref'])
            pred = np.array(data[split][target]['pred'])
            n = len(ref)
            
            if n == 0:
                row[f'{col_base}_RMSE'] = 0.0
                row[f'{col_base}_MAE'] = 0.0
                row[f'{col_base}_SD'] = 0.0
                row[f'{col_base}_CI_95'] = 0.0
                continue
                
            errors = pred - ref
            abs_errors = np.abs(errors)
            
            # Core Metrics
            row[f'{col_base}_RMSE'] = np.round(np.sqrt(np.mean(errors**2)), 5)
            row[f'{col_base}_MAE'] = np.round(np.mean(abs_errors), 5)
            
            # Rigorous Statistics 
            sd = np.std(abs_errors, ddof=1) if n > 1 else 0.0
            row[f'{col_base}_SD'] = np.round(sd, 5)
            row[f'{col_base}_CI_95'] = np.round(1.96 * (sd / np.sqrt(n)), 5)
            
    return row
```


```python
unique_df = pd.DataFrame([Parse_Iteration(images, i) for i, images in enumerate(unique_iter)])
unique_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Iteration</th>
      <th>Train_E_RMSE</th>
      <th>Train_E_MAE</th>
      <th>Train_E_SD</th>
      <th>Train_E_CI_95</th>
      <th>Train_F_RMSE</th>
      <th>Train_F_MAE</th>
      <th>Train_F_SD</th>
      <th>Train_F_CI_95</th>
      <th>Train_E_ads_RMSE</th>
      <th>...</th>
      <th>Test_E_SD</th>
      <th>Test_E_CI_95</th>
      <th>Test_F_RMSE</th>
      <th>Test_F_MAE</th>
      <th>Test_F_SD</th>
      <th>Test_F_CI_95</th>
      <th>Test_E_ads_RMSE</th>
      <th>Test_E_ads_MAE</th>
      <th>Test_E_ads_SD</th>
      <th>Test_E_ads_CI_95</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.02786</td>
      <td>0.02209</td>
      <td>0.01698</td>
      <td>0.00126</td>
      <td>0.12925</td>
      <td>0.09071</td>
      <td>0.09207</td>
      <td>0.00067</td>
      <td>0.38501</td>
      <td>...</td>
      <td>0.01344</td>
      <td>0.00263</td>
      <td>0.15229</td>
      <td>0.09926</td>
      <td>0.11550</td>
      <td>0.00222</td>
      <td>0.34834</td>
      <td>0.23069</td>
      <td>0.26231</td>
      <td>0.05141</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.01984</td>
      <td>0.01582</td>
      <td>0.01197</td>
      <td>0.00090</td>
      <td>0.11088</td>
      <td>0.07839</td>
      <td>0.07842</td>
      <td>0.00059</td>
      <td>0.22765</td>
      <td>...</td>
      <td>0.01793</td>
      <td>0.00340</td>
      <td>0.12246</td>
      <td>0.08437</td>
      <td>0.08877</td>
      <td>0.00169</td>
      <td>0.20894</td>
      <td>0.15908</td>
      <td>0.13610</td>
      <td>0.02579</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.01972</td>
      <td>0.01437</td>
      <td>0.01350</td>
      <td>0.00100</td>
      <td>0.10174</td>
      <td>0.07100</td>
      <td>0.07287</td>
      <td>0.00054</td>
      <td>0.18048</td>
      <td>...</td>
      <td>0.01834</td>
      <td>0.00338</td>
      <td>0.12332</td>
      <td>0.08123</td>
      <td>0.09280</td>
      <td>0.00176</td>
      <td>0.19907</td>
      <td>0.13814</td>
      <td>0.14397</td>
      <td>0.02655</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.01694</td>
      <td>0.01279</td>
      <td>0.01111</td>
      <td>0.00082</td>
      <td>0.09331</td>
      <td>0.06397</td>
      <td>0.06794</td>
      <td>0.00051</td>
      <td>0.18891</td>
      <td>...</td>
      <td>0.01383</td>
      <td>0.00266</td>
      <td>0.10101</td>
      <td>0.06795</td>
      <td>0.07474</td>
      <td>0.00153</td>
      <td>0.17771</td>
      <td>0.14121</td>
      <td>0.10842</td>
      <td>0.02084</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.01252</td>
      <td>0.00986</td>
      <td>0.00773</td>
      <td>0.00058</td>
      <td>0.09191</td>
      <td>0.06470</td>
      <td>0.06527</td>
      <td>0.00049</td>
      <td>0.14742</td>
      <td>...</td>
      <td>0.00780</td>
      <td>0.00145</td>
      <td>0.10502</td>
      <td>0.07284</td>
      <td>0.07565</td>
      <td>0.00144</td>
      <td>0.16383</td>
      <td>0.10806</td>
      <td>0.12370</td>
      <td>0.02301</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>0.01142</td>
      <td>0.00880</td>
      <td>0.00727</td>
      <td>0.00053</td>
      <td>0.09299</td>
      <td>0.06523</td>
      <td>0.06628</td>
      <td>0.00050</td>
      <td>0.13319</td>
      <td>...</td>
      <td>0.00666</td>
      <td>0.00142</td>
      <td>0.10275</td>
      <td>0.07028</td>
      <td>0.07496</td>
      <td>0.00170</td>
      <td>0.13451</td>
      <td>0.09871</td>
      <td>0.09192</td>
      <td>0.01954</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>0.01207</td>
      <td>0.00846</td>
      <td>0.00861</td>
      <td>0.00064</td>
      <td>0.08341</td>
      <td>0.05807</td>
      <td>0.05988</td>
      <td>0.00046</td>
      <td>0.14626</td>
      <td>...</td>
      <td>0.00609</td>
      <td>0.00112</td>
      <td>0.09057</td>
      <td>0.06067</td>
      <td>0.06724</td>
      <td>0.00126</td>
      <td>0.14389</td>
      <td>0.10042</td>
      <td>0.10352</td>
      <td>0.01909</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>0.01346</td>
      <td>0.00948</td>
      <td>0.00957</td>
      <td>0.00070</td>
      <td>0.08344</td>
      <td>0.05912</td>
      <td>0.05888</td>
      <td>0.00045</td>
      <td>0.14409</td>
      <td>...</td>
      <td>0.00857</td>
      <td>0.00175</td>
      <td>0.09312</td>
      <td>0.06514</td>
      <td>0.06655</td>
      <td>0.00139</td>
      <td>0.13758</td>
      <td>0.10360</td>
      <td>0.09102</td>
      <td>0.01860</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>0.01117</td>
      <td>0.00845</td>
      <td>0.00731</td>
      <td>0.00055</td>
      <td>0.07880</td>
      <td>0.05592</td>
      <td>0.05552</td>
      <td>0.00044</td>
      <td>0.11054</td>
      <td>...</td>
      <td>0.01708</td>
      <td>0.00319</td>
      <td>0.08923</td>
      <td>0.05589</td>
      <td>0.06956</td>
      <td>0.00135</td>
      <td>0.12434</td>
      <td>0.08136</td>
      <td>0.09445</td>
      <td>0.01765</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>0.00685</td>
      <td>0.00484</td>
      <td>0.00485</td>
      <td>0.00035</td>
      <td>0.05303</td>
      <td>0.03833</td>
      <td>0.03665</td>
      <td>0.00028</td>
      <td>0.09173</td>
      <td>...</td>
      <td>0.00671</td>
      <td>0.00133</td>
      <td>0.07298</td>
      <td>0.04795</td>
      <td>0.05501</td>
      <td>0.00118</td>
      <td>0.11109</td>
      <td>0.07668</td>
      <td>0.08079</td>
      <td>0.01608</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 37 columns</p>
</div>



### The confidence interval (CI) is defined $C_I = \alpha \frac{\sigma}{\sqrt{N}}$, where $\alpha$ is chosen to represent 95% of the uncertainty.

#### It helps to make a minimal plot to showcase the basic trends.


```python
from matplotlib.lines import Line2D

def plot_metric_with_ci(df, metric):
    
    metrics = ['E', 'F', 'E_ads']
    if metric not in metrics: 
        raise ValueError(f'Invalid metric: {metric}. Expected one of {metrics}')
    
    plt.figure(figsize=(5, 5))
    
    splits = {
        'Train': {'ls': '-', 'color': 'blue', 'offset': -0.15},
        'Valid': {'ls': '--', 'color': 'green', 'offset': 0.0},
        'Test': {'ls': ':', 'color': 'red', 'offset': 0.15}
    }
    
    for split, props in splits.items():
        mae_col = f'{split}_{metric}_MAE'
        ci_col = f'{split}_{metric}_CI_95'
        
        if mae_col not in df.columns or ci_col not in df.columns:
            raise KeyError(f'Missing columns for {split} split.')
        
        x_shifted = df['Iteration'] + props['offset']
        
        plt.plot(x_shifted, df[mae_col], ls=props['ls'], color=props['color'], label=split)
        plt.errorbar(x_shifted, df[mae_col], yerr=df[ci_col], 
                     fmt='none', capsize=4, ecolor=props['color'], alpha=0.9)
        plt.fill_between(x_shifted, 
                         df[mae_col] - df[ci_col], 
                         df[mae_col] + df[ci_col], 
                         color=props['color'], alpha=0.15)
                         
    plt.grid(ls=':', alpha=0.5)
    plt.xticks(df['Iteration'].unique())    
    plt.xlabel('Iteration')
    plt.ylabel(f'{metric} MAE (with 95% CI)')
    plt.legend()
    plt.show()

def plot_all_maes(df):
    
    plt.figure(figsize=(5, 5))

    splits = ['Train', 'Valid', 'Test']
    lines = ['-', '--', ':']
    marks = ['o', '^', 'x']

    for split, mark, line in zip(splits, marks, lines):
        
        x = df['Iteration']
        e_ads = df[f'{split}_E_ads_MAE']
        f_err = df[f'{split}_F_MAE']
        e_err = df[f'{split}_E_MAE']

        plt.plot(x, e_ads, c='b', marker=mark, lw=2, ls=line, alpha=0.5)
        plt.plot(x, f_err, c='r', marker=mark, lw=2, ls=line, alpha=0.5)
        plt.plot(x, e_err, c='g', marker=mark, lw=2, ls=line, alpha=0.5)

    legend_keys = [
        Line2D([0], [0], color='gray', lw=2, marker='o', alpha=0.7, label='train'),
        Line2D([0], [0], color='gray', lw=2, ls='--', marker='^', alpha=0.7, label='valid'),
        Line2D([0], [0], color='gray', lw=2, ls=':', marker='x', alpha=0.7, label='test'),
        Line2D([0], [0], color='b', lw=4, alpha=0.7, label='E_ads (eV)'),
        Line2D([0], [0], color='r', lw=4, alpha=0.7, label='Forces (eV/Å)'),
        Line2D([0], [0], color='g', lw=4, alpha=0.7, label='Energy (eV/atom)')
    ]

    plt.legend(handles=legend_keys, loc='upper right', ncol=2, handletextpad=0.5)

    plt.xlabel('Iteration')
    plt.ylabel('Mean Absolute Error')
    plt.xticks(df['Iteration'])
    plt.grid(ls=':', alpha=0.5)
    plt.show()
```


```python
plot_metric_with_ci(unique_df, 'E_ads')
```


    
![png](ALA_Analysis_files/ALA_Analysis_11_0.png)
    



```python
plot_all_maes(unique_df)
```


    
![png](ALA_Analysis_files/ALA_Analysis_12_0.png)
    


## We can also examine each Iteration's performace with parity plots.
For this we do not use the dataframe, but rather access the ASE atom's objects directly.


```python
all_images = read('DFT-all.xyz', index=':')
all_iter = [[i for i in all_images if i.info['Iteration']==it] for it in range(10)]
[len(i) for i in all_iter]
```




    [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]




```python
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde

def get_fast_density(x, y, bins=150, smooth=2.0):
    '''Fast 2D histogram density estimation for massive arrays.'''
    H, xedges, yedges = np.histogram2d(x, y, bins=bins)
    H_smooth = gaussian_filter(H, sigma=smooth)
    
    xidx = np.clip(np.searchsorted(xedges, x) - 1, 0, bins - 1)
    yidx = np.clip(np.searchsorted(yedges, y) - 1, 0, bins - 1)
    
    return H_smooth[xidx, yidx]

def plot_parity(x, y, kind, outlier_thresh=10):
    '''Plots a parity density scatter with smart KDE routing.'''
    
    labels = {
        'e_ads': ('DFT E_ads (eV)', 'NN E_ads (eV)'),        
        'energy': ('DFT Energy (eV/atom)', 'NN Energy (eV/atom)'),
        'forces': ('DFT Forces (eV/Å)', 'NN Forces (eV/Å)')
    }
    
    if kind not in labels:
        raise ValueError(f'Invalid kind: {kind}. Expected one of {list(labels.keys())}.')
        
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    
    # Outlier Filter
    dropped_count = 0
    if outlier_thresh is not None:
        valid_mask = np.abs(y - x) <= outlier_thresh
        dropped_count = len(x) - np.sum(valid_mask)
        x = x[valid_mask]
        y = y[valid_mask]
        
    if len(x) == 0:
        raise ValueError('Outlier threshold is too strict; all data points were dropped.')
    
    # Metric Calculations on the Filtered Data
    rmse = np.sqrt(np.mean((y - x)**2))
    mae = np.mean(np.abs(y - x))
    span = np.abs(np.max(x) - np.min(x))
    maep = np.round((mae / span) * 100, 3) if span > 0 else 0.0
    
    # Density Estimation
    if kind == 'forces' or len(x) > 5000:
        z = get_fast_density(x, y)
    else:
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        
    z = (z - z.min()) / (z.max() - z.min())
    idx = z.argsort()
    x_plot, y_plot, z_plot = x[idx], y[idx], z[idx]

    # Canvas Rendering
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(x_plot, y_plot, c=z_plot, s=15, cmap='turbo', edgecolor='none', alpha=0.75, zorder=2)

    if kind == 'e_ads': 
        lims = [-6.5, 2.5]
        ax.set_xticks(np.linspace(-6, 2, 5))
        ax.set_yticks(np.linspace(-6, 2, 5))
    else:
        min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
        max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
        lims = [min_val, max_val]

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.plot(lims, lims, 'k--', alpha=0.5, zorder=1)
    
    # Dynamic Text Box
    text = f'RMSE: {rmse:.3f}\nMAE: {mae:.3f}\nMAE%: {maep:.2f}'
    text += f'\nDropped: {dropped_count}'
        
    ax.text(0.05, 0.95, text, transform=ax.transAxes,
            va='top', ha='left',
            bbox=dict(boxstyle='round, pad=0.3', edgecolor='gray', facecolor='white', alpha=0.75))

    ax.set_xlabel(labels[kind][0])
    ax.set_ylabel(labels[kind][1])
    ax.grid(ls=':', alpha=0.5, zorder=0)
    plt.show()
```

### Note there are many ways to showcase this dataset


```python
Ex, Ey = [], []
Ax, Ay = [], []
Fx, Fy = [], []

for i in all_iter[9]:
    Ex.append(i.info['DFT_Energy']/len(i))
    Ey.append(i.info['NN_Energy']/len(i))
    Ax.append(i.info['DFT_E_ads'])
    Ay.append(i.info['NN_E_ads'])
    Fx.extend(i.arrays['DFT_Forces'].ravel())
    Fy.extend(i.arrays['NN_Forces'].ravel())

limit = 1.0
plot_parity(Ex, Ey, kind='energy', outlier_thresh=limit)
plot_parity(Ax, Ay, kind='e_ads', outlier_thresh=limit)
plot_parity(Fx, Fy, kind='forces', outlier_thresh=limit)
```


    
![png](ALA_Analysis_files/ALA_Analysis_17_0.png)
    



    
![png](ALA_Analysis_files/ALA_Analysis_17_1.png)
    



    
![png](ALA_Analysis_files/ALA_Analysis_17_2.png)
    


### Finally we exemplify some basica per adsorbate analysis
Note for this we are using the final fully relaxed images (1k)


```python
final_images = read('final-DFT.xyz', index=':')
len(final_images)
```




    1000




```python
def plot_nnd_scatter(images, target_adsorbate, max_nnd=4.0):
    '''Plots a jittered scatter of NNDs grouped by element, shielded by a maximum distance limit.'''
    
    if isinstance(images[0], list):
        flat_images = [img for sublist in images for img in sublist]
    else:
        flat_images = images
    
    data = {}
    dropped_count = 0
    for img in flat_images:
        if img.info.get('Adsorbate') == target_adsorbate:
            elem = img.info['Nearest_Neighbor_Element']
            nnd = img.info['Nearest_Neighbor_Distance']
            if nnd <= max_nnd:
                if elem not in data:
                    data[elem] = []
                data[elem].append(nnd)
            else:
                dropped_count += 1
    if not data:
        raise ValueError(f'No valid data found in image list for adsorbate: {target_adsorbate} within limit {max_nnd} A.')

    element_colors = {
        'Ag': 'red',
        'Au': 'orange',
        'Cu': 'yellow', 
        'Ni': 'green', 
        'Pd': 'blue', 
        'Pt': 'violet'
    }

    unique_elems = sorted(data.keys())
    fig, ax = plt.subplots(figsize=(5, 5))

    for i, elem in enumerate(unique_elems):
        elem_data = np.array(data[elem])
        # Apply a uniform random jitter to spread the points horizontally
        x_jitter = i + np.random.uniform(-0.15, 0.15, size=len(elem_data))
        color = element_colors.get(elem, 'gray')
        ax.scatter(x_jitter, elem_data, color=color, alpha=0.7, edgecolor='k', s=40, zorder=2)
        mean_val = elem_data.mean()
        ax.hlines(mean_val, i - 0.25, i + 0.25, color='k', lw=2.5, zorder=3)

    ax.set_xticks(range(len(unique_elems)))
    ax.set_xticklabels(unique_elems)
    ax.set_title(f'{target_adsorbate} Anchoring Distances')
    ax.set_ylabel('NND (Å)')
    ax.set_xlabel('Nearest Neighbor Element')
    
    # Dynamically report dropped outliers
    if dropped_count > 0:
        ax.text(0.95, 0.95, f'Dropped > {max_nnd} Å: {dropped_count}', transform=ax.transAxes, 
                ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='lightgray'))
    ax.grid(ls=':', alpha=0.5, zorder=0)
    plt.show()
```


```python
plot_nnd_scatter(final_images, 'NH3', max_nnd=4)
```


    
![png](ALA_Analysis_files/ALA_Analysis_21_0.png)
    


all code written by anywallsocket


```python

```
