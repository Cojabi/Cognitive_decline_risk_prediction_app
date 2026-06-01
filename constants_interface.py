import numpy as np

FEATURE_ORDER_DEMO = ['Sex', #'Ethnicity', 
                      'Education', 'ptau217_Age', 'ptau217_harm',
       'Hypertension_ever', 'BP_Systolic', 'BP_Diastolic', 'BMI', 'Race_Asian',
       'Race_Black or African American', 'Race_White']

FEATURE_ORDER_APOE =['Sex', #'Ethnicity', 
                     'Education', 'e4_carrier', 'ptau217_Age',
       'ptau217_harm', 'Hypertension_ever', 'BP_Systolic', 'BP_Diastolic',
       'BMI', 'Race_Asian', 'Race_Black or African American', 'Race_White']

FEATURE_ORDER_APOE_NO_HYP =['Sex', #'Ethnicity', 
                            'Education', 'e4_carrier', 'ptau217_Age',
       'ptau217_harm', 'BP_Systolic', 'BP_Diastolic',
       'BMI', 'Race_Asian', 'Race_Black or African American', 'Race_White']

FEATURE_ORDER_BM_PTAU = ['Sex', #'Ethnicity', 
                         'Education', 'ptau217_Age', 'ptau217_harm',
       'Hypertension_ever', 'MMSE', 'BP_Systolic', 'BP_Diastolic', 'BMI',
       'Race_Asian', 'Race_Black or African American', 'Race_White']

TIME_GRID = [1,2,3,4,5,6,7,8,9,10,11,12] #np.linspace(0, 12, 1)