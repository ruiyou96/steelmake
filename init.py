import numpy as np
from sys import argv
from scipy import constants

def thcond_ar(T):
    scaledT=T/1000.0
    thcond=1.0
    if(scaledT<5.4):
       # thcond=0.065  this is for N2!
         thcond=0.076#0.026
    else:
        thcond=7.20941793e-01-4.27258433e-01*scaledT+7.30502406e-02*(scaledT**2)+\
                -3.32195385e-03*(scaledT**3)+5.09542436e-05*(scaledT**4)
    return(thcond)

def thcond_h2(T):
    scaledT=T/1000.0
    thcond=1.0
    if(scaledT<1.75):
        thcond=0.44#0.72
    else:
        thcond=(18.0*np.exp(-(scaledT-3.7)**2)+5.0*np.exp(-0.07*(scaledT-14)**2)+scaledT**(1.1)/7)
    return(thcond)


def setmodelparams():
    modelParams = {}
    modelParams['kFeO'] = 3.0#0.58 #W/m/K for FeO
    modelParams['kFe'] = 40.0#73.0 #W/m/K for pure Fe
    modelParams['rho_Fe'] = 7874.0 #kg/m3
    modelParams['rho_FeO'] = 5740.0 #kg/m3
    modelParams['rho_Fe2O3'] =5240.0 #kg/m3
    modelParams['rho_Fe3O4']=5170.0
    modelParams['Cvsolid'] = 200.0 #J/K/kg
    modelParams['D_H2'] = 2e-8# #m2/s #ref:meshram0.000296142
    modelParams['D_H2O'] = 1e-15#m2/s
    
    
    
    #modelParams['D_mix'] =1e-14
    modelParams['D_H']  = 1e-3 #m2/s
    modelParams['D_Hp'] = 1e-3 #m2/s
    modelParams['Tgas']   = 1123.0# K= 850C
    modelParams['Tsolid'] = 1123.0
    modelParams['eps']=0.3 #porosity
    modelParams['tau']=5.0 #tortuosity


    modelParams['pres']=101325.0 #atm pressure
    
    #rest is Ar or N2 //mol fraction
    modelParams['xH']=0.0#0.059
    modelParams['xHp']=0.0#0.012
    modelParams['xH2']=0.33
    modelParams['xH2O']=0.0
    modelParams['xAr']=1.0-modelParams['xH']-\
            modelParams['xHp']-modelParams['xH2']

    #updated in main
    modelParams['C_H2']=0.0
    modelParams['C_H']=0.0
    modelParams['C_Hp']=0.0
    modelParams['C_H2O']=0.0

    modelParams['k0']=2858.34#80#
    modelParams['Ea']=117230.0#
    modelParams['Ke_Ta']=1586.9
    modelParams['Ke_c']=0.9317
    
    #Molecular Mass; Unit: g/mol; ref: bird pp864
    modelParams['M_H2']=2.016
    modelParams['M_H']=1.008
    modelParams['M_Hp']=1.008
    modelParams['M_N2']=28.0134
    modelParams['M_CO']=28.010
    modelParams['M_CO2']=44.010
    modelParams['M_Ar']=39.948
    modelParams['M_H2O']=18.015


    
    
    #P.R. BEHERA, B. BHOI, R.K. PARAMGURU, P.S. MUKHERJEE, and B.K. MISHRA,
    #Hydrogen Plasma Smelting Reduction of Fe2O3
    #METALLURGICAL AND MATERIALS TRANSACTIONS B
    #262—VOLUME 50B, FEBRUARY 2019
    modelParams['facH2']=1.0
    modelParams['facH']=3.0
    modelParams['facHp']=15.0
    
    modelParams['L']=25.0e-6
    modelParams['rg']=25.0e-6
    modelParams['rho_O']=7.5e4
    modelParams['c_0']=1.406e5 #mol/m^3
    modelParams['c_0_feo']=1.406e5
    modelParams['c_0_fe']=1.406e5
    modelParams['c_0_fe2o3']=7.03e4
    modelParams['c_0_fe3o4']=4.687e4
    return(modelParams)
