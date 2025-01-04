import numpy as np
from sys import argv
from scipy import constants
from init import *
#from tempeq import *
#from gas_speceq import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import copy
import time
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve





#----------------------------------------
#
#     Implicit PDE solver for Diff Eq
#
#----------------------------------------
def solver(init, nx, t, dx, dt, De_A, r_A, c_spec, k_m):

    # Create matrices for implicit solver
    A = np.zeros((nx, nx))
    b = np.zeros(nx)
    
    
    # Inner Coefficients for mass transfer
    for i in range(1, nx-1):
    
        # Compute diffusion coefficient at the cell faces
        De_A_face_left = 0.5 * (De_A[i-1] + De_A[i])  #  left face
        De_A_face_right = 0.5 * (De_A[i] + De_A[i+1])  #  right face

        # FVM discretization for diffusion term
       
        A[i, i] = dx/dt - (De_A_face_left + De_A_face_right)/dx
        A[i, i+1] = De_A_face_right /dx
        A[i, i-1] = De_A_face_left /dx
        
        #r_A = reaction_rate(xA[i])
        b[i] = r_A[i]*dx + dx/dt*c_spec[i]
        #print ("r_A",r_A[i])


#this was for FDS
#    # at the center-0 flux
#    b[-1] = c_spec[-2]
#    A[-1, -1] = 1 #Last node
#    A[-1, -2] = 0  # Second-to-last node

# FVM
    A[-1, -1] = 1
    A[-1, -2] = -1  # Enforces the symmetric condition c_spec[-1] = c_spec[-2]
    b[-1] = 0       # Zero flux implies no change in concentration at the center
    


# this was for FDS
#    b[-1] = c_spec[-2]  # 0 flux at the center
#    A[-1, -1] = 1 #Last node
#    A[-1, -2] = 0  # Second-to-last node

    A[0, 0] = 1
    b[0] = init[0]
    
    # Solve for new concentration
    #c_spec = np.linalg.solve(A, b)
    A= csr_matrix(A)
    c_spec =  spsolve(A, b)
    
    #shape = A.shape
    #print("Shape of the matrix:", shape)

    return c_spec


#----------------------------------------
#
#            plot concentration
#
#----------------------------------------
def plot_separate_figures(x, cH2, cH2O):

    # Create a 2-row, 1-column subplot layout
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 6))

    # Plot H2 concentration
    ax1.plot(x, cH2, label='cH2', color='b', linewidth=2)
    ax1.set_xlabel('x(m)', fontsize=12)
    ax1.set_ylabel("Concentration of H₂ (mol/m³)", fontsize=12)
    #ax1.set_title('Concentration of H2', fontsize=14)
    ax1.grid(True)
    ax1.legend()

    # Plot H2O concentration
    ax2.plot(x, cH2O, label='cH2O', color='g', linewidth=2)
    ax2.set_xlabel('x(m)', fontsize=12)
    ax2.set_ylabel("Concentration of H₂O (mol/m³)", fontsize=12)
    #ax2.set_title('Concentration of H2O', fontsize=14)
    ax2.grid(True)
    ax2.legend()


    plt.tight_layout()
    plt.show()
    
    
def plot_separate_figures_solid(x, xFe, xFeO):

    # Create a 2-row, 1-column subplot layout
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 6))

    # Plot H2 concentration
    ax1.plot(x, x_Fe, label='xFe', color='b', linewidth=2)
    ax1.set_xlabel('Position (x)', fontsize=12)
    ax1.set_ylabel('xFe', fontsize=12)
    ax1.set_title('xFe', fontsize=14)
    ax1.grid(True)
    ax1.legend()

    # Plot H2O concentration
    ax2.plot(x, x_FeO, label='xFeO', color='g', linewidth=2)
    ax2.set_xlabel('Position (x)', fontsize=12)
    ax2.set_ylabel('xFeO', fontsize=12)
    ax2.set_title('xFeO', fontsize=14)
    ax2.grid(True)
    ax2.legend()


    plt.tight_layout()
    plt.show()
#----------------------------------------
#
#     movie of concentration vs. time
#
#----------------------------------------
def animate_concentration(x, cH2_list, cH2O_list, dt):
    """
    Function to animate the concentrations of H2 and H2O over time.

    Parameters:
    x -- array of spatial positions
    cH2_list -- list of arrays with H2 concentrations at each time step
    cH2O_list -- list of arrays with H2O concentrations at each time step
    dt -- time step size
    """

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    # Set up H2 plot
    ax1.set_title("Concentration of H2 over time")
    line1, = ax1.plot(x, cH2_list[0], color='b', label='cH2', linewidth=2)
    ax1.set_xlabel("Position (x)")
    ax1.set_ylabel("Concentration of H2")
    ax1.grid(True)
    
    # Set up H2O plot
    ax2.set_title("Concentration of H2O over time")
    line2, = ax2.plot(x, cH2O_list[0], color='g', label='cH2O', linewidth=2)
    ax2.set_xlabel("Position (x)")
    ax2.set_ylabel("Concentration of H2O")
    ax2.grid(True)

    # Function to update the plots at each frame
    def update(frame):
        line1.set_ydata(cH2_list[frame])  # Update H2 data
        line2.set_ydata(cH2O_list[frame]) # Update H2O data
        ax1.set_title(f"Concentration of H2 at time {frame*dt:.2f}")
        ax2.set_title(f"Concentration of H2O at time {frame*dt:.2f}")
        return line1, line2

    # Create animation
    ani = FuncAnimation(fig, update, frames=len(cH2_list), interval=100, blit=True)

    plt.tight_layout()
    plt.show()



# Safe conversion function to avoid negative values in power calculation
def safe_conversion(x, tau, power):
    # Ensure x is a NumPy array for consistent indexing
    x = np.asarray(x)
    
    result = np.zeros_like(x)
    valid_time = x <= tau
    result[valid_time] = 1 - (1 - x[valid_time] / tau)**power
    result[~valid_time] = 1  # Cap conversion at 1 for times greater than tau
    return result


#----------------------------------------
#
#                main
#
#----------------------------------------



if(__name__ == "__main__"):
    #   # of cells
    ncells=int(argv[1])
    #   max # of time steps
    maxsteps=int(argv[2])
    #   flow time
    fintime=float(argv[3])
    #   output interval
    outputint=int(argv[4])
    #   run option: 1: FeO; 2: Fe2O3
    run_option=int(argv[5])
    
    screenint=int(outputint/10)



    mp=setmodelparams()

    # concentration mol/m^3
    fsolid=np.zeros(ncells)
    cH2=np.zeros(ncells)
    cFe=np.zeros(ncells)
    cFeO=np.zeros(ncells)
    cFe2O3=np.zeros(ncells)
    cFe3O4=np.zeros(ncells)
    x_Fe=np.zeros(ncells)
    x_FeO=np.zeros(ncells)
    x_Fe2O3=np.zeros(ncells)
    x_Fe3O4=np.zeros(ncells)
    f=np.zeros(ncells)
    cH=np.zeros(ncells)
    cHp=np.zeros(ncells)
    cH2O=np.zeros(ncells)
    Temp=np.zeros(ncells)
    eps=np.zeros(ncells)
    knu_k0=np.zeros(ncells)
    
    avlb=np.zeros(ncells)
    avlb_1=np.zeros(ncells)
    csmp=np.zeros(ncells)
    alpha=np.zeros(ncells)
    alpha_1=np.zeros(ncells)
    f_alpha=np.zeros(ncells)

    
    cH2_init=np.zeros(ncells)
    cH_init=np.zeros(ncells)
    cHp_init=np.zeros(ncells)
    cH2O_init=np.zeros(ncells)
    Temp_init=np.zeros(ncells)

    thermalcond=np.zeros(ncells)
    alpha_gas=np.zeros(ncells)
    D_knu_H2=np.zeros(ncells)
    D_knu_H2O=np.zeros(ncells)
    alpha_hp=np.zeros(ncells)
    alpha_m=np.zeros(ncells)
    Deff_H2O=np.zeros(ncells)
    Deff_H2=np.zeros(ncells)
    
    Temp[:]=mp['Tsolid']
    
    #ctotal= P/RT
    conc_total=mp['pres']/mp['Tgas']/constants.R
    
    #Species Concentrations=conc_total*mole fractions
    mp['C_H2']=conc_total*mp['xH2']
    mp['C_H']=conc_total*mp['xH']
    mp['C_Hp']=conc_total*mp['xHp']
    mp['C_H2O']=conc_total*mp['xH2O']
    #note: cAr represents any unreacted gas, it could be N2, or Ar
    cAr=conc_total-mp['C_H2']-mp['C_H']-mp['C_Hp']-mp['C_H2O']
    print("C_H2,C_H,C_Hp:",mp['C_H2'],mp['C_H'],mp['C_Hp'])
    
    
    #density=concentrations* molar masses
    # N2
    dens=(cAr*mp['M_N2']+mp['C_H2']*mp['M_H2']+mp['C_H']*mp['M_H']+mp['C_Hp']*mp['M_Hp']+ cH2O*mp['M_H2O'])/1000.0
    #print("dens:",dens)
    
    #Copy Initial Values:
    Temp_init=copy.deepcopy(Temp)
    cH2_init=copy.deepcopy(cH2)
    cH_init=copy.deepcopy(cH)
    cHp_init=copy.deepcopy(cHp)
    cH2O_init=copy.deepcopy(cH2O)

    
    # print("ch2", cH2_init)
    
    #Initialize Reaction Source Array
    impl_rxnsrc=np.zeros((5,ncells))
    s_src=np.zeros((7,ncells))
    
    #Define Grid/ TIME STEP
    plo=0.0 #LOW
    phi=mp['L'] #HIGH
    dx=(phi-plo)/ncells
    x=np.linspace(0.5*dx,phi-0.5*dx,ncells)
    dt=fintime/maxsteps

    t=0.0
    nsteps=0
    outint=0
    
    
   
    # porosity
    eps_start = 0.3
    eps_end = 0.4
    eps_increment = (eps_end - eps_start) / maxsteps
    
    
    # Initialize iron ore
    if run_option == 1:
        for c in range(ncells):
            cFeO[c] = mp['c_0']
            x_FeO[c]=1.0
    else:
        for c in range(ncells):
            cFe2O3[c] = mp['c_0_fe2o3']
            x_Fe2O3[c]=1.0
        
    # Initialize lists to store time and conversion degree values
    time_values = []
    f_values = []
    scm_f= []
    f_mean= []
    
    # Define a list to store conversion and time data
    conversion_vs_time = []
    conversion_vs_time_h2o = []
    rc_vs_time = []
    
    
    # Shrinking Core Model-related parameters
    b = 1  # Stoichiometric coefficient
    r_c = np.zeros(maxsteps+1)  #
    r_c[0] = mp['rg']



##--------------------------------------------------
##
##           MAIN LOOP SOLVING PDE
##
##--------------------------------------------------
    # Start the timer
    start_time = time.time()
    while(t<fintime and nsteps<maxsteps):
        
#        Temp_init=copy.deepcopy(Temp)
#        cH2_init=copy.deepcopy(cH2)
#        cH_init=copy.deepcopy(cH)
#        cHp_init=copy.deepcopy(cHp)
#        cH2O_init=copy.deepcopy(cH2O)
        
        cH2_init[0]=np.array(mp['C_H2'])#*eps[0]
        #cH2_init=np.array(mp['C_H2'])
        
        #print("ch2",cH2_init)
        #print("ch2",cH2O_init)
        
       # mp['eps'] += eps_increment
#        if time < fintime / 2:
#           mp['eps'] = 0.3
#        else:
#           mp['eps'] = 0.4

        mp['eps'] =1.0
        
        

        
       # mp['tau']=mp['eps']**(-1/3)
        mp['tau']=mp['eps']**(-1/2)
        
        rg=mp['rg']
        Vg=4.0/3.0*np.pi*(rg**3)


        for c in range(ncells):
            xAr=cAr/(cAr+cH2[c]+cH[c]+cHp[c])
            xH2=cH2[c]/(cAr+cH2[c]+cH[c]+cHp[c])
            #Thermal Conductivity
            kgas=xAr*thcond_ar(Temp[c])+xH2*thcond_h2(Temp[c])
            
            #EFF Conductivity
            thinv=mp['eps']/kgas+(1-mp['eps'])*\
                    ((1.0-fsolid[c])/mp['kFeO']+fsolid[c]/mp['kFe'])
            thermalcond=1.0/thinv;
            
            
            dens_gas=(cAr*mp['M_N2']+mp['C_H2']*mp['M_H2']+mp['C_H']*mp['M_H']+mp['C_Hp']*mp['M_Hp']+ cH2O*mp['M_H2O'])/1000.0
            
            
            #
            cp=1.6/0.6*constants.R/mp['M_N2']/1000.0 #argon
            
            #thermal diffusivity   porosity is not involved- ryou #alpha_gas/tau was used
            #alpha_gas[c]=kgas/dens_gas/cp


#-------------porosity--------------
           
#            if f[c] == 1.0:
#                eps[c] = (mp['rho_Fe'] - mp['rho_FeO']) / mp['rho_FeO']
#            elif f[c] == 0.0:
#                eps[c] = 0.01 #initial porosity
#            elif 1.0 > f[c] > 0.0:
#                eps[c] = f[c] * (mp['rho_Fe'] - mp['rho_FeO']) / mp['rho_FeO']
                
            if run_option != 1:
                eps[c] =(mp['rho_Fe']*x_Fe[c]+mp['rho_FeO']*x_FeO[c]+ mp['rho_Fe2O3']*x_Fe2O3[c]+mp['rho_Fe3O4']*x_Fe3O4[c])/mp['rho_Fe2O3']-1.0+0.01
            
            if run_option == 1:
                eps[c] = (mp['rho_Fe']*x_Fe[c]+mp['rho_FeO']*x_FeO[c])/mp['rho_FeO']-1.0+0.01
          #  print(eps[c])



#-------------D_EFF--------------
            # # of pores n_p
            
            n_p=1.0e13
            knu_k0[c]=np.sqrt(eps[c]/np.pi/n_p)
            
            #print("knu_k0",knu_k0[c])

            D_knu_H2O[c]= 4.0/3.0*knu_k0[c] *np.sqrt(8.0*Temp[c]*constants.R/np.pi/mp['M_H2O']/1000)
            
            D_knu_H2[c]= 4.0/3.0*knu_k0[c] *np.sqrt(8.0*Temp[c]*constants.R/np.pi/mp['M_H2']/1000)
            #print("D_knu_k0",D_knu_H2[c])


            #cancel /mp[eps] if eps was in d/dt
#            Deff_H2[c]=(mp['D_H2'])*mp['eps']/mp['tau']/mp['eps']
#            Deff_H2O[c]=(mp['D_H2O'])*mp['eps']/mp['tau']/mp['eps']
            Deff_H2[c]=D_knu_H2[c]
            Deff_H2O[c]=D_knu_H2O[c]




            #print("tau",mp['tau'])

        #solid will mostly provide the thermal mass
        # effective heat capacity
        #rhoc=(1-mp['eps'])*mp['rhosolid']*mp['Cvsolid']

        
        #eqbm constant
        KeH2=np.exp(-mp['Ke_Ta']/Temp+mp['Ke_c'])
        #print("EQ_H2",KeH2) = 0.62
        
        k_m= 0.13#0.06577 #25.0
        
        
        
        

#--------------------------------------------------
#
#               Multi-reaction
#           Fe2O3->Fe3O4->FeO->Fe
#
#--------------------------------------------------


 #----------------------gas------------------------
        if run_option == 1:
            #species concentration
            cH2= solver(cH2_init, ncells, t, dx, dt, -Deff_H2, -impl_rxnsrc[0,:], cH2*eps, k_m)
            cH2O = solver(cH2O_init, ncells, t, dx, dt, -Deff_H2O, impl_rxnsrc[0,:], cH2O*eps, k_m)
        
        
        if run_option != 1:
            cH2= solver(cH2_init, ncells, t, dx, dt, -Deff_H2, -(impl_rxnsrc[0,:]+impl_rxnsrc[1,:]+impl_rxnsrc[3,:]), cH2*eps, k_m)
            cH2O = solver(cH2O_init, ncells, t, dx, dt, -Deff_H2O, (impl_rxnsrc[0,:]+impl_rxnsrc[1,:]+impl_rxnsrc[3,:]), cH2O*eps, k_m)
        
        

 #----------------------solid------------------------

        if run_option != 1:


#No.1
#---------3Fe2O3 + H2 → 2Fe3O4 + H2O--------
            kH2_1=166.37*np.exp(-89130/constants.R/Temp)
            #rrate=kH2_1*cH2*cFe2O3
            

            alpha_1[:]=np.minimum(np.maximum((mp['c_0_fe2o3'] - cFe2O3[:])/mp['c_0_fe2o3'], 1e-5), 1 - 1e-5)
            
           # print("alpha_1",alpha_1) #start from 0
            f_alpha[:] = 2.0 * (1.0 - alpha_1[:]) * (-np.log(1.0 - alpha_1[:]))**(1.0/2.0)
            #print(f_alpha) # 0.2
          
          
           # f_alpha[:]=1.0
            impl_rxnsrc[1,:]=kH2_1*cH2[:]*f_alpha[:]#*mp['c_0_fe2o3']
    
    
            s_src[1,:]=kH2_1*cH2[:]*cFe2O3[:]* 2.0*(-np.log((cFe2O3[:]-1e-5)/mp['c_0_fe2o3']))**(1.0/2.0)
            #s_src[2,:]=kH2_1*cH2[:]*cFe3O4[:]* 2.0*(-np.log(cFe3O4[:]/mp['c_0_fe3o4']))**(1.0/2.0)
            
            cFe3O4[:] += 2.0*s_src[1,:]*dt
            cFe2O3[:] -= 3.0*s_src[1,:]*dt
            cFe2O3[:]= np.maximum(cFe2O3[:], 2e-5)
            #print(impl_rxnsrc[1,:])

            
          #  print("cFe3O4",cFe3O4)
          #  print("cFe2O3",cFe2O3)
            
        
##---------Fe3O4 + 4H2 → 3Fe + 4H2O--------
#            kH2_2=1.46e5*np.exp(-70410/constants.R/Temp)
#           # rrate=kH2_2*cH2**4.0*cFe3O4
#            rrate=kH2_2*cH2*cFe3O4
#            impl_rxnsrc[2,:]=rrate
#            cFe[:] += 3.0*impl_rxnsrc[2,:]*dt
#            cFe3O4[:] -= impl_rxnsrc[2,:]*dt
        

#No.3
#---------Fe3O4 + H2 → 3FeO + H2O--------

            kH2_3=11.0*np.exp(-61505/constants.R/Temp)
            #kH2_3=600.0*np.exp(-77300/constants.R/Temp)
            #avaliable fe3o4
            
            # avlb cFe3O4
            avlb[:]=alpha_1[:]/mp['c_0_fe2o3']*mp['c_0_fe3o4']
            avlb_1[:]=avlb[:]

            
            s_src[3,:]=kH2_3*cH2*3.0* mp['c_0_fe3o4']**(1.0/3.0)*cFe3O4[:]**(2.0 / 3.0)
            #s_src[4,:]=kH2_3*cH2*3.0* mp['c_0_feo']**(1.0/3.0)*cFeO[:]**(2.0 / 3.0)
          
            cFeO[:] += 3.0*s_src[3,:]*dt
            cFe3O4[:] -= s_src[3,:]*dt
            cFe3O4[:]= np.maximum(cFe3O4[:], 0)
           # print("cFe3O4_1",cFe3O4)
           
           
            csmp[:]=avlb[:]-cFe3O4[:]
            alpha[:]=np.minimum(np.maximum(np.abs(csmp[:]/avlb[:]), 1e-5), 1 - 1e-5)
         #   print("avlb",avlb)
         #   print("csmp",csmp)
#            print("cFe3O4",cFe3O4)
         #   print("alpha",alpha)
            for c in range(ncells):
                if alpha[c] > 0.8:
                    f_alpha[c] = (1.0 - alpha[c])**(2.0 / 3.0)
                else:
                    f_alpha[c] = 1.0 - alpha[c]
            
              # f_alpha[:]=1.0
            impl_rxnsrc[3,:]=kH2_3*cH2*f_alpha[:]#*cFe3O4
            
            
           
           
           
#
##---------Fe2O3 + H2 → 2FeO + H2O--------
#            kH2_4=80*np.exp(-66516/constants.R/Temp)
#            rrate=kH2_4*cH2*cFe2O3
#            impl_rxnsrc[4,:]=rrate
#            cFeO[:] += 2.0*impl_rxnsrc[4,:]*dt
#            cFe2O3[:] -= impl_rxnsrc[4,:]*dt
#        
        

        

        # Run only the last reaction if the input is 1
        if run_option == 1 or run_option != 1:
 
#No.5
#---------Fe(1−x)O + H2 → (1−x)Fe + H2O--------
            #kh2 in m/s
            #kH2=mp['k0']*np.exp(-mp['Ea']/constants.R/Temp)
            kH2=20.0*np.exp(-63597/constants.R/Temp)

            
            avlb[:]= (avlb[:]-cFe3O4[:])/mp['c_0_fe3o4']*mp['c_0_feo']

            
            #s_src[5,:]=kH2*cH2*3.0* mp['c_0_fe']**(1.0/3.0)*cFe[:]**(2.0 / 3.0)
            s_src[6,:]=kH2*cH2*3.0* mp['c_0_feo']**(1.0/3.0)*cFeO[:]**(2.0 / 3.0)
        
            cFe[:] += s_src[6,:]*dt
            cFeO[:] -= s_src[6,:]*dt
            cFeO[:]= np.maximum(cFeO[:], 0)
            #print("cFeO_1",cFeO)
            
            
            csmp[:]=avlb[:]-cFeO[:]
            alpha[:]=np.minimum(np.maximum(np.abs(csmp[:]/avlb[:]), 1e-5),1 - 1e-5 )
          #  print("avlb5",avlb)
          #  print("csmp5",csmp)
#            print("cFeO",cFeO)
         #   print("alpha5",alpha)
            for c in range(ncells):
                if alpha[c] > 0.8:
                    f_alpha[c] = (1 - alpha[c])**(2.0 / 3.0)
                else:
                    f_alpha[c] = (1 - alpha[c])**(1.0 / 2.0)
                
            #f_alpha[:]=1.0
            #print("KH2",kH2) =0.01
            ##Reaction Source
            impl_rxnsrc[0,:]=kH2*cH2*f_alpha[:]
        
           # break
        
#--------------------------------------------------
#
#             conversion degree
#            Fe2O3->Fe3O4->FeO->Fe
#
#--------------------------------------------------
        
        # One reaction: FeO
        if run_option == 1:
            f = cFe[:]/mp['c_0']
            x_Fe=cFe[:]/mp['c_0']
            x_FeO=cFeO[:]/mp['c_0']
        
        else:
            # conversion degree (f) for each cell
            f = cFe[:] / mp['c_0']
            tol_solid=cFeO[:]+cFe[:]+cFe2O3[:]+cFe3O4[:]
            x_Fe=cFe[:]/tol_solid
            x_Fe3O4=cFe3O4[:]/tol_solid
            x_FeO=cFeO[:]/tol_solid
            x_Fe2O3=1.0-(x_Fe+x_Fe3O4+x_FeO)
            #print("", )
        
        # Initialize accumulators
        xfe_integral = 0
        x_0_integral = 0
        # conversion degree (f) for the grain
        for c in range(ncells):
            xfe_integral += x_Fe[c]*dx
            x_0_integral += 1.0*dx
            f_pde=xfe_integral/x_0_integral


#--------------------------------------------------
#
#               Multigrain Model
#
#--------------------------------------------------
#
#        #no overflow of f
#        for c in range(ncells):
#            if(fsolid[c]<1.0):
#                rrate[c]*=(1.0-fsolid[c])**(2.0/3.0)
#            else:
#                rrate[c]=0.0
#

#        impl_rxnsrc[1,:]=rrate
#        impl_rxnsrc[1,:]=0.0#mp['facH']*rrate
#        impl_rxnsrc[2,:]=0.0#mp['facHp']*rrate
#        
        #print("src:",impl_rxnsrc[0,:]) being positive
#
#        # Source Term for Solid Fraction Change
#        #mol/s / (m3 * mol/m3) = 1/s
#                #fsrc=(impl_rxnsrc[0,:]*cH2+impl_rxnsrc[1,:]*cH+impl_rxnsrc[2,:]*cHp)/(Vg*mp['rho_O'])
#        #print("fsrc:",fsrc)
#
#        #fsrc=(impl_rxnsrc[0,:]*cH2)/(Vg*mp['rho_O'])
#        fsrc=(impl_rxnsrc[0,:])/(Vg*mp['rho_O'])
#
#       # print("fsrc:",fsrc)
#
#        #volume scaling
#        vscale=(1-mp['eps'])/Vg/conc_total/mp['eps'] #
#        #vscale=(1-mp['eps'])/Vg # if eps was d(eps)/dt
#        
#        
#
#        #Update Solid Fraction;
#        fsolid=fsolid+fsrc*dt
#        #print("fsolid",fsolid)
#        
#        
#        
#        
#        #Temp=solve_temp(Temp_init,thermalcond/rhoc,mp,dx,time,dt,ncells)
#        
      

        
#        
##--------------------------------------------------
##
##               Shrinking Core Model
##
##--------------------------------------------------

    #while(t<fintime and nsteps<maxsteps):
        if(nsteps<maxsteps and r_c[nsteps] <= 0 ):
        # mean over cells
            cH2_mean = 3.58#np.mean(cH2)
            cH2O_mean = np.mean(cH2O)
            kH2_mean=np.mean(kH2)
        
        #knu_k0=1e-9
            D_knu_H2_mean= 4.0/3.0*1.086e-7 *np.sqrt(8.0*np.mean(Temp)*constants.R/np.pi/mp['M_H2']/1000)
        #print(D_knu_H2_mean)
        
        
        
#        #mass transfer coef. Kg
#        kg_h2= 2.0*(mp['D_H2'])/mp['rg']
#        #kg_h2O= 2.0*(mp['D_H2O'])/mp['rg']
#         
#        # Characteristic time
#        tau_gas_film = mp['c_0'] * mp['rg'] / (3 * b * (kg_h2 * cH2_mean))
#        tau_ash_layer = mp['c_0'] * mp['rg'] ** 2 / (6 * b * (D_knu_H2_mean *cH2_mean))
#        tau_reaction = mp['c_0'] * mp['rg'] / (b * (kH2_mean * cH2_mean ))
#        
#
#        # Conversion for Gas Film Control
#        X_gas_film = safe_conversion(t, tau_gas_film, 1/3)
#
#        # Conversion for Ash Layer Control
#        X_ash_layer = np.zeros_like(t)
#        if t <= tau_ash_layer:
#                X_ash_layer= 1 - 3 * (1 - t / tau_ash_layer)**(2/3) + 2 * (1 - t / tau_ash_layer)
#        else:
#                X_ash_layer= 1  # Full conversion after tau_ash_layer
#
#        # Conversion for Reaction Control
#        X_reaction = safe_conversion(t, tau_reaction, 1/3)
#
#




#        # Combination-Rate of change of core radius function
#        # sphere
#        dr_c_dt = (b / mp['c_0']) / (
#        (r_c[nsteps]**2) / (mp['rg']**2 * ((kg_h2 * cH2_mean))) +
#        ((mp['rg']  - r_c[nsteps]) * r_c[nsteps]) / (mp['rg'] * ((D_knu_H2_mean *cH2_mean))) +
#        (1 / (kH2_mean * cH2_mean ))
#        )
#


        # slab w/o gas film
            dr_c_dt = (b / mp['c_0']) / (
                r_c[nsteps] / ((D_knu_H2_mean *cH2_mean)) + (1 / (kH2_mean * cH2_mean ))
                    )
        
            r_c[nsteps+1] =  r_c[nsteps]-dr_c_dt * (dt)  #
        
         
            if r_c[nsteps+1] <= 0:  # Core radius reaches zero
        #else:
                r_c[nsteps+1] = 0  # Cap it at zero
           # break
        
#            with open("rc_output.txt", "a") as file:
#            #file.write("Time (s)\tCore Radius (r_c) (m)\n")
#                file.write(f"{t:.6e}\t{r_c[nsteps]:.6e}\n")
#            
#
#            with open("X_reaction.txt", "a") as file:
#                file.write(f"{t:.6e}\t{X_gas_film:.6e}\t{X_ash_layer:.6e}\t{X_reaction:.6e}\n")


            conversion = 1- r_c[nsteps+1]/mp['rg']
        #conversion= 1 - (((4/3) * np.pi * r_c[nsteps+1]**3 )/ ((4/3) * np.pi * mp['rg']**3))
        
        
        #print("conversion",conversion)

        # Append the current time and conversion to the list
        #conversion_vs_time.append([t, conversion])
            scm_f.append(conversion)
 
 
#--------------------------------------------------
#
#               output
#
#--------------------------------------------------


        if(nsteps%outputint==0):
            
            plt.figure()
            #plt.title("Fraction of Iron Species")
            plt.plot(x, x_Fe, label="x_Fe", color="blue", marker="o", markevery=10)
            plt.plot(x, x_FeO, label="x_FeO", color="green", marker="s", markevery=10)
            plt.plot(x, x_Fe2O3, label="x_Fe2O3", color="red", marker="^", markevery=10)
            plt.plot(x, x_Fe3O4, label="x_Fe3O4", color="purple", marker="x", markevery=10)
            plt.xlabel("x(m)")
            plt.ylabel("Fraction of Iron Species")
            plt.legend()
            plt.show()
        
#
            plot_separate_figures(x, cH2, cH2O)
            
            plt.figure()
            plt.title("porosity")
            plt.plot(x,eps)
            plt.xlabel("x(m)")
            plt.ylabel("porosity")
            plt.show()

            plt.figure()
            plt.title("reaction rate")
            plt.plot(x,impl_rxnsrc[0,:])
            plt.xlabel("x(m)")
            plt.ylabel("reaction rate")
            plt.show()

            
            
        
            
        time_values.append(t)
        f_values.append(f_pde)
     #   f_mean.append(np.mean(f))

       
       # animate_concentration(x, cH2, cH2O, dt)
        

       # print("src:",impl_rxnsrc[0,:]) #impl_rxnsrc[0,:]: always being positive
        #print("ch2:",cH2)
        #print("ch2O:",cH2O)


# output
        t+=dt
        nsteps+=1
        if(nsteps%screenint==0):
            print("iter,t,  f_pde,scm_f,Temp,cH2,ch2o:",nsteps,t,f_pde,\
                    np.mean(Temp),np.mean(cH2),np.mean(cH2O))
#
        if(nsteps%outputint==0):
            outint+=1
            np.savetxt("data_%4.4d"%(outint),np.transpose(np.vstack((x,cH2, cH2O, x_Fe, x_FeO,f))),delimiter="  ")
#

    # End the timer
    end_time = time.time()
    # Calculate the execution time
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
    
    np.savetxt("final_data",np.transpose(np.vstack((x,cH2, cH2O, x_Fe, x_FeO,f))),delimiter="  ")

    plot_separate_figures(x, cH2, cH2O)
    plot_separate_figures_solid(x, x_Fe, x_FeO)
    
    plt.figure()
    #plt.title("Fraction of Iron Species")
    plt.plot(x, x_Fe, label="x_Fe", color="blue", marker="o", markevery=10)
    plt.plot(x, x_FeO, label="x_FeO", color="green", marker="s", markevery=10)
    plt.plot(x, x_Fe2O3, label="x_Fe2O3", color="red", marker="^", markevery=10)
    plt.plot(x, x_Fe3O4, label="x_Fe3O4", color="purple", marker="x", markevery=10)
    plt.xlabel("x(m)")
    plt.ylabel("Fraction of Iron Species")
    plt.legend()
    plt.show()

    plt.figure()
    plt.title("porosity")
    plt.plot(x,eps)
    plt.xlabel("x(m)")
    plt.ylabel("porosity")
    plt.show()

    plt.figure()
    plt.title("reaction rate")
    plt.plot(x,impl_rxnsrc[0,:])
    plt.xlabel("x(m)")
    plt.ylabel("reaction rate")
    plt.show()
    
        #plot (f v.s. t)
    plt.plot(time_values, f_values, label='PDE_Conversion Degree', color='b')
  #  plt.plot(time_values, f_mean, label='PDE_Conversion Degree', color='b')
    plt.plot(time_values, scm_f, label='SCM_Conversion Degree', color='g')
    plt.xlabel('Time [s]')
    plt.ylabel('Conversion Degree (f)')
    plt.title('Conversion Degree vs Time')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("tau_gas_film:", tau_gas_film)
    print("tau_ash_layer:", tau_ash_layer)
    print("tau_reaction:", tau_reaction)


   # np.savetxt("conversion_vs_time.txt", np.column_stack((time_values, scm_f,f_mean)), delimiter="  ", header="Time(s)  Conversion", comments='')

