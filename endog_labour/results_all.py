import dill as pickle
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.stats import gaussian_kde
import numpy as np
from tabulate import tabulate
from gini import gini 
import matplotlib.cm as cm
from numba import jit
from endog_labour import ConsumerProblem, FirmProblem, Operator_Factory
from statsmodels.nonparametric.smoothers_lowess import lowess


def runres(namefilein, namefileout,model_path,T, mass=1):


    #====davila 1 results====#

    results_in = open('{}'.format(namefilein), 'rb')

    name = namefileout 

    Results = pickle.load(results_in)

    results_in.close()

    #=====clean up the files, add entries====#

    #Results["CP"]["CP_r"] = Results["CP"]["CP_r,CP_w"][0]
    #Results["CP"]["CP_w"] = Results["CP"]["CP_r,CP_w"][1]
    Results["CP"]["CP_gini_a"] = gini(Results["CP"]["CP_a"])
    Results["CP"]["CP_gini_i"] = gini(Results["CP"]["CP_z_rlz"])

    Results["IM"]["im_gini_a"] = gini(Results["IM"]["IM_a"])
    Results["IM"]["im_gini_i"] = gini(Results["IM"]["IM_z_rlz"])


    # load the model parameter values from saved file 

    model = open('Settings/pjmas2.mod', 'rb')
    model_in = pickle.load(model)
    name = model_in["filename"]
    model.close()

    # initialise the consumer and firm class, load operators 


    Gamma  =    np.array([[0.746464, 0.252884, 0.000652, 0.000000, 0.000000, 0.000000,0.000000],
            [0.046088,  0.761085,  0.192512,  0.000314,  0.000000,  0.000000,  0.000000],
            [0.000028,  0.069422,  0.788612,  0.141793,  0.000145,  0.000000,  0.000000],
            [0.000000,  0.000065,  0.100953,  0.797965,  0.100953,  0.000065,  0.000000],
            [0.000000,  0.000000,  0.000145,  0.141793,  0.788612,  0.069422,  0.000028],
            [0.000000,  0.000000,  0.000000,  0.000314,  0.192512,  0.761085,  0.046088],
            [0.000000,  0.000000,  0.000000,  0.000000,  0.000652,  0.252884,  0.746464]])

    Gamma_bar   =   np.array([0.0154,   0.0847,    0.2349,    0.3300,    0.2349,    0.0847,    0.0154])


    e_shocks = np.array([np.exp(-1.385493),np.exp(-0.923662),np.exp(-0.461831),
            np.exp(0.000000),
            np.exp(0.461831),
            np.exp(0.923662),
            np.exp(1.385493)])

    e_shocks = e_shocks/np.dot(e_shocks, Gamma_bar)


    cp = ConsumerProblem(Pi = Gamma,
                        z_vals = e_shocks,
                        gamma_c= model_in["gamma_c"],
                        gamma_l = model_in["gamma_l"],
                        A_L = model_in["A_L"],
                        grid_max = 65,
                        grid_size= 200,
                        beta = .945)


    fp = FirmProblem(delta = .083)

    #===unpack SP and IM results====#

    results_FB = Results["FB"] 
    IM = Results["IM"] 
    CP = Results["CP"] 
    CF = Results["CF"] 

    #===Plot Poliy Functions for Assets=====#

    im_policy = IM["IM_policy"]
    CP_policy = CP["CP_policy"]
    CF_policy = CF["CF_policy"]

    h_im = im_policy[0]
    l_im = im_policy[1]
    L_im = (1 - l_im)*100
    L_imf = lambda x, iz: np.interp(x,cp.asset_grid, L_im[iz,:])


    h_CP = CP_policy[0]
    l_CP = CP_policy[1]
    L_CP = (1 - l_CP)*100
    L_CPf = lambda x, iz: np.interp(x,cp.asset_grid, L_CP[iz,:])
    C_CPf = lambda x, iz: x*(1+ CP["CP_r"]) + 1e-2*CP["CP_w"]*L_CPf(x,iz)*cp.z_vals[iz] - np.interp(x,cp.asset_grid, h_CP[iz,:])
    CP_Lambda_H = CP['CP_Lambda']
    CP_cap_lab_ratio = CP['CP_K']/CP['CP_L']
    CP_Lambda_E = - (CP_Lambda_H*CP_cap_lab_ratio/cp.beta)
    ducprime = lambda xprime, z_prime: cp.du((C_CPf(xprime, z_prime)))

    tau_K = np.zeros((len(cp.z_vals), len(cp.asset_grid)))

    for z in range(len(cp.z_vals)):
        for i in range(len(cp.asset_grid)):
            x = cp.asset_grid[i]
            xprime = np.interp(x,cp.asset_grid, h_CP[z,:])
            ducprime_vec = np.zeros(len(cp.z_vals))
            for z_prime_ind in range(len(cp.z_vals)):
                ducprime_vec[z_prime_ind] = ducprime(xprime, z_prime_ind)
            E_ducprime_vec = np.inner(cp.Pi[z,:], ducprime_vec)
            tau_K[z,i] = - CP_Lambda_H/(E_ducprime_vec*(1+CP["CP_r"]))

    tau_L = np.zeros((len(cp.z_vals), len(cp.asset_grid)))

    for z in range(len(cp.z_vals)):
        for i in range(len(cp.asset_grid)):
            x = cp.asset_grid[i]
            tau_L[z,i] = - (CP_Lambda_E/(ducprime(x,z)*CP["CP_w"]))



    h_CF = CF_policy[0]
    l_CF = CF_policy[1]
    L_CF = (1 - l_CF)*100
    L_CFf = lambda x, iz: np.interp(x,cp.asset_grid, L_CF[iz,:])

    fig_hpol, ax_hpol = plt.subplots(nrows =3, ncols=1, sharex=True,figsize=(6,6))
    
    sns.set_color_codes("dark")

    titles = ['Low productivity (e = 0.22)','Medium productivity (e = 0.34)','High productivity (e = 0.54)']
    y_vals = [[h_im[i,:],h_CP[i,:],h_CF[i,:]] for i in [0,3,6]]

    for ax, title, y in zip(ax_hpol.flat, titles, y_vals):
        ax.plot(cp.asset_grid[0:200], y[0][0:200], label = "IM", color = 'b', linewidth = .9)
        ax.plot(cp.asset_grid[0:200], y[1][0:200], label = "CP", color = 'r', ls = "dashed", linewidth = .9)
        ax.plot(cp.asset_grid[0:200], y[2][0:200], label = "IM with CP prices", color = 'g', ls = "dotted", linewidth = .9)
        #ax.plot(cp.asset_grid[0:20], cp.asset_grid, ":", label = "45 deg. line", color = "k", linewidth = .5)
        ax.set_title(title, fontsize= 10)
        #ax.grid(True)
        #ax.tight_layout()

    #for ax in ax_hpol.flat:
    #    ax.set(xlabel='assets', ylabel='next period assets')
    ax_hpol.flat[2].legend(loc=9, ncol=3,  bbox_to_anchor=(0.5, -0.65), frameon=False)
    fig_hpol.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel('Assets')
    plt.ylabel('Next period assets')
    
    

    #fig_hpol.suptitle("Equilibrium Policy Functions", fontsize=12, x=0.55)
    #colors = ["windows blue", "pale red", "greyish", "faded green", "dusty purple"]


   
    fig_hpol.tight_layout()
    sns.despine()


    fig_hpol.savefig("{}_hpol.eps".format(name))


    fig_hpol, ax_hpol = plt.subplots(nrows =3, ncols=1, sharex=True,figsize=(6,6))
    
    sns.set_color_codes("dark")

    titles = ['Low assets','Medium assets','High assets'.format(cp.asset_grid[5],cp.asset_grid[100],cp.asset_grid[190])]
    y_vals = [[h_im[:,i],h_CP[:,i],h_CF[:,i]] for i in [10,50,100]]

    for ax, title, y in zip(ax_hpol.flat, titles, y_vals):
        ax.plot(cp.z_vals, y[0], label = "IM", color = 'b', linewidth = .9)
        ax.plot(cp.z_vals, y[1], label = "CP", color = 'r', ls = "dashed", linewidth = .9)
        ax.plot(cp.z_vals, y[2], label = "IM with CP prices", color = 'g', ls = "dotted", linewidth = .9)
        #ax.plot(cp.asset_grid[0:20], cp.asset_grid, ":", label = "45 deg. line", color = "k", linewidth = .5)
        ax.set_title(title, fontsize= 10)
        #ax.grid(True)
        #ax.tight_layout()

    ax_hpol.flat[2].legend(loc=9, ncol=3,  bbox_to_anchor=(0.5, -0.65), frameon=False)
    fig_hpol.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel('Productivity')
    plt.ylabel('Next period assets')
    fig_hpol.tight_layout()
    sns.despine()


    fig_hpol.savefig("{}_hpol_prod.eps".format(name))


    #===Plot Poliy Functions for Hours Worked=====#


    x = np.linspace(0, cp.grid_max, 200)

    fig_lpol, ax_lpol = plt.subplots(nrows=3, ncols=1, sharex=True, sharey = True,figsize=(6,6))
    sns.set_color_codes("dark")
    titles = ['Low productivity (e = 0.22)','Medium productivity (e = 0.34)','High productivity (e = 0.54)']
    #titles = ['e = {0:.2f}'.format(title_list[i]) for i in range(len(cp.z_vals))]
    y_vals = [[L_imf(x,i),L_CPf(x,i),L_CFf(x,i)] for i in [0,3,6]]

    for ax, title, y in zip(ax_lpol.flat, titles, y_vals):
        ax.plot(cp.asset_grid[0:200], y[0][0:200], label = "IM", color = 'b', linewidth = .9)
        ax.plot(cp.asset_grid[0:200], y[1][0:200], label = "CP", color = 'r', ls = "dashed", linewidth = .9)
        ax.plot(cp.asset_grid[0:200], y[2][0:200], label = "IM with CP prices", color = 'g', ls = "dotted", linewidth = .9)
        #ax.plot(cp.asset_grid[0:20], cp.asset_grid, ":", label = "45 deg. line", color = "k", linewidth = .5)
        ax.set_title(title, fontsize= 10)
        #ax.grid(True)
        #ax.tight_layout()

    #for ax in ax_hpol.flat:
    #    ax.set(xlabel='assets', ylabel='next period assets')
    ax_lpol.flat[2].legend(loc=9, ncol=3,  bbox_to_anchor=(0.5, -0.65), frameon=False)

    #fig_hpol.suptitle("Equilibrium Policy Functions", fontsize=12, x=0.55)
    #colors = ["windows blue", "pale red", "greyish", "faded green", "dusty purple"]
    fig_lpol.add_subplot(111, frameon = False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel('Assets')
    plt.ylabel('Hours worked')
    
    #fig_lpol.text(0.04, 0.5, 'Hours worked', va='center', rotation='vertical')
    #fig_lpol.text(0.5, 0.04, 'Assets', ha='center')
   
    fig_lpol.tight_layout()
    sns.despine()

    fig_lpol.savefig("{}_lpol.eps".format(name))


    fig_lpol, ax_lpol = plt.subplots(nrows=3, ncols=1, sharex=True, sharey = True, figsize=(6,6))
    
    sns.set_color_codes("dark")

    titles = ['Low assets','Medium assets','High assets'.format(cp.asset_grid[5],cp.asset_grid[100],cp.asset_grid[190])]
    y_vals = [[L_im[:,i],L_CP[:,i],L_CF[:,i]] for i in [10,50,100]]

    for ax, title, y in zip(ax_lpol.flat, titles, y_vals):
        ax.plot(cp.z_vals, y[0], label = "IM", color = 'b', linewidth = .9)
        ax.plot(cp.z_vals, y[1], label = "CP", color = 'r', ls = "dashed", linewidth = .9)
        ax.plot(cp.z_vals, y[2], label = "IM with CP prices", color = 'g', ls = "dotted", linewidth = .9)
        #ax.plot(cp.asset_grid[0:20], cp.asset_grid, ":", label = "45 deg. line", color = "k", linewidth = .5)
        ax.set_title(title, fontsize= 10)
        #ax.grid(True)
        #ax.tight_layout()

    #for ax in ax_hpol.flat:
    #    ax.set(xlabel='assets', ylabel='next period assets')
    ax_lpol.flat[2].legend(loc=9, ncol=3,  bbox_to_anchor=(0.5, -0.65), frameon=False)
    fig_lpol.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel('Productivity')
    plt.ylabel('Hours worked')

    fig_lpol.tight_layout()
    sns.despine()

    fig_lpol.savefig("{}_lpol_prod.eps".format(name))

    """
    #=====Tabulate Main Results=====#

    mainres_CP = ["CP", CP["CP_r"]*100,CP["CP_w"], CP["CP_K"], CP["CP_H"]*100, CP["CP_L"]*100,\
                    CP["CP_gini_a"], CP["CP_gini_i"]]

    mainres_IM = ["CE", IM["im_r"]*100, \
                    IM["im_w"], IM["im_K"], IM["im_H"]*100, \
                    IM["im_L"]*100, IM["im_gini_a"], IM["im_gini_i"] ]

    mainres_FB = ["RA", results_FB["fb_r"]*100, fp.AA*(1-fp.alpha)*(np.power(results_FB["fb_K"]/results_FB["fb_H"],fp.alpha)), \
                    results_FB["fb_K"], results_FB["fb_H"]*100, \
                    results_FB["fb_H"]*100, 0, 0]


    header = ["Interest rate", "Wage", "Capital", "Hours", "Labour", "Wealth gini", "Income gini"]


    table= [mainres_FB, mainres_IM, mainres_CP]

    print(tabulate(table, headers = header,  tablefmt="latex_booktabs", floatfmt=".2f"))

    restab = open("{}_tab.tex".format(name), 'w')

    restab.write(tabulate(table, headers = header,  tablefmt="latex_booktabs", floatfmt=".2f"))

    restab.close()

    caption = "Model parameters are $\beta$ = {}, $\alpha$ = {}, $\delta$ = {}, $\gamma_c$ = {}, $\gamma_l$ = {}, $A_L$ = {}, $A$ = {} and size of grid is {}. The probmatrix is {} and shocks are {}".\
                format(cp.beta, fp.alpha, fp.delta, cp.gamma_c, cp.gamma_l, cp.A_L, fp.AA, cp.grid_size, cp.Pi, cp.z_vals)

    print(caption)

    restab = open("{}_caption.tex".format(name), 'w')
    restab.write(caption)
    restab.close()
    """

    ##====set Plot style====#

    def set_style():
        # This sets reasonable defaults for font size for
        # a figure that will go in a paper
        sns.set_context("paper")
        
        # Set the font to be serif, rather than sans
        sns.set(font='serif')
        
        # Make the background white, and specify the
        # specific font family
        sns.set_style("white", {
            "font.family": "serif",
            "font.serif": ["Times", "Palatino", "serif"]
        })

    set_style()


    #sns.set_color_codes("light")
    EDEN2, KDEN2 = 0,0

    L_val_im = 1-IM["IM_l_val"]
    a_im = IM["IM_a"]
    a_cp = CP["CP_a"]
    L_val_cp = 1-CP["CP_l_val"]

    filtered = lowess(L_val_cp, a_cp, is_sorted=False, frac=0.025, it=0, xvals =cp.asset_grid)

    plt.plot(cp.asset_grid, filtered, 'b')
    plt.savefig('loess.png')

    if mass ==1:
        #==== Plotting Distribution of Labour And Capital====##


        h_val = IM["IM_h_val"]
        z_rlz = IM["IM_z_rlz"]
        L_val = 1-IM["IM_l_val"]
        a = IM["IM_a"]
        MUC = 1/cp.du(IM["IM_a"]*(1+IM["IM_r"]) + IM["IM_w"]*z_rlz - h_val)



        h_val_cp = CP["CP_h_val"]
        z_rlz_cp = CP["CP_z_rlz"]
        L_val_cp = 1-CP["CP_l_val"]
        a_cp = CP["CP_a"]


        MUC_GRID = np.linspace(0, np.max(MUC), 100)

        def Phi(z_rlz, MUC, MUC_GRID):
            # mu is a 1D array with dim+ 1 elements
            # y is a 1D array with dim+1 elements 
            # K is a 1D array with dim+1 elements 
            delta = np.equal(MUC, MUC_GRID[0]) # select desired type here - maybe some int?
            delta.astype(int)
            delta = np.expand_dims(delta, 0) # expand for concating

            for i in range(1, len(MUC_GRID)):
                new = np.logical_and(np.greater(MUC,MUC_GRID[i-1]),np.less_equal(MUC,MUC_GRID[i]))
                delta = np.concatenate([delta,np.expand_dims(new, 0)], 0)

            mu_prime = np.matmul(z_rlz, np.transpose(delta))
            return mu_prime

        EDEN = Phi(z_rlz, MUC, MUC_GRID)/(IM["IM_L"]*T)
        KDEN = Phi(a, MUC, MUC_GRID)/(IM["IM_K"]*T)
        print('done_NUC')

        # IM asssets on assets 
        KDEN_assets = Phi(a,a, cp.asset_grid)/(IM["IM_K"]*T)
        KDEN_assets_aux = np.random.choice(cp.asset_grid, 100000, p=KDEN_assets)
        density_k_assets = gaussian_kde(KDEN_assets_aux)
        density_k_assets.covariance_factor = lambda : .2
        density_k_assets._compute_covariance()
        print('done_IM_a')

        #IM hours on assets 
        lDEN_assets = Phi(L_val,a, cp.asset_grid)/(IM["IM_H"]*T)
        lDEN_assets_aux = np.random.choice(cp.asset_grid, 100000, p=lDEN_assets)
        density_l_assets = gaussian_kde(lDEN_assets_aux)
        density_l_assets.covariance_factor = lambda : .35
        density_l_assets._compute_covariance()
        print('done_IM')
        density_p_assets = gaussian_kde(a)
        density_p_assets.covariance_factor = lambda : .35
        density_p_assets._compute_covariance()

        # CP asssets on assets 
        KDEN_assets_cp = Phi(a_cp,a_cp, cp.asset_grid)/(CP["CP_K"]*T)
        KDEN_assets_aux_cp = np.random.choice(cp.asset_grid, 100000, p=KDEN_assets_cp)
        density_k_assets_cp = gaussian_kde(KDEN_assets_aux_cp)
        density_k_assets_cp.covariance_factor = lambda : .2
        density_k_assets_cp._compute_covariance()

        # CP hours on assets 
        lDEN_assets_cp = Phi(L_val_cp,a_cp, cp.asset_grid)/(CP["CP_H"]*T)
        lDEN_assets_aux_cp = np.random.choice(cp.asset_grid, 100000, p=lDEN_assets_cp)
        density_l_assets_cp = gaussian_kde(lDEN_assets_aux_cp)
        density_l_assets_cp.covariance_factor = lambda : .25
        density_l_assets_cp._compute_covariance()

        density_p_assets_cp = gaussian_kde(a_cp)
        density_p_assets_cp.covariance_factor = lambda : .35
        density_p_assets_cp._compute_covariance()

        print(np.sum(EDEN))

        CGRAVE = np.mean(MUC*z_rlz)/IM["IM_L"]
        CGRAVA = np.mean(MUC*a)/IM["IM_K"]
        CGRAVA_ASSETS = IM["IM_K"]
        CGRAVA_ASSETS_cp = CP["CP_K"]

        print(CGRAVA)

        EDEN2  = np.random.choice(MUC_GRID, 100000, p=EDEN)
        KDEN2  = np.random.choice(MUC_GRID, 100000, p=KDEN)

        density_e = gaussian_kde(EDEN2)
        density_k = gaussian_kde(KDEN2)
        density_p = gaussian_kde(MUC)

        xs = np.linspace(0,np.max(MUC),200)
        density_e.covariance_factor = lambda : .2
        density_e._compute_covariance()

        density_k.covariance_factor = lambda : .2
        density_k._compute_covariance()

        density_p.covariance_factor = lambda : .2
        density_p._compute_covariance()
        
        figmass, axmass = plt.subplots()

        axmass.plot(xs, density_k(xs), label = "Capital", color = "k", ls= "solid",linewidth=.8)
        axmass.plot(xs, density_e(xs), color = "k", ls = "dashed", label = "Effective labor", linewidth=.8)
        axmass.plot(xs, density_p(xs),label = "Population", color = "k", ls = "dotted", linewidth=.8)



        
        axmass.fill_between(xs, 0, density_p(xs), alpha = .8, facecolor='#f97306')
        axmass.fill_between(xs, 0, density_e(xs),alpha = .5, facecolor='g')
        axmass.fill_between(xs, 0, density_k(xs), alpha =.5, facecolor='#cea2fd')

        
        #with sns.color_palette(sns.color_palette("colorblind", 10)):
         #   sns.kdeplot(KDEN2, shade=True, label="Capital", ls = "solid", lcolor = "k", gridsize= 100)
        #   sns.kdeplot(EDEN2, shade=True, label="Labor", ls = "dashed", gridsize= 100)
         #   sns.kdeplot(MUC, shade=True, label="Population", ls = "dotted", gridsize= 100)

        #axmass.bar(MUC_GRID, EDEN, label = "Lab Den", linewidth=.8, alpha=0.5)
        axmass.plot([CGRAVE], [.5], "o",color= "k", label ="Center of Labor")
        axmass.plot([CGRAVA], [.5], "d",color= "k",  label = "Center of Capital")
        axmass.set_xlabel("Inverse of marginal utility of consumption")
        #figmass.suptitle("Equilibrium Asset and Labor Density", fontsize=12, x=0.55)

        plt.xticks(np.arange(0, 3, step=1.5))

        sns.despine()

        axmass.legend(loc=9, ncol=3,  bbox_to_anchor=(0.5, -0.2), frameon=False)


        figmass.savefig("{}_mass.pdf".format(name),
        bbox_inches="tight")
        #figmass.show()
        figmass_assets, axmass_assets = plt.subplots()
        axmass_assets.plot(cp.asset_grid, density_k_assets(cp.asset_grid), label = "Asset density (IM)", color = "k", ls= "dashed",linewidth=.8)
        axmass_assets.plot(cp.asset_grid, density_k_assets_cp(cp.asset_grid), label = "Asset density (CP)", color = "k", ls= "solid",linewidth=.8)


        
        axmass_assets.fill_between(cp.asset_grid, 0, density_k_assets(cp.asset_grid), alpha = .8, facecolor='#f97306')
        axmass_assets.fill_between(cp.asset_grid, 0, density_k_assets_cp(cp.asset_grid), alpha = .8, facecolor='g')

        
        #with sns.color_palette(sns.color_palette("colorblind", 10)):
         #   sns.kdeplot(KDEN2, shade=True, label="Capital", ls = "solid", lcolor = "k", gridsize= 100)
        #   sns.kdeplot(EDEN2, shade=True, label="Labor", ls = "dashed", gridsize= 100)
         #   sns.kdeplot(MUC, shade=True, label="Population", ls = "dotted", gridsize= 100)

        #axmass.bar(MUC_GRID, EDEN, label = "Lab Den", linewidth=.8, alpha=0.5)
        axmass_assets.plot([CGRAVA_ASSETS], [.05], "o",color= "k", label ="Mean (IM)")
        axmass_assets.plot([CGRAVA_ASSETS_cp], [.05], "x",color= "k", label ="Mean (CP)")
       # axmass_assets.plot([CGRAVA], [.5], "d",color= "k",  label = "Center of Capital")
        axmass_assets.set_xlabel("Assets")
        #figmass.suptitle("Equilibrium Asset and Labor Density", fontsize=12, x=0.55)

        #plt.xticks(np.arange(0, cp.asset_grid[-1], step=1.5))

        sns.despine()

        axmass_assets.legend(loc=9, ncol=3,  bbox_to_anchor=(0.5, -0.2), frameon=False)
        figmass_assets.savefig("{}_mass_assets.pdf".format(name),
        bbox_inches="tight")


        figmass_assets, axmass_assets = plt.subplots()
        axmass_assets.plot(cp.asset_grid, density_l_assets(cp.asset_grid)/density_p_assets(cp.asset_grid), label = "Hours density (IM)", color = "k", ls= "dashed",linewidth=.8)
        axmass_assets.plot(cp.asset_grid, density_l_assets_cp(cp.asset_grid)/density_p_assets_cp(cp.asset_grid), label = "Hours density (CP)", color = "k", ls= "solid",linewidth=.8)


        
        axmass_assets.fill_between(cp.asset_grid, 0, density_l_assets(cp.asset_grid)/density_p_assets(cp.asset_grid), alpha = .8, facecolor='#f97306')
        axmass_assets.fill_between(cp.asset_grid, 0, density_l_assets_cp(cp.asset_grid)/density_p_assets_cp(cp.asset_grid), alpha = .8, facecolor='g')

        
        #with sns.color_palette(sns.color_palette("colorblind", 10)):
         #   sns.kdeplot(KDEN2, shade=True, label="Capital", ls = "solid", lcolor = "k", gridsize= 100)
        #   sns.kdeplot(EDEN2, shade=True, label="Labor", ls = "dashed", gridsize= 100)
         #   sns.kdeplot(MUC, shade=True, label="Population", ls = "dotted", gridsize= 100)

        #axmass.bar(MUC_GRID, EDEN, label = "Lab Den", linewidth=.8, alpha=0.5)
        #axmass_assets.plot([IM["IM_H"]], [.05], "o",color= "k", label ="Mean (IM)")
        #axmass_assets.plot([CP["CP_H"]], [.05], "x",color= "k", label ="Mean (CP)")
       # axmass_assets.plot([CGRAVA], [.5], "d",color= "k",  label = "Center of Capital")
        axmass_assets.set_xlabel("Assets")
        #figmass.suptitle("Equilibrium Asset and Labor Density", fontsize=12, x=0.55)

        #plt.xticks(np.arange(0, cp.asset_grid[-1], step=1.5))

        sns.despine()

        axmass_assets.legend(loc=9, ncol=3,  bbox_to_anchor=(0.5, -0.2), frameon=False)
        figmass_assets.savefig("{}_mass_hours.pdf".format(name),
        bbox_inches="tight")





    # Plot taxes (Capital wedge)
    fig_tau, ax_tau = plt.subplots(nrows=2, ncols=1)
    y_vals_tauk = [tau_K[:,10],tau_K[:,50],tau_K[:,100]]
    y_vals_taul = [tau_L[:,10],tau_L[:,50],tau_L[:,100]]


    ax_tau[0].plot(cp.z_vals, y_vals_tauk[0], color = 'b', linewidth = .9, label = "Low assets",ls = "dotted")
    ax_tau[0].plot(cp.z_vals, y_vals_tauk[1], color = 'r', linewidth = .9, label = "Medium assets",ls = "dashed")
    ax_tau[0].plot(cp.z_vals, y_vals_tauk[2], color = 'g', linewidth = .9, label = "High assets",ls = "solid")
    ax_tau[0].set_title('Capital tax', fontsize= 10)
    ax_tau[1].plot(cp.z_vals, y_vals_taul[0], color = 'b', linewidth = .9, label = "Low assets",ls = "dotted")
    ax_tau[1].plot(cp.z_vals, y_vals_taul[1], color = 'r', linewidth = .9, label = "Medium assets",ls = "dashed")
    ax_tau[1].plot(cp.z_vals, y_vals_taul[2], color = 'g', linewidth = .9, label = "High assets",ls = "solid")
    ax_tau[1].set_title('Labor income tax', fontsize= 10)

    ax_tau[1].legend(loc=9, ncol=3,  bbox_to_anchor=(0.5, -0.65), frameon=False)
    fig_tau.tight_layout()
    sns.despine()
    fig_tau.add_subplot(111, frameon=False)
    plt.xlabel('Productivity')
    plt.ylabel('Tax')
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    fig_tau.tight_layout()
    fig_tau.savefig("{}_tau_prod.eps".format(name))


    return EDEN2, KDEN2


if __name__ == "__main__":

    results_path = "/scratch/kq62/endoglabour/results_062021/results_PJmas.mod"
    results_path_out = "Results/"
    model_path = 'Settings/pjmas2.mod'

    EDEN2, KDEN2 = runres(results_path,results_path_out, model_path,1E7, mass = 0)

