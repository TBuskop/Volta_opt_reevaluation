"""
Created on Wed Nov 24 13:28:27 2021

@author: aow001 based on https://github.com/JazminZatarain/MUSEH2O/blob/main/susquehanna_model.py
"""
#Lower Volta River model

#import Python packages
import os
import numpy as np


#Install using pip and restart kernel if unavailable
import utils #install using:  pip install utils
from numba import njit #pip install numba #restart kernel (Ctrl + .)

import pandas as pd

# define path
def create_path(rest): 
    my_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.abspath(os.path.join(my_dir, rest))


class VoltaModel:
    gammaH20 = 1000.0 #density of water- 1000 kg/m3
    GG = 9.81 #acceleration due to gravity- 9.81 m/s2
    n_days_in_year = 365 #days in a year
    
    #initial conditions
    def __init__(self, l0_Akosombo, d0, n_years, n_samples, rbf, data='historic',
                 energy_storage =0,
                                  
                 treatiesBenin = 0, treatiesBurkinaFaso =0,
                 treatiesCoteIvoire = 0, treatiesTogo=0,
                 
                 Cjanuary=0,Cfebruary=0,Cmarch=0,
                 Capril=0, Cmay=0, Cjune=0,
                 Cjuly=0,Caugust=0, Cseptember=0,
                 Coctober=0,Cnovember=0, Cdecember=0,
                 
                 waterUseBenin=0, waterUseBurkinaFaso=0, 
                 waterUseCoteIvoire=0,waterUseTogo=0,
                 
                 irriDemandMultiplier=1
                 ):
        
        """

        Parameters
        ----------
        l0_Akosombo : float; initial condition (water level) at Akosombo
        d0 : int; initial start date 
        n_years : int
        rbf : callable
        data: string, historic, 100x100, 1000x1
        """

        self.init_level = l0_Akosombo  # initial water level @ start of simulation_ feet
        self.day0 = d0 # start day

        self.n_samples = n_samples
        self.energy_storage = energy_storage
        self.energy_stored = 0
                         
        self.treatiesBenin = treatiesBenin
        self.treatiesBurkinaFaso =treatiesBurkinaFaso
        self.treatiesCoteIvoire = treatiesCoteIvoire
        self.treatiesTogo=treatiesTogo
        
        self.Cjanuary=Cjanuary
        self.Cfebruary=Cfebruary
        self.Cmarch=Cmarch
        self.Capril=Capril
        self.Cmay=Cmay
        self.Cjune=Cjune
        self.Cjuly=Cjuly
        self.Caugust=Caugust
        self.Cseptember=Cseptember
        self.Coctober =Coctober
        self.Cnovember=Cnovember
        self.Cdecember=Cdecember
        
        self.waterUseBenin=waterUseBenin
        self.waterUseBurkinaFaso=waterUseBurkinaFaso 
        self.waterUseCoteIvoire=waterUseCoteIvoire
        self.waterUseTogo=waterUseTogo
        
        self.irriDemandMultiplier=irriDemandMultiplier

        self.daily_energy_usage = 12.1 #GWh
        
        self.log_level_release = True                                  
        
        # variables from the header file
        self.input_min = []
        self.input_max = []
        self.output_max = []
        self.rbf = rbf
        
        # log level / release
        self.blevel_Ak = [] # water level at Akosombo
        self.rirri = []     # water released for irrigation
        self.renv = []      # environmental flow release
        
        # historical record (1965- 2016) and 1000 year simulation horizon (using extended dataset)
        self.n_years = n_years # number of years of data
        self.time_horizon_H = self.n_days_in_year * self.n_years    #simulation horizon
        self.hours_between_decisions = 24  # daily time step 
        self.decisions_per_day = int(24 / self.hours_between_decisions)
        self.n_days_one_year = 365
        
        # Constraints for the reservoirs
        self.min_level_Akosombo = 240 # ft _ min level for hydropower generation 
        self.flood_warn_level = 276 #ft _ flood warning level where spilling starts (absolute max is 278ft)
        
        #Volta Basing Characteristics
        self.QavgMonth = pd.read_csv('./Data/QavgMonthPerStation.csv') * 35.3146 #m³/s to cfs
        #Akosombo Characteristics
        self.lsv_rel = utils.loadMatrix(
            create_path("./Data/Akosombo_xtics/1.Level-Surface area-Volume/lsv_Ak.txt"), 3, 6 
            )  # level (ft) - Surface (acre) - storage (acre-feet) relationship
        self.turbines = utils.loadVector(
            create_path("./Data/Akosombo_xtics/2.Turbines/turbines_Ak2.txt"), 3
            )  # Max capacity (cfs) - min capacity (cfs) - efficiency of each Akosombo plant turbine (6 identical turbines)
        self.spillways = utils.loadMatrix(
            create_path("./Data/Akosombo_xtics/4.Spillways/spillways_Ak.txt"), 3, 4
            ) #level (ft) - max release (cfs) - min release (cfs) for level > 276 ft
                
        #Kpong Characteristics
        self.turbines_Kp = utils.loadVector(
            create_path("./Data/Kpong_xtics/2.Turbine/turbines_Kp.txt"), 3 
            ) #Max capacity (cfs) - min capacity (cfs) - efficiency of Kpong (4 identical turbines)
        
        
        self.data = data
        if self.data == 'historic':
            self.n_samples = 1
            self.n_years = 29
            self.load_historic_data()
            self.evaluate = self.evaluate_historic

        elif self.data == '50x20':
            self.n_samples = 50
            self.n_years = 20
            self.time_horizon_H = self.n_days_in_year * self.n_years
            self.load_stochastic_data()
            self.evaluate = self.evaluate_mc


        elif self.data == '1000x1':
            self.n_samples = 1000
            self.n_years = 1
            self.time_horizon_H = self.n_days_in_year * self.n_years
            self.load_stochastic_data()
            self.evaluate = self.evaluate_mc

        #Objective parameters
        # self.annual_power = utils.loadVector(
        #     create_path("./Data/Objective_parameters/annual_power.txt"), self.n_days_one_year
        # )   # annual hydropower target (GWh) (=4415GWh/year)
        self.annual_irri = utils.loadVector(
            create_path("./Data/Objective_parameters/irrigation_demand.txt"), self.n_days_one_year
        ) * self.irriDemandMultiplier  # annual irrigation demand (cfs) (=38m3/s rounded to whole number) * uncertainty parameter
        self.flood_protection = utils.loadVector(
            create_path("./Data/Objective_parameters/q_flood.txt"), self.n_days_one_year
        )  # q_flood: flow release above which flooding occurs (cfs) (81233cfs = 2300m3/s)
        self.clam_eflows_l = utils.loadVector(
            create_path("./Data/Objective_parameters/l_eflows.txt"), self.n_days_one_year
        )  # lower bound of of e-flow required in November to March (cfs) (=50 m3/s)
        self.clam_eflows_u = utils.loadVector(
            create_path("./Data/Objective_parameters/u_eflows.txt"), self.n_days_one_year
        )  # upper bound of of e-flow required in November to March (cfs) (=330 m3/s)
        self.eflows2_l = utils.loadVector(
            create_path("./Data/Objective_parameters/l_eflows2.txt"), self.n_days_one_year
        ) # e-flows scenario with 2300m3/s in Sept-Oct and 700m3/s rest of the year- low flow
        self.eflows2_u = utils.loadVector(
            create_path("./Data/Objective_parameters/u_eflows2.txt"), self.n_days_one_year
        ) # e-flows scenario with 2300m3/s in Sept-Oct and 700m3/s rest of the year- high flow
        self.eflows3_l = utils.loadVector(
            create_path("./Data/Objective_parameters/l_eflows3.txt"), self.n_days_one_year
        ) # e-flows scenario with 3000m3/s in Sept-Oct and 500m3/s rest of the year- low flow
        self.eflows3_u = utils.loadVector(
            create_path("./Data/Objective_parameters/u_eflows3.txt"), self.n_days_one_year
        ) # e-flows scenario with 3000m3/s in Sept-Oct and 500m3/s rest of the year- high flow
        
        
        # standardization of the input-output of the RBF release curve      
        self.input_max.append(self.n_days_in_year * self.decisions_per_day - 1) #max number of inputs
        self.input_max.append(278)  #max resevoir level in ft                                            
        
        self.output_max.append(utils.computeMax(self.annual_irri))
        self.output_max.append(787416)
        # max release = total turbine capacity + spillways @ max storage (56616 + 730800= 787416 cfs) 

    def load_historic_data(self):   #ET, inflow, Akosombo tailwater, Kpong fixed head
        self.evap_Ak = utils.loadMultiVector(
            create_path("./Data/Historical_data/vectors2/evapAk_history.txt"),
            self.n_years, self.n_days_one_year
            ) #evaporation losses @ Akosombo 1984-2012 (inches per day)
        self.inflow_Ak = utils.loadMultiVector(
            create_path("./Data/Historical_data/vectors2/InflowAk_history.txt"), #historical
            #create_path("./Data/CC_data/vectors2/InflowAk_cc5.txt"), #CC scenarios
            self.n_years, self.n_days_one_year
            )# inflow, i.e. flows to Akosombo (cfs) 1984-2012
        self.tailwater_Ak = utils.loadMultiVector(
            create_path("./Data/Historical_data/vectors2/tailwaterAk_history.txt"),
            self.n_years, self.n_days_one_year
            ) # historical tailwater level @ Akosombo (ft) 1984-2012
        self.fh_Kpong = utils.loadMultiVector(create_path(
            "./Data/Historical_data/vectors2/fhKp_history.txt"),
            self.n_years, self.n_days_one_year
            ) # historical fixed head @ Kpong (ft) 1984-2012

        self.evap_Ak = np.reshape(
            pd.read_csv("./Data/Stochastic/evap_Ak-1x29.csv", header=None).to_numpy(),
            (self.n_years, self.n_days_one_year))
        # evaporation losses @ Akosombo_ stochastic data (inches per day)

        self.tailwater_Ak = np.reshape(
            pd.read_csv("./Data/Stochastic/tailwater_Ak-1x29.csv", header=None).to_numpy(),
            (self.n_years, self.n_days_one_year))
        # tailwater level @ Akosombo (ft)

        self.fh_Kpong = np.reshape(pd.read_csv("./Data/Stochastic/fhKp-1x29.csv", header=None).to_numpy(),
                                   (self.n_years, self.n_days_one_year))
        
        
    def load_stochastic_data(self):   # historical data in matrix format
        self.evap_Ak = np.reshape(pd.read_csv("./Data/Stochastic/evap_Ak-" + self.data + ".csv", header=None).to_numpy(),
                             (self.n_samples, self.n_days_one_year * self.n_years))
        # evaporation losses @ Akosombo_ stochastic data (inches per day)

        self.inflow_Ak = np.reshape(pd.read_csv("./Data/Stochastic/InflowAk-" + self.data + ".csv", header=None).to_numpy(),
                               (self.n_samples, self.n_days_one_year * self.n_years)) * 35.3146
        # inflow, i.e. flows to Akosombo_stochastic data

        self.tailwater_Ak = np.reshape(
            pd.read_csv("./Data/Stochastic/tailwater_Ak-" + self.data + ".csv", header=None).to_numpy(),
            (self.n_samples, self.n_days_one_year * self.n_years))
        # tailwater level @ Akosombo (ft)

        self.fh_Kpong = np.reshape(pd.read_csv("./Data/Stochastic/fhKp-" + self.data + ".csv", header=None).to_numpy(),
                              (self.n_samples, self.n_days_one_year * self.n_years))
        # fixed head Kpong (ft)

        # self.evap_Ak = utils.loadArrangeMatrix(
        #     create_path("./Data/Stochastic/evap_Ak-"+ self.data +".csv"),
        #     self.n_samples, self.n_days_one_year * self.n_years
        #     ) #evaporation losses @ Akosombo_ stochastic data (inches per day)
        # self.inflow_Ak = utils.loadArrangeMatrix(
        #     create_path("./Data/Stochastic/InflowAk-"+ self.data +".csv"),
        #     self.n_samples, self.n_days_one_year * self.n_years
        # ) * 35.3146  # inflow, i.e. flows to Akosombo_stochastic data
        # self.tailwater_Ak = utils.loadArrangeMatrix(
        #     create_path("./Data/Stochastic/tailwater_Ak-"+ self.data +".csv"),
        #     self.n_samples, self.n_days_one_year * self.n_years
        #     ) # tailwater level @ Akosombo (ft)
        # self.fh_Kpong = utils.loadArrangeMatrix(
        #     create_path("./Data/Stochastic/fhKp-"+ self.data +".csv"),
        #     self.n_samples, self.n_days_one_year * self.n_years
        #     ) # fixed head Kpong (ft)
        
            
    def set_log(self, log_objectives):
        if log_objectives:
            self.log_objectives = True
        else:
            self.log_objectives = False

    def get_log(self):
        return self.blevel_Ak, self.rirri, self.renv

        
    def apply_rbf_policy(self, rbf_input):
        # normalize inputs 
        formatted_input = rbf_input / self.input_max
        #print(formatted_input)
        
        #apply rbf
        normalized_output = self.rbf.apply_rbfs(formatted_input)
        #print(normalized_output)
        
        # scale back normalized output
        scaled_output = normalized_output * self.output_max #uu
        #print(scaled_output)
        
        #uu =[]
        #for i in range (0, self.):
            #uu.append(uu[i] * self.output_max[i])
        return scaled_output
    
    def evaluate_historic(self, var, opt_met=1):
        return self.simulate(var, self.inflow_Ak, self.evap_Ak, 
                             self.tailwater_Ak, self.fh_Kpong, opt_met) #for running 1 year of data

    def evaluate_mc(self, var, opt_met=1):
        obj, j_hyd_a, j_hyd_k, j_energy_reliability, j_irri, j_env, j_fldcntrl = [], [], [], [], [], [], []
        self.levels = []
        self.releases_irri = []
        self.releases_down = []
        # MC simulations
        n_samples = self.n_samples
        for i in range(0, n_samples):
            Jhydropower_A, Jhydropower_K, J_energy_reliability, Jirrigation, Jenvironment, Jfloodcontrol   = self.simulate(
                var,
                np.reshape(self.inflow_Ak[i],(self.n_years, self.n_days_one_year)),
                np.reshape(self.evap_Ak[i],(self.n_years, self.n_days_one_year)),
                np.reshape(self.tailwater_Ak[i],(self.n_years, self.n_days_one_year)),
                np.reshape(self.fh_Kpong[i],(self.n_years, self.n_days_one_year)),
                opt_met,
            )

            j_hyd_a.append(Jhydropower_A)
            j_irri.append(Jirrigation)
            j_env.append(Jenvironment)
            j_fldcntrl.append(Jfloodcontrol)
            j_hyd_k.append(Jhydropower_K)
            j_energy_reliability.append(J_energy_reliability)

            if self.log_objectives:
                self.levels.append(self.blevel_Ak)
                self.releases_down.append(self.renv)
                self.releases_irri.append(self.rirri)
                # self.rirri.append(release_i)
                # self.renv.append(release_d)
                # print('samples done: ', i)

        # objectives aggregation
        # (worst case/robust formulation: for obj that are maximised, insert minimum of the 29 years and vice versa)
        # obj.insert(0, np.min(Jhyd_a))
        # obj.insert(1, np.min(Jirri))
        # obj.insert(2, np.min(Jenv))
        # obj.insert(3, np.max(Jflood))
        # obj.insert(4, np.min(Jhyd_k))

        # (99% reliability formulation: insert x-percentile of the 29 years)
        # obj.insert(0, np.percentile(Jhyd_a, 99))
        # obj.insert(1, np.percentile(Jirri, 99))
        # obj.insert(2, np.percentile(Jenv, 99))
        # obj.insert(3, np.percentile(Jflood, 99))
        # obj.insert(4, np.percentile(Jhyd_k, 99))

        return j_hyd_a, j_hyd_k, j_energy_reliability, j_irri, j_env, j_fldcntrl

    #convert storage at current timestep to level and then level to surface area
    def storage_to_level(self, s):
        # s : storage 
        s_ = utils.cubicFeetToAcreFeet(s)
        h = utils.interpolate_linear(self.lsv_rel[2], self.lsv_rel[0], s_)
        return h
    
    def level_to_storage(self, h):
        s = utils.interpolate_linear(self.lsv_rel[0], self.lsv_rel[2], h)        
        return utils.acreFeetToCubicFeet(s)

    def level_to_surface(self, h):
        s = utils.interpolate_linear(self.lsv_rel[0], self.lsv_rel[1], h)
        return utils.acreToSquaredFeet(s)
    

    def actual_release(self, uu, level_Ak, day_of_year):
        #uu = prescribed flow release policy
        Tcap = 56616 # total turbine capacity at Akosombo(cfs)
        #maxSpill = 730800 # total spillway capacity (cfs)
        
        # minimum discharge values for irrigation and downstream
        qm_I = 0.0
        qm_D = 5050.0 #  turbine flow corresponding to 6GWh for system stability 5050 cfs
        
        # maximum discharge values (can be as much as the demand)
        qM_I = self.annual_irri[day_of_year]
        qM_D = Tcap
        
        # reservoir release constraints
        if level_Ak <= self.min_level_Akosombo:
            qM_I = 0.0
            qM_D = 0.0
        else:
            qM_I = self.annual_irri[day_of_year]
            qM_D = Tcap
            
        if level_Ak > self.flood_warn_level:
            qM_D = (utils.interpolate_linear(self.spillways[0], 
                                             self.spillways[1],
                                             level_Ak)) + Tcap
            
            qm_D = (utils.interpolate_linear(self.spillways[0],
                                             self.spillways[1], 
                                             level_Ak)) +Tcap
    
            
        #actual release
        rr = []
        rr.append(min(qM_I, max(qm_I, uu[0])))
        rr.append(min(qM_D, max(qm_D, uu[1])))
        #print(rr)
        return rr

        
    @staticmethod
    @njit
    def g_hydAk(r, h, day_of_year, hour0, GG, gammaH20, turbines, data):
         #hydropower @ Akosombo =f(prescribed release, water level
         # day of year, hour, gravitational acc,
         #water density, tailwater level, flow through turbines)
         
         cubicFeetToCubicMeters = 0.0283
         feetToMeters = 0.3048
         Nturb = 6 #number of turbines at Akosombo
         g_hyd = [] #power generated in GWh each day
         pp = [] #power generated in GWh at each turbine per day
         c_hour = len(r) * hour0
         #TODO: why does this not get called?
         for i in range(0, len(r)):
             #print(i)
             if data == 'historic':
                 if i < 8035:
                     eff = 0.9
                 else:
                     eff = 0.93 #efficiency of turbines increased from 0.9 to 0.93 in 2006
             else:
                 eff = 0.9

             deltaH = h[i] 
             q_split = r[i]     #cycling through turbines
             for j in range(0, Nturb):
                 if q_split < turbines[1]: #turbine flow for system stability. if flow falls below this, the turbine is essentially shut off
                     qturb = 0.0
                 elif q_split > turbines[0]:
                     qturb = turbines[0]
                 else:
                     qturb = q_split
                 q_split = q_split - qturb
                 
                 p = (
                     eff
                     * GG
                     * gammaH20
                     * (cubicFeetToCubicMeters * qturb)
                     * (feetToMeters * deltaH)
                     * 3600
                     / (3600 * 1000 * pow(10,6))  
                     )
                 pp.append(p)
             g_hyd.append(np.sum(np.asarray(pp)))
             pp.clear()
             c_hour = c_hour +1
         Gp = np.sum(np.asarray(g_hyd))
         return Gp
             
    @staticmethod
    @njit
    def g_hydKp(r, h, day_of_year, hour0, GG, gammaH20, turbines_Kp):
        #hydropower @ Kpong =f(prescribed release, fixed head,
        # day of year, hour, gravitational acc,
        #water density, flow through turbines)
         
        cubicFeetToCubicMeters = 0.0283
        feetToMeters = 0.3048
        n_turb = 4
        g_hyd_Kp = []
        pp_K = []
        c_hour = len(r) * hour0
        for i in range(0, len(r)):
            #print (i)
            if i < 8035:
                eff = 0.9
            else:
                eff = 0.93
             
            deltaH = h[i]
            q_split = r[i]
            for j in range(0, n_turb):
                if q_split < 0:
                    qturb = 0.0
                elif q_split > turbines_Kp[0]:
                    qturb = turbines_Kp[0]
                else:
                    qturb = q_split
                q_split = q_split - qturb
                p = (
                    eff
                    * GG
                    * gammaH20
                    * (cubicFeetToCubicMeters * qturb)
                    * (feetToMeters * deltaH)
                    * 3600
                    / (3600 * 1000 * pow(10,6))
                    )
                pp_K.append(p)
            g_hyd_Kp.append(np.sum(np.asarray(pp_K)))
            pp_K.clear()
            c_hour = c_hour + 1
        Gp_Kp = np.sum(np.asarray(g_hyd_Kp))
        return Gp_Kp


    def res_transition_h(self, s0, uu, n_sim, ev, day_of_year, hour0, fh_Kp, tw_Ak):
        HH = self.hours_between_decisions
        sim_step = 3600 #seconds per hour
        leak = 0 #loss to seepage
        
        #storages and levels
        shape = (HH+1, ) #+1 to account for initial time step
        storage_Ak = np.empty(shape)
        level_Ak = np.empty(shape)

        #Actual releases
        shape = (HH, )
        release_I =np.empty(shape) #irrigation
        release_D = np.empty(shape) # downstream relaease including e-flows
        
        #initial conditions
        storage_Ak[0] = s0
        c_hour = HH * hour0
        
        for i in range(0, HH):
            #print (i)
            level_Ak[i] = self.storage_to_level(storage_Ak[i])
            
            #Compute actual release
            rr = self.actual_release(uu, level_Ak[i], day_of_year)
            release_I[i] = rr[0]
            release_D[i] = rr[1]
            #print(release_D)
            #print(release_I)
            
            #compute surface level, ET loss, head @Ak(reservoir level -tailwater level), fixedhead @Kp
            surface_Ak = self.level_to_surface(level_Ak[i])
            evaporation_losses_Ak = utils.inchesToFeet(ev) * surface_Ak / 86400 #cfs_daily ET
            h_Ak = level_Ak[i] - tw_Ak # head @Akosombo
            h_Kp = fh_Kp #head @ Kpong
            
            
            #system transition
            storage_Ak[i +1] = storage_Ak[i] + sim_step *(
                n_sim - evaporation_losses_Ak  - 
                release_D[i] -release_I[i]- leak
                )
            
            c_hour = c_hour + 1
            #print (c_hour)
            
        sto_ak = storage_Ak[HH]
        #print(sto_ak)
        rel_i = utils.computeMean(release_I)
        #print(rel_i)
        rel_d = utils.computeMean(release_D)
        #print(rel_d)
        
        level_Ak = np.asarray(level_Ak) 
        #print("water level = ", utils.computeMean(level_Ak)) #daily water level
        h_Ak = np.asarray(np.tile(h_Ak, int(len(level_Ak))))
        #print(h_Ak)
        h_Kp = np.asarray(np.tile(h_Kp, int(len(level_Ak))))
        #print(h_Kp)
        #r = np.asarray(release_D) #r
        #print(r)
        
        #decision timestep
        hp = VoltaModel.g_hydAk(
            np.asarray(release_D + release_I),
            h_Ak,
            day_of_year,
            hour0,
            self.GG,
            self.gammaH20,
            self.turbines,
            self.data
            )
        
        hp_kp = VoltaModel.g_hydKp(
            np.asarray(release_D),
            h_Kp,
            day_of_year,
            hour0,
            self.GG,
            self.gammaH20,
            self.turbines_Kp
            )

        self.energy_stored = max(min(self.energy_stored + hp_kp + hp - self.daily_energy_usage, self.energy_storage),0)
        energy_deficiency = min(min(self.energy_stored + hp_kp + hp - self.daily_energy_usage, self.energy_storage), 0)
        
        return sto_ak, rel_i, rel_d, hp, hp_kp, self.energy_stored, energy_deficiency
    
    
    #OBJECTIVE FUNCTIONS
        
    #Flood protection (Ak release < 2300m3/s- q_flood objective) -Minimization
    def q_flood_protectn_rel(self, q1, qTarget):
        delta = 24 * 3600
        qTarget = np.tile(qTarget, int(len(q1) / self.n_days_one_year))
        maxarr = (q1 * delta) - (qTarget * delta)
        maxarr[maxarr < 0] = 0
        gg = maxarr / (qTarget * delta)
        #g = np.mean(np.square(gg)) # to penalize larger floods more than smaller floods
        # g = np.mean(gg)
        p2 = np.reshape(gg, (self.n_years, self.n_days_one_year))
        p2 = p2.mean(axis=1)
        return p2 #target value = 0
        
    
    #Clam E-flows - Maximization 
    def g_eflows_index(self, q, lTarget, uTarget):
        delta = 24*3600
        e = 0
        for i, q_i in np.ndenumerate(q):
            tt = i[0] % self.n_days_one_year
            if ((lTarget[tt] >0) or (uTarget[tt] > 0)):
                if ((lTarget[tt] * delta) > (q_i * delta)) or ((q_i * delta) > (uTarget[tt] * delta)):
                    e = e + 1
        #TODO how should this be defined? there is always an upper limit but only looks at lower limit
        G = 1 - (e / (self.n_years * np.sum(lTarget > 0)))
        return G  #target value = 1
    
    #E-flows 2 and 3-  Maximization
    def g_eflows_index2(self, q, lTarget, uTarget):
        delta = 24*3600
        e = 0
        for i, q_i in np.ndenumerate(q):
            tt = i[0] % self.n_days_one_year
            if ((lTarget[tt] * delta) < (q_i * delta)) or ((q_i * delta) < (uTarget[tt] * delta)):
                e = e + 1          
        
        G = 1 - (e / (self.n_years * np.sum(lTarget > 0)))
        return G  #target value = 1
    
    #Irrigation - Maximization
    def g_vol_rel(self, q, qTarget):
        delta = 24 * 3600
        qTarget = np.tile(qTarget, int(len(q) / self.n_days_one_year))
        g = (q * delta)  / (qTarget * delta )
        # G = utils.computeMean(g)
        p2 = np.reshape(g, (self.n_years, self.n_days_one_year))
        p2 = p2.mean(axis=1)
        #print(q)
        return p2 #target value = 1
    
    
    #Annual hydropower - Minimization of deviation #not used
    #reshape (n_years, 365), aggregation by year (across row), store and compare to target
    def g_energy_reliability(self, energy_deficiency):
        energy_deficiency = energy_deficiency
        # G = np.mean(np.square(energy_deficiency))
        p1 = np.reshape(np.square(energy_deficiency), (self.n_years, self.n_days_one_year))
        p2 = np.sum(p1, axis=1)

        # pTarget = np.tile(pTarget, int(len(p) / self.n_days_one_year))
        # pTarget1 = np.reshape(pTarget,(self.n_years, self.n_days_one_year))
        # p1 = np.reshape(p, (self.n_years, self.n_days_one_year))
        # pTarget2 = np.sum(pTarget1, axis=1)
        # p2 = np.sum(p1, axis=1)
        # maxhyd = (pTarget2 - p2)
        # maxhyd[maxhyd < 0] = 0 #no penalty when hydropower is more than target
        # gg = maxhyd
        # G = np.mean(np.square(gg))
        return p2   #target value = 0
    
    #Hydropower- Maximization of hydropower generated
    def g_hydro_max(self, p):
        p1 = np.reshape(p, (self.n_years, self.n_days_one_year))
        p2 = np.sum(p1, axis=1)
        # G = np.mean(p2)
        return p2 #target value = max  hydropower possible (approx 1.038GW x24 x 365 =9093GWh)
        
    
    #Simulation
    #self.simulate(var, self.evap_Ak, self.inflow_Ak,
                         #self.tailwater_Ak, self.fh_Kpong, opt_met)
    def simulate(self, input_variable_list_var, inflow_Ak_n_sim, evap_Ak_e_ak, 
                 tailwater_Ak, fh_Kpong,  opt_met
                 ):
        #check tailwater and fixed head inclusion
        
        # Initializing daily variables
        # storages and levels

        shape = (self.time_horizon_H + 1, )
        storage_ak = np.empty(shape)
        level_ak = np.empty(shape)
        # Akosombo actual releases
        shape = (self.time_horizon_H,)
        #print(shape)
        release_i = np.empty(shape) #irrigation
        release_d = np.empty(shape) #downstream release (hydro, e-flows, floods)
        
        # hydropower production
        hydropowerProduction_Ak = []  # energy production at Akosombo
        hydropowerProduction_Kp = []  # energy production at Kpong
        energy_stored = []
        energy_deficiency = []
        # release decision variables (irrigation) only
        # Downstream in Baseline
        self.rbf.set_decision_vars(np.asarray(input_variable_list_var))
        #print(np.asarray(input_variable_list_var).shape)

        # initial condition
        level_ak[0] = self.init_level
        storage_ak[0] = self.level_to_storage(level_ak[0])
        QavgMonth = self.QavgMonth
        # identification of the periodicity (365 x fdays)
        decision_steps_per_year = self.n_days_in_year * self.decisions_per_day
        year = 0
        #print(decision_steps_per_year)
        
        
        #run simulation
        for t in range (self.time_horizon_H):
            #print(t)
            day_of_year = t % self.n_days_in_year
            #print("day # ", day_of_year)
            if day_of_year%self.n_days_in_year == 0 and t !=0:
                year = year + 1
                #print("year# ", year)
                
                
            shape = (self.decisions_per_day +1,)
            daily_storage_ak = np.empty(shape)
            daily_level_ak = np.empty(shape)
            
            shape = (self.decisions_per_day,)
            daily_release_i = np.empty(shape)
            daily_release_d = np.empty(shape)
            
            #initialization of sub-daily cycle 
            daily_level_ak[0] = level_ak[day_of_year] 
            #print("daily level = ", daily_level_ak)
            daily_storage_ak[0] = storage_ak[day_of_year]
            #print("daily storage = ",daily_storage_ak)
            
            # sub-daily cycle #incase subdaily optimisation is done later
            for j in range (self.decisions_per_day):
                decision_step = (day_of_year * self.decisions_per_day) + j
                #print (j)
                
                # decision step i in a year
                jj = decision_step % decision_steps_per_year
                #print (jj)
                
                #compute decision
                if opt_met == 0:# fixed release
                    uu.append(uu[0]) #not used
                elif opt_met == 1:# RBF-PSO
                    rbf_input = np.asarray([jj, daily_level_ak[j]])
                    uu = self.apply_rbf_policy(rbf_input)
                    #print(rbf_input)
                    #print(uu)

                #Apply climate multiplier
                if day_of_year <= 31:
                    InflowDay = inflow_Ak_n_sim[year][day_of_year] * (1+ self.Cjanuary)
                    month = 1
                elif 31 < day_of_year <= 59:
                    InflowDay = inflow_Ak_n_sim[year][day_of_year] * (1+ self.Cfebruary)
                    month = 2
                elif 59 < day_of_year <= 90:
                    InflowDay = inflow_Ak_n_sim[year][day_of_year] * (1+ self.Cmarch)
                    month = 3
                elif 90 < day_of_year <= 120:
                    InflowDay = inflow_Ak_n_sim[year][day_of_year] * (1+ self.Capril)
                    month = 4
                elif 120 < day_of_year <= 151:
                    InflowDay = inflow_Ak_n_sim[year][day_of_year] * (1+ self.Cmay)
                    month = 5
                elif 151 < day_of_year <= 181:
                    InflowDay = inflow_Ak_n_sim[year][day_of_year] * (1+ self.Cjune)
                    month = 6
                elif 181 < day_of_year <= 212:
                    InflowDay = inflow_Ak_n_sim[year][day_of_year] * (1+ self.Cjuly)
                    month = 7
                elif 212 < day_of_year <= 243:
                    InflowDay = inflow_Ak_n_sim[year][day_of_year] * (1+ self.Caugust)
                    month = 8
                elif 243 < day_of_year <= 273:
                    InflowDay = inflow_Ak_n_sim[year][day_of_year] * (1+ self.Cseptember)
                    month = 9
                elif 273 < day_of_year <= 305:
                    InflowDay = inflow_Ak_n_sim[year][day_of_year] * (1+ self.Coctober)
                    month = 10
                elif 305 < day_of_year <= 334:
                    InflowDay = inflow_Ak_n_sim[year][day_of_year] * (1+ self.Cnovember)
                    month = 11
                elif 334 < day_of_year <= 365:
                    InflowDay = inflow_Ak_n_sim[year][day_of_year] * (1+ self.Cdecember)
                    month = 12

                QavgMonthTagou = QavgMonth['Tagou'][month-1]
                QavgMonthArly = QavgMonth['Arly'][month-1]
                QavgMonthPorga = QavgMonth['Porga'][month-1]
                QavgMonthSabari = QavgMonth['Sabari'][month-1]
                QavgMonthYarugu = QavgMonth['Yarugu'][month-1]
                QavgMonthNangodi = QavgMonth['Nangodi'][month-1]
                QavgMonthOuessa = QavgMonth['Ouessa'][month-1]
                QavgMonthBamboi = QavgMonth['Bamboi'][month-1]

                ratioTagou = 0.0076
                ratioArly = 0.0081
                ratioPorga = 0.0309
                ratioSabari = 0.2538
                ratioYarugu = 0.0604
                ratioNangodi = 0.0181
                ratioOuessa = 0.0361
                ratioBamboi = 0.1868
                ratioSenchi = 0.3982

                collectedTagou = InflowDay * ratioTagou
                collectedArly = InflowDay * ratioArly
                collectedPorga = InflowDay * ratioPorga
                collectedSabari = InflowDay * ratioSabari
                collectedYarugu = InflowDay * ratioYarugu
                collectedNangodi = InflowDay * ratioNangodi
                collectedOuessa = InflowDay * ratioOuessa
                collectedBamboi = InflowDay * ratioBamboi
                collectedSenchi = InflowDay * ratioSenchi

                #oti river
                inflowFromTagou = max(collectedTagou - QavgMonthTagou * self.waterUseBurkinaFaso,
                                      collectedTagou * self.treatiesBurkinaFaso)
                inflowFromArly = max(collectedArly - QavgMonthArly * self.waterUseBurkinaFaso,
                                     collectedArly * self.treatiesBurkinaFaso)
                inflowFromPorga = max(inflowFromArly + collectedPorga - QavgMonthPorga * self.waterUseBenin,
                                      (inflowFromArly + collectedPorga) * self.treatiesBenin)
                inflowFromSabari = max(inflowFromTagou + inflowFromPorga + collectedSabari - QavgMonthSabari * self.waterUseTogo,
                                       (inflowFromTagou + inflowFromPorga + collectedSabari) * self.treatiesTogo)

                #White volta
                inflowFromYarugu = max(collectedYarugu - QavgMonthYarugu * self.waterUseBurkinaFaso,
                                       collectedYarugu * self.treatiesBurkinaFaso)
                inflowFromNangodi= max(collectedNangodi - QavgMonthNangodi * self.waterUseBurkinaFaso,
                                       collectedNangodi * self.treatiesBurkinaFaso)

                #black volta
                inflowFromOuessa = max(collectedOuessa - QavgMonthOuessa * self.waterUseBurkinaFaso,
                                       collectedOuessa * self.treatiesBurkinaFaso)
                inflowFromBamboi = max(inflowFromOuessa + collectedBamboi - QavgMonthBamboi * self.waterUseCoteIvoire,
                                       (inflowFromOuessa + collectedBamboi)* self.treatiesCoteIvoire)

                inflowFromSenchi = collectedSenchi

                inflowToLakeVolta = inflowFromBamboi + inflowFromNangodi + inflowFromYarugu + inflowFromSabari + inflowFromSenchi

                #def res_transition_h(self, s0, uu, n_sim, ev, day_of_year, hour0)
                #system transition
                ss_rr_hp = self.res_transition_h(
                    daily_storage_ak[j],
                    uu,
                    inflowToLakeVolta,
                    evap_Ak_e_ak[year][day_of_year],
                    day_of_year,
                    j,
                    fh_Kpong[year][day_of_year],
                    tailwater_Ak[year][day_of_year]
                    )
                
                daily_storage_ak[j+1] = ss_rr_hp[0]
                #print(daily_storage_ak)
                daily_level_ak[j+1] = self.storage_to_level(daily_storage_ak[j+1])
                #print(daily_level_ak)
                daily_release_i[j] = ss_rr_hp[1]
                #print("rel i= ", daily_release_i)
                daily_release_d[j] = ss_rr_hp[2]
                #print("rel d= ", daily_release_d)
                
                #Hydropower production
                hydropowerProduction_Ak.append(
                    ss_rr_hp[3])# daily energy production (GWh/day) at Akosombo
                hydropowerProduction_Kp.append(
                    ss_rr_hp[4])# daily energy production (GWh/day) at Kpong

                energy_stored.append(ss_rr_hp[5])
                energy_deficiency.append(ss_rr_hp[6])

            #daily values
            level_ak[t + 1] = daily_level_ak[self.decisions_per_day] #ft
            storage_ak[t + 1] = daily_storage_ak[self.decisions_per_day] 
            release_i[t] = np.mean(daily_release_i) #cfs
            release_d[t] = np.mean(daily_release_d) #cfs
            #print("release_i= ", release_i)
            #print("release_d= ", release_d)
            
            
        #log level / release
        if self.log_objectives:
            self.blevel_Ak.append(level_ak)
            self.rirri.append(release_i)
            self.renv.append(release_d) 
            
        # compute objectives
        j_hyd_a = self.g_hydro_max(hydropowerProduction_Ak) # GWh/year  # Maximization of annual hydropower- Akosombo
        #j_hyd_a = self.g_hydro_rel(hydropowerProduction_Ak, self.annual_power) # Minimization of deviation from target of 4415GWh
        j_irri = self.g_vol_rel(release_i, self.annual_irri) #Maximisation
        #j_env = self.g_eflows_index(release_d, self.clam_eflows_l, self.clam_eflows_u) #Maximisation
        j_env = self.g_eflows_index2(release_d, self.eflows2_l, self.eflows2_u) #Maximisation
        #j_env = self.g_eflows_index2(release_d, self.eflows3_l, self.eflows3_u) #Maximisation
        j_fldcntrl = self.q_flood_protectn_rel(release_d, self.flood_protection) #Minimization
        j_hyd_k = self.g_hydro_max(hydropowerProduction_Kp) # GWh/year  # Maximization of annual hydropower- Kpong

        j_energy_reliability = self.g_energy_reliability(energy_deficiency)
        #print(j_irri)
        return j_hyd_a,j_hyd_k, j_energy_reliability, j_irri, j_env, j_fldcntrl
