import numpy as np


class EnvManager:
    def __init__(self, meta_file, num_envs, seed=None):
        self._mNumEnvs = num_envs
        self._mEnvs = np.full(num_envs, None)

        if seed is not None:  # dart::math::seedRand();
            np.random.seed(seed)

        # TODO: omp_set_num_threads(mNumEnvs);
        for id in range(num_envs):
            self._mEnvs(id) = None # TODO: mEnvs.push_back(new MASS::Environment());
            self._mEnvs(id).Initialize(meta_file, False) # TODO: env->Initialize(meta_file,false);

        self._muscle_torque_cols = self._mEnvs(0).GetMuscleTorques().rows() # muscle_torque_cols = mEnvs[0]->GetMuscleTorques().rows();
        self._tau_des_cols = self._mEnvs(0).GetDesiredTorques().rows() # tau_des_cols = mEnvs[0]->GetDesiredTorques().rows();

        self._mEos = np.full(num_envs, None)
        self._mRewards = np.full(num_envs, None)
        self._mStates = np.full(num_envs, None)
        self._mMuscleTorques = np.full((num_envs, self._muscle_torque_cols), None)
        self._mDesiredTorques = np.full((num_envs, self._tau_des_cols), None)

        self._mMuscleTuplesJtA = None
        self._mMuscleTuplesTauDes = None
        self._mMuscleTuplesL = None
        self._mMuscleTuplesb = None
    
        # others, refer to EnvManager(std::string meta_file,int num_envs)

    def GetNumState(self) -> int:
        return self._mEnvs(0).GetNumState()
    

    def GetNumAction(self) -> int:
        return self._mEnvs(0).GetNumAction()
    
    def GetSimulationHz(self) -> int:
        return self._mEnvs(0).GetSimulationHz()
    
    def GetControlHz(self) -> int:
        return self._mEnvs(0).GetControlHz()
    
    def GetNumSteps(self) -> int:
        return self._mEnvs(0).GetNumSteps()
    
    def UseMuscle(self) -> int:
        return self._mEnvs(0).UseMuscle()
    
    def Step(self, id):
        self._mEnvs(id).Step()

    def Reset(self, RSI, id):
        self._mEnvs(id).Reset(RSI)

    def IsEndOfEpisode(self, id) -> bool:
        return self._mEnvs(id).IsEndOfEpisode()

    def GetReward(self, id) -> float:
        return self._mEnvs(id).GetReward()
    
    def Steps(self, num):
        #TODO: pragma omp parallel for
        for id in range(self._mNumEnvs):
            for _ in range(num):
                self.Step(id)
    
    def StepsAtOnce(self):
        num = self.GetNumSteps()
        for id in range(self._mNumEnvs):
            for _ in range(num):
                self.Step(id)

    def Resets(self, RSI):
        for id in range(self._mNumEnvs):
            self._mEnvs(id).Reset(RSI)
    
    def IsEndOfEpisodes(self):
        for id in range(self._mNumEnvs):
            self._mEos(id) = self._mEnvs(id).IsEndOfEpisode()
        return self._mEos
    
    def GetStates(self):
        for id in range(self._mNumEnvs):
            self._mStates(id) = self._mEnvs(id).GetState().transpose()
        return self._mStates
    
    def SetActions(self, actions):
        for id in range(self._mNumEnvs):
            self._mEnvs(id).SetAction(actions.row(id).transpose)

    def GetRewards(self):
        for id in range(self._mNumEnvs):
            self._mRewards(id) = self._mEnvs(id).GetReward()
        return self._mRewards
    
    def GetMuscleTorques(self):
        for id in range(self._mNumEnvs):
            self._mMuscleTorques(id) = self._mEnvs(id).GetMuscleTorques()
        return self._mMuscleTorques
    
    def GetDesiredTorques(self):
        for id in range(self._mNumEnvs):
            self._mDesiredTorques(id) = self._mEnvs(id).GetDesiredTorques()
        return self._mDesiredTorques

    def SetActivationLevels(self, activations):
        for id in range(self._mNumEnvs):
            self._mEnvs(id).SetActivationLevels(activations.row(id))

    
    def ComputeMuscleTuples(self):
        n = 0
        rows_JtA = 0
        rows_tau_des = 0
        rows_L = 0
        rows_b = 0

        for id in range(self._mNumEnvs):
            tps = self._mEnvs(id).GetMuscleTuples()
            n += tps.shape(0)
            if tps.size() is not 0:
                rows_JtA += tps(0).JtA.rows()
                rows_tau_des += tps(0).tau_des.rows()
                rows_L += tps(0).L.rows()
                rows_b += tps(0).b.rows()
        
        self._mMuscleTuplesJtA = np.full((n, rows_JtA), None)
        self._mMuscleTuplesTauDes = np.full((n, rows_tau_des), None)
        self._mMuscleTuplesL = np.full((n, rows_L), None)
        self._mMuscleTuplesb = np.full((n, rows_b), None)

        o = 0
        for id in range(self._mNumEnvs):
            tps = self._mEnvs(id).GetMuscleTuples()
            for i in range(tps.shape(0)):
                self._mMuscleTuplesJtA(o) = tps(i).JtA
                self._mMuscleTuplesTauDes(o) = tps(i).tau_des
                self._mMuscleTuplesL(o) = tps(i).L
                self._mMuscleTuplesb(o) = tps(i).b
                o += 1
            tps.clear()
        
    def GetMuscleTuplesJtA(self):
        return self._mMuscleTuplesJtA
    
    def GetMuscleTuplesTauDes(self):
        return self._mMuscleTuplesTauDes
    
    def GetMuscleTuplesL(self):
        return self._mMuscleTuplesL
    
    def GetMuscleTuplesb(self):
        return self._mMuscleTuplesb
    
    


    



    
    
    


