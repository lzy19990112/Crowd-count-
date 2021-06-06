import scipy.io as sio
import numpy as np
import numpy as np
import os
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import make_scorer, mean_squared_error

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, Matern, WhiteKernel

path = 'C:\\Users\\lizhaoyang\\Downloads\\Crowd-Count\\ShanghaiTech\\part_B'


def GPR_model():
    ###Read descriptors and ground truth for train set
    matFile=sio.loadmat(os.path.join(path,"train"))
    desc_train=matFile["lbp_descriptors"]
    labels_train=matFile["labels"]

    kernel = RBF(length_scale=1)+DotProduct(sigma_0=1)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1, normalize_y=True)
    gpr.fit(desc_train, labels_train)
    return gpr


def KRR_model():
    ###Read descriptors and ground truth for train set
    matFile=sio.loadmat(os.path.join(path,"train"))
    desc_train=matFile["lbp_descriptors"]
    labels_train=matFile["labels"]



    Grid_Dict = {"alpha": [1e-13,1e-5,1e-4,1e-3,1e-2],"gamma": np.logspace(-3, 2, 10)}
    krr_Tuned = GridSearchCV(KernelRidge(kernel='rbf'), cv=5 ,param_grid=Grid_Dict, scoring=make_scorer(mean_squared_error,greater_is_better=False),refit=True)
    krr_Tuned.fit(desc_train, labels_train)

    return krr_Tuned

def predict(krr_model, gpr_model):

    matFile=sio.loadmat(os.path.join(path,"test"))
    desc_test=matFile["lbp_descriptors"]
    labels_test=matFile["labels"]

    pred_test1 = krr_model.predict(desc_test)
    pred_test2 = gpr_model.predict(desc_test)
    MSE1 = mean_squared_error(labels_test, pred_test1)
    MSE2 = mean_squared_error(labels_test, pred_test2)
    return MSE1, MSE2

if __name__ == "__main__":
    gpr = GPR_model()
    krr = KRR_model()
    MSE1, MSE2 = predict(krr, gpr)
    print("  minTestError1:"+str(MSE1), " minTestError2:"+str(MSE2), "Best alpha:"+ str(krr.best_params_['alpha']), "Best rbf param:"+ str(krr.best_params_['gamma']))


