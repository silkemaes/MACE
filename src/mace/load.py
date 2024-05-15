import src.mace.utils       as utils
import src.mace.loss        as loss
import src.mace.test        as test
import matplotlib.pyplot    as plt
import src.mace.CSE_0D.dataset       as ds
from src.mace.CSE_0D.plotting        import plot_abs
import numpy as np


class Trained_MACE():
    '''
    Class to load a trained MACE model.
    '''
    def __init__(self, outloc, dirname, epoch = -1):
        '''
        Load all the components of a MACE model.

        Input:
            - outloc: output location
            - dirname: name of the directory
            - epoch: specific epoch to load
                if epoch == -1, the last epoch is loaded (i.e. final state of the model)
                if epoch >= 0, the epoch*10's epoch is loaded
        
        Returns:
            - meta: file with meta data
            - model: torch model
            - trainloss: training loss per epoch
            - testloss: test loss per epoch
        '''

        self.loc   = outloc+dirname+'/'
        self.epoch = epoch

        self.meta = utils.load_meta(self.loc)

        self.model, self.num_params = utils.load_model(self.loc, self.meta, epoch)

        self.trainloss = utils.Loss_analyse(self.loc, self.meta, 'train')

        self.testloss  = utils.Loss_analyse(self.loc, self.meta, 'test')

        self.plotpath = self.loc + 'plots/'

        
    def get_meta(self):
        return self.meta
    
    def get_model(self):
        return self.model
    
    def get_trainloss(self):
        return self.trainloss
    
    def get_testloss(self):
        return self.testloss
    
    def get_num_params(self):
        return self.num_params
    
    def get_loc(self):
        return self.loc
    
    def get_epoch(self):
        return self.epoch
    
    def plot_loss(self, log=True, ylim=False, limits=None, save = True):

        len = self.get_meta()['epochs']

        fig = loss.plot(self.trainloss, self.testloss, len, log = log, ylim = ylim, limits = limits)

        if save == True:
            plt.savefig(self.plotpath+'loss.png')

        plt.show()

        return
    
    def test(self, testpath, specs, plotting = False, save = False):
        '''
        Test the model on a test set.

        Input:
            - testpath: path to the test data
            - plotting: plot the results, default = False
        '''

        model1D, input, info = ds.get_test_data(testpath, self.meta)
        id = info['path'] +'_'+ info['name']

        n, n_hat, t, comptime = test.test_step(self.model, input)
        n_evol, mace_time  = test.test_evolution(self.model, input, start_idx=0)

        print('\n>>> Denormalising... ')
        n = ds.get_abs(n)
        n_hat = ds.get_abs(n_hat)
        n_evol = ds.get_abs(n_evol)

        err, err_test = utils.error(n, n_hat)
        err, err_evol = utils.error(n, n_evol)

        print('\nErrors (following Eq. 23 of Maes et al., 2024):')
        print('      Step error:', np.round(err_test,3))
        print(' Evolution error:', np.round(err_evol,3))

        if plotting == True:
            print('\n>>> Plotting...')

            if len(specs) == 0:
                print('No species specified, using a default list:')
                print('     CO, H2O, OH, C2H2, C2H, CH3C5NH+, C10H2+')
                specs = ['CO', 'H2O','OH',  'C2H2',  'C2H', 'CH3C5NH+', 'C10H2+']

            ## plotting results for the step test
            fig_step = plot_abs(model1D, n, n_hat, specs=specs, step = True)
            if save == True:
                plt.savefig(self.plotpath+'step_'+self.epoch+'_'+id+'.png', dpi=300)
                print('Step test plot saved at:', self.plotpath+'step_'+self.epoch+'_'+id+'.png')
            
            ## plotting results for the evolution test
            fig_evol = plot_abs(model1D, n, n_evol, specs=specs)
            if save == True:
                plt.savefig(self.plotpath+'evol_'+self.epoch+'_'+id+'.png', dpi=300)
                print('Evolution test plot saved at:', self.plotpath+'evol_'+self.epoch+'_'+id+'.png')

            plt.show()

        return np.sum(mace_time)
            

        

