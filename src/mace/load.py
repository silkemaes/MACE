'''
This script loads a trained MACE model 
and provides the user with the possibility to apply (test) the model on a test dataset.

The class Trained_MACE() contains the loaded model, 
together with the training and test losses, and the meta data.
'''



import src.mace.utils       as utils
import src.mace.loss        as loss
import matplotlib.pyplot    as plt



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

        self.trainloss = loss.LoadedLoss(self.loc, self.meta, 'train')

        self.testloss  = loss.LoadedLoss(self.loc, self.meta, 'valid')

        self.plotpath = self.loc + 'figs/'

        utils.makeOutputDir(self.plotpath)

        
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

        return fig
    


        

