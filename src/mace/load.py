import utils
import loss
import matplotlib.pyplot as plt


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

        fig = loss.plot(self.trainloss, self.testloss, log = log, ylim = ylim, limits = limits)

        if save == True:
            plt.savefig(self.plotpath+'loss.png')

        plt.show()

        return
    


