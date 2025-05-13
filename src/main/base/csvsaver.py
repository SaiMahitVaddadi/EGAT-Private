from ..ml.setup import MLSetup
import pandas as pd
import numpy as np
import os

def check_metrics_list(metrics_list):
    return metrics_list is None or all(metric is None for metric in metrics_list)

class CSVCommands(MLSetup):
    def __init__(self, arguments):
        super().__init__(arguments)
        self.params = arguments
        
    def idcolumn(self):
        self.columns = ['ID']

    def _baseiterator(self,iterator):
        if isinstance(self.params.smiles,list):
            for smi in self.params.smiles:
                self.columns += [f'{iterator}_{smi}']
        else:
            self.columns += [f'{iterator}']

    def rxntypecolumn(self):
        self._baseiterator('RTYPE')

    def smilescolumn(self,prefix='R'):
        self._baseiterator(f'{prefix}smiles')

    def inchicolumn(self,prefix='R'):
        self._baseiterator(f'{prefix}inchi')

    def _basetensoriterator(self,iterator):
        if isinstance(iterator,list):
            self.columns  += [t+'_PRED' for t in iterator]
            self.columns  += iterator
        else:
            self.columns  += [iterator+'_PRED']
            self.columns  += [iterator]

    def targetcolumn(self):
        self._basetensoriterator(self.params.target)
    
    def additionalcolumn(self):
        self._basetensoriterator(self.params.additional)
    

    def setupcolumns(self):
        self.idcolumn()
        self.rxntypecolumn()
        self.smilescolumn()
        if 'reaction' in self.params.graph: self.smilescolumn('P')
        self.inchicolumn()
        if 'reaction' in self.params.graph: self.inchicolumn('P')
        self.targetcolumn()
        if self.params.additional != None: self.additionalcolumn()
    
    def CreateCSV(self):
        csv = []
        self.setupcolumns()
        return csv
    

    def CreateAllCSVs(self):
        train = self.CreateCSV()
        val = self.CreateCSV()
        test = self.CreateCSV()
        return train, val, test
    
    

    def setupsavedata(self,train,val,test,ensemble=None,fold=None,epoch=None):
        self.datatosave = {'train': train, 'val': val, 'test': test}
        self.filenames = {'train': 'best_train', 'val': 'best_val', 'test': 'best_test'}
        self.checkforensembles(self.filenames,ensemble)
        self.checkforfolds(self.filenames,fold)
        self.checkforepochs(self.filenames,epoch)
        self.capwithcsv(self.filenames)

    def setupsaveembeddings(self,train,val,test,ensemble=None,fold=None,epoch=None):
        self.embeddingdatatosave = {'train': train, 'val': val, 'test': test}
        self.embeddingfilenames = {'train': 'train_embeddings', 'val': 'val_embeddings', 'test': 'test_embeddings'}
        self.checkforensembles(self.embeddingfilenames,ensemble)
        self.checkforfolds(self.embeddingfilenames,fold)
        self.checkforepochs(self.embeddingfilenames,epoch)
        self.capwithcsv(self.embeddingfilenames)

    def checkforensembles(self,filenames,ensemble=None):
        if ensemble is not None:
            self.ensemble = ensemble
            for key in filenames.keys():
                filenames[key] += f'_ensemble_{self.ensemble}'

    def checkforfolds(self,filenames,fold=None):
        if fold is not None:
            self.fold = fold
            for key in filenames.keys():
                filenames[key] += f'_fold_{self.fold}'

    def checkforepochs(self,filenames,epoch=None):
        if epoch is not None:
            self.epoch = epoch
            for key in filenames.keys():
                filenames[key] += f'_epoch_{self.epoch}'

    def capwithcsv(self,filenames):
        for key in filenames.keys():
            if not filenames[key].endswith('.csv'):
                filenames[key] += '.csv'

    def savedata(self,filenames,datatosave):
        ###### SAVE RESULTS AS CSV    
        for key in datatosave:
            if datatosave[key] is not None:
                df = pd.DataFrame(self.datatosave[key], columns=self.columns)
                df.to_csv(filenames[key])

    def SaveData(self,train,val,test,ensemble=None,fold=None,epoch=None):
        self.setupsavedata(train,val,test,ensemble,fold,epoch)
        self.savedata(self.filenames,self.datatosave)

    def SaveEmbeddingsData(self,train,val,test,ensemble=None,fold=None,epoch=None):
        self.setupsaveembeddings(train,val,test,ensemble,fold,epoch)
        self.savedata(self.embeddingfilenames,self.embeddingdatatosave)

    def SavePredictedData(self,data):
        df = pd.DataFrame(data, columns=self.columns)
        df.to_csv(self.params.save_path)

    def SavePredictedEmbeddings(self,embeddings):
        df = pd.DataFrame(embeddings)
        df.to_csv(self.params.save_path.split('.')[0]+'_embeddings.csv')

    def createdatadict(self,train_loss_list, val_loss_list, test_loss_list, lr,epoch=None):
        data = {
            'epoch': [epoch],
            'learning_rate': [lr],
            'train_loss': [np.mean(train_loss_list)],
            'val_loss': [np.mean(val_loss_list)],
            'test_loss': [np.mean(test_loss_list) if test_loss_list is not None else None]
        }
        return data
    
    def addmetrictodata(self,metric_list,data,metric_name):
        if not check_metrics_list(metric_list):
            data[metric_name] = [np.mean(metric_list) if metric_list is not None else None]
        else:
            data[metric_name] = [None]
        return data
    
    def grabfilename(self,ensemble=None,fold=None):
        filename = 'losses'
        if ensemble is not None: filename += f'_ensemble_{ensemble}'
        if fold is not None: filename += f'_fold_{fold}'
        filename += '.csv'
        return filename

    def savelossfile(self,filename,df):
        if not os.path.isfile(filename):
            df.to_csv(filename, index=False)
        else:
            df.to_csv(filename, mode='a', header=False, index=False)

    def SaveLossesToCSV(self, train_loss_list, val_loss_list, test_loss_list, lr, 
                        epoch,train_metrics_list,val_metrics_list,test_metrics_list,train_comb_metrics_list,
                        val_comb_metrics_list,test_comb_metrics_list,ensemble=None,fold=None):
        # Create a dictionary with the data
        data = self.createdatadict(train_loss_list, val_loss_list, test_loss_list, lr, epoch)
        for metrics in [train_metrics_list, val_metrics_list, test_metrics_list, train_comb_metrics_list, val_comb_metrics_list, test_comb_metrics_list]:
            metric_name = metrics.__name__.replace('_list', '')
            data = self.addmetrictodata(metrics, data, f'{metric_name}_metrics')

        # Convert the dictionary to a DataFrame
        df = pd.DataFrame(data)
        
        # Define the filename
        filename = self.grabfilename(ensemble, fold)
        
        # Save the DataFrame to a CSV file
        self.savelossfile(filename, df)

    def SaveAttentionMaps(self, attention_maps, fold=None,ensemble=None):
        # Save these as database.
        pass
       




    