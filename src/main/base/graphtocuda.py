from ..ml.setup import MLSetup



class DGLtoCUDA(MLSetup):
    def __init__(self, arguments):
        super().__init__(arguments)
        self.params = arguments

    def graph2cuda(self,Ggs):
        Ggs = Ggs.to(self.device)
        Ggs.ndata['x'].to(self.device)
        Ggs.edata['x'].to(self.device)
        return Ggs
    
    def multicompgraph2cuda(self,Ggs):
        res = []
        for Gg in Ggs:
            res.append(self.graph2cuda(Gg))
        return res
    
    def movegraphs2cuda(self,Ggs):
        if isinstance(Ggs, (list,tuple)):
            Ggs = self.multicompgraph2cuda(Ggs)
        else:
            Ggs = self.graph2cuda(Ggs)
        return Ggs

    def addon2cuda(self,add):
        add = add.to(self.device)
        return add
    
    def multicompaddon2cuda(self,add):
        res = []
        for a in add:
            res.append(self.addon2cuda(a))
        return res

    def moveaddons2cuda(self,add):
        if isinstance(add, (list,tuple)):
            add = self.multicompaddon2cuda(add)
        else:
            add = self.addon2cuda(add)
        return add
    

    def nonecase(self,RGgs,PGgs,RAdd,PAdd):
        RGgs = RGgs if 'RGgs' in locals() else None
        PGgs = PGgs if 'PGgs' in locals() else None
        RAdd = RAdd if 'RAdd' in locals() else None
        PAdd = PAdd if 'PAdd' in locals() else None
        return RGgs,PGgs,RAdd,PAdd
    

    def molecularLoadGraphstoCUDA(self,Rgs,Radd):
        Rgs = self.movegraphs2cuda(Rgs)
        if self.params.addons is not None:
            Radd = self.moveaddons2cuda(Radd)
        else:
            Radd = None
        return Rgs,Radd

    def reactionLoadGraphstoCUDA(self,Rgs,Pgs,Radd,Padd):
        Rgs,Radd = self.molecularLoadGraphstoCUDA(Rgs,Radd)
        Pgs,Padd = self.molecularLoadGraphstoCUDA(Pgs,Padd)
        return Rgs,Pgs,Radd,Padd
        
    def LoadGraphstoCUDA(self,Rgs,Pgs,Radd,Padd):
        if 'reaction' in self.params.graph:
            RGgs,PGgs,RAdd,PAdd = self.reactionLoadGraphstoCUDA(Rgs,Pgs,Radd,Padd)
        elif 'molecular' in self.params.graph:
            RGgs,RAdd = self.molecularLoadGraphstoCUDA(Rgs,Radd)
            PGgs,PAdd = None,None
        return RGgs,PGgs,RAdd,PAdd
    

    def LoadAddonstoCUDA(self,Radd,Padd):
        if 'reaction' in self.params.graph:
            RAdd = self.moveaddons2cuda(Radd)
            PAdd = self.moveaddons2cuda(Padd)
        elif 'molecular' in self.params.graph:
            RAdd = self.moveaddons2cuda(Radd)
            PAdd = None
        return RAdd,PAdd