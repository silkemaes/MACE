
# Readme inputfiles voor training MACE


## Uitleg bij inputparameters

- lr = learning rate    
- tot_epochs = totaal aantal epochs
- nb_epochs = number of epochs dat de losses geschaald worden met de eerste fractie
    Dus tot_epochs - nb_epochs = hoeveel epochs er getraind wordt met tweede fractie
- losstype = type losses die moeten meegenomen worden tijdens training
- z_dim = dimensie latente ruimte
- nb_samples = hoeveelheid samples gebruikt voor de training


## Grenzen input variabele parameters

### input
- lr : [1.e-5, 1.e-3] (of misschien maar t.e.m. 1e-4)
- tot_epochs = 100
- nb_epochs = 70
- losstype = mse_idn_evo
- z_dim : [8, 10, 16, 32]
- nb_samples = 18000 (dan heb ik nog 314 modellen over voor te testen)

### loss fracties
    altijd in hoeveelheiden van 10
    
- mse1 : [1e3, 1e7]
- rel1 = 0
- evo1 : [1e0, 1e3]     maar altijd minstens 1 grootteorde kleiner dan mse1
- idn1 : [1e2, 1e5]

- mse2 : [1, 1e2]       ik zou op het einde misschien enkel de mse boosten? De andere losses zullen hier automatisch op reageren    
- rel2 = 0
- evo1 = 1
- idn1 = 1

