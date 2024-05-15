import numpy as np
import re


loc = '/STER/silkem/MACE/data_info/'

specs = np.loadtxt(loc+'rate16.specs', usecols=(1), dtype=str, skiprows = 1) 



def get_elements(specs):
    """
    Extract different elements present in a list of chemical species.

    Input:
    - specs: list of chemical species

    NOTE:
    Does not extract electrons!
    """
    elements = []

    for spec in specs:
        # Split on capital letters
        comps = re.findall('[A-Z][^A-Z]*', spec)

        for comp in comps:
            # Get element
            elem = re.findall('[a-zA-Z]+', comp)
            if len(elem) > 1:
                raise ValueError('More than one element found!')
            element = elem[0]
            elements.append(element)

    return sorted(list(set(elements)))


## Get elements & add charge
elements = get_elements(specs)
elements.append('charge')

## Create dictionary for elements
elements_dict = {e: i for i, e in enumerate(elements)}

## Create matrix M
M = np.zeros((len(specs), len(elements)))

charge_dict = {'+': +1, '-': -1}

for i, spec in enumerate(specs):
    # Split on capital letters
    comps = re.findall('[A-Z][^A-Z]*', spec)

    charge = 0
    if spec == 'e-':
        charge = -1

    for comp in comps:
        # Get element
        elem = re.findall('[a-zA-Z]+', comp) # This does not find electrons!
        if len(elem) > 1:
            raise ValueError('More than one element found!')
        element = elem[0]

        # Get number of atoms
        numb = re.findall('\d+', comp)
        if len(numb) > 1:
            raise ValueError('More than one number found!')
        number = int(numb[0]) if numb else 1 

        # Get + or - for charge
        char = re.findall('[+-]+', comp)
        if len(char) > 1:
            raise ValueError('More than one charge found!')
        elif len(char) == 1:
            charge += charge_dict[char[0]]
        
        # print(element, elements_dict[element], number, charge)

        M[i][elements_dict[element]] = number

    M[i][len(elements_dict)-1] = charge

    print(spec, M[i])

## Save matrix M as numpy array
np.save(loc+'M_rate16.npy', M)