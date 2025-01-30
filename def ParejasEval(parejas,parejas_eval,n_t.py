def ParejasEval(parejas,parejas_eval,n_ttos,n_meas,P,fitness):
    """ Function to obtain the number of stable pairs (NSP) vs the probability P_0 """
    n_imp = parejas.shape[0]
    n_pairs = int(parejas.shape[1]/2)
    if P == 'all':
        prob = np.linspace(0.5,1,int(n_meas/2))
    else:
        prob = np.array([P])
    NSP = np.zeros((n_imp,len(prob)))
    Rel = np.zeros((n_imp,len(prob)))
    HW = np.zeros((n_imp,len(prob)))
    Rel_all = np.zeros(n_imp)
    pairs_eval = np.zeros((n_pairs,3))
    lista = np.concatenate((np.array(range(1,int(n_ttos))),[0]))
    aux = np.cumsum(lista[::-1])
    for imp in range(n_imp):
        parejas_imp = parejas[imp,:]
        for pair in range(n_pairs):
            pareja = parejas_imp[[int(2*pair),int(2*pair +1)]]
            GR_change = False
            if pareja[0]>pareja[1]:
                pareja[0],pareja[1] = pareja[1],pareja[0]
                GR_change = True
            index_pair = aux[int(pareja[0]-1)] + (pareja[1]-pareja[0]) - 1
            pairs_eval[pair,[0,1]] = parejas_eval[int(index_pair),[0,1]]
            if GR_change == True:
                pairs_eval[pair,2] = 1 - parejas_eval[int(index_pair),2]
            else:
                pairs_eval[pair,2] =  parejas_eval[int(index_pair),2]
        for k in range(len(prob)):
            P_0 = prob[k]
            stable_pairs = pairs_eval[:, 1] >= P_0
            if pairs_eval[stable_pairs,1].size > 0:
                NSP[imp,k] = np.sum(stable_pairs)/n_pairs
                Rel[imp,k] = np.mean(pairs_eval[stable_pairs,1])
                HW[imp,k]  = np.mean(pairs_eval[stable_pairs,2])
            else:
                NSP[imp,k] = 0
                Rel[imp,k] = np.nan
                HW [imp,k] = np.nan
        Rel_all[imp] = np.mean(pairs_eval[:,1])
    if P != 'all':
        NSP = NSP.reshape(n_imp,)
    if fitness == 'NSP':
        fitness =  NSP
    elif fitness == 'Rel_all':
        fitness =  Rel_all
    else:
        raise ValueError("Not a valid fitness function")
    return fitness,NSP,Rel,Rel_all,prob,HW