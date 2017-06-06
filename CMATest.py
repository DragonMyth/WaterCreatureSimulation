try:
    import multiprocessing as mp
    import cma
    es = cma.CMAEvolutionStrategy(22 * [0.0], 1.0, {'maxiter':10})
    pool = mp.Pool(es.popsize)
    while not es.stop():
        X = es.ask()
        f_values = pool.map_async(cma.felli, X).get()
        # use chunksize parameter as es.popsize/len(pool)?
        es.tell(X, f_values)
        es.disp()
        es.logger.add()
except ImportError:
    pass