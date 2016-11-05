from __future__ import division

import glob
import pandas as pd
import numpy as np
import pickle
import pandas
import numpy
import copy



import logging
reload(logging)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs).03d %(name)s %(message)s",
    datefmt='%Y-%m-%dT%H:%M:%S'
)
logger = logging.getLogger(__name__)

import protease_experimental_analysis.data as data
#for src_data_file in ['ssm2_tryp']:
#for src_data_file in ['rd3_tryp','rd3_redo_tryp']:
#for src_data_file in ['rd2_tryp','rd2_redo_tryp','rd1_tryp','pdb_tryp']:
#for src_data_file in ['ssm2_tryp','ssm2_chymo']:


def fixmap(m):
    newm = copy.deepcopy(m)
    newm['sel_k'] = [newm['sel_k']]
    return newm

src_data_files = """ssm2_tryp
rd3_tryp
rd3_redo_tryp
rd2_tryp
rd2_redo_tryp
rd1_tryp
pdb2_tryp""".split('\n')

src_data_files = """rd1_tryp
pdb2_tryp
rd2_redo_chymo
rd1_chymo
pdb2_chymo""".split('\n')

#src_data_files=['rd3_redo_chymo']

for src_data_file in src_data_files:
    for cycle in range(4):
        print src_data_file, 'cycle', cycle
        file_prefix = '%s.sel_k_4_flat2.5_nd_minfrac5e-7_c%s' % (src_data_file, cycle)


        counts_df=pd.read_csv('%s.counts' % src_data_file,delim_whitespace=True)

        src_data = data.model_input[src_data_file]
        for x in src_data:
            if x != 0:
                if src_data[x]['num_selected'] != None:
                    src_data[x]['min_fraction'] = min([0.5 / src_data[x]['num_selected'], 5e-7])
                #src_data[x]['fraction_selected'] = None


        import protease_experimental_analysis.protease_sequencing_model_edit2 as protease_sequencing_model
        src_model = (
            protease_sequencing_model.FractionalSelectionModel(response_fn="expit",
                homogenous_k=True,
                leak_prob_mass=False,
                sel_k=4,
                sel_k_sd=0.0001,
                sel_k_w=2.5,
                init_sel_k=4) #originally I did sel_k 1.5 and sel_k_sd 0.02
            .build_model(src_data))

        unrestricted = (
            protease_sequencing_model.FractionalSelectionModel(response_fn="expit",
                homogenous_k=True,
                leak_prob_mass=False) #originally I did sel_k 1.5 and sel_k_sd 0.02
            .build_model(src_data))

        model=src_model

        #if cycle == 2:
        #    with open('%s1.map' % file_prefix[0:-1]) as file:
        #        mymap = pickle.load(file)
        #    #mymap = model.find_MAP()
        #else:
        if cycle == 0:
            mymap = model.find_MAP()
        else:
            mymap = model.find_MAP(mymap)

        print 'sel_k', mymap['sel_k']
        print 'logp', unrestricted.logp(fixmap(mymap))


        with open('%s.map' % file_prefix,'w') as file:
            pickle.dump(mymap,file)
        
        with open('%s.output.steepness' % file_prefix,'w') as file:
            file.write('%s\n' % mymap['sel_k'])
            file.write('%s\n' % unrestricted.logp(fixmap(mymap)))
            
        counts_df['ec50']=mymap['sel_ec50']


        parallel_ci = []
        for i in range(len(mymap['sel_ec50'])):
            if i % 1000 == 0:
                logger.info("scan_ec50_outliers: %i / %i", i, len(mymap['sel_ec50']))
            out = model.estimate_ec50_cred(mymap, i, cred_spans=[.95])
            parallel_ci.append((out['cred_intervals'][.95][0], out['cred_intervals'][.95][1]))
        
        lbound, ubound = zip(*parallel_ci)
        
        counts_df['ec50_95ci_lbound']=lbound
        counts_df['ec50_95ci_ubound']=ubound
        counts_df['ec50_95ci']=np.array(ubound)-np.array(lbound)
        
        sel_sum=model.model_outlier_summary(mymap)
        mean_llh = numpy.nanmean(
            numpy.stack([v["sel_log_likelihood"] for v in sel_sum.values()]),
            axis=0)
        mean_llh[~numpy.isfinite(mean_llh)] = -250

        mean_llh_signed = numpy.nanmean(
            numpy.stack([v["sel_log_likelihood_signed"] for v in sel_sum.values()]),
            axis=0)
        mean_llh_signed[~numpy.isfinite(mean_llh_signed)] = -250


        llh = pd.DataFrame(np.array([v["sel_log_likelihood"] for v in sel_sum.values()]).T)



        sum_llh = numpy.nansum(
            numpy.stack([v["sel_log_likelihood"] for v in sel_sum.values()]),
            axis=0)


        sum_llh[~numpy.isfinite(mean_llh)] = -1000
        sum_llh[sum_llh<-1000] = -1000

        sum_llh_signed = numpy.nansum(
            numpy.stack([v["sel_log_likelihood_signed"] for v in sel_sum.values()]),
            axis=0)
        sum_llh_signed[~numpy.isfinite(mean_llh_signed)] = -1000
        sum_llh_signed=np.clip(sum_llh_signed,-1000,1000)
        

        llh_signed=pd.DataFrame(np.array([v["sel_log_likelihood_signed"] for v in sel_sum.values()]).T)

        for i in llh:
            print 'llh', i, np.nanmedian(llh[i]), np.nanmedian(llh_signed[i]), np.nansum(llh_signed[i])

        counts_df['sum_llh'] = sum_llh
        counts_df['sum_llh_signed'] = sum_llh_signed
        counts_df['mean_llh'] = mean_llh
        counts_df['mean_llh_signed'] = mean_llh_signed
        
        counts_df.to_csv('%s.output' % file_prefix,sep=' ',index=False)
        print
        print
        print
        print
