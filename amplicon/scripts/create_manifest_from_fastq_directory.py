#There are two methods to prepare input manifest.csv file which is required for import fastq.gz into qiime2
#The first one is preparing manifiest.csv manually from excel file, and converted to csv file. 
#The second one is using python scripts t prepare manifest.csv file 
#I suppose all files is paired of sequences
#!/usr/bin/env python

import sys
import os
import glob
import re

# usage
usage = '''
create_manifest_from_fastq_directory.py fastq_dir_in manifest_out_pe

    fastq_dir_in - full path of directory containing fastq.gz files
        of the form CN18SESPkoa_SC36_S80_L001_R1_001.fastq.
    manifest_out_pe - output path of manifest_pe.csv

    This script makes the following assumptions:
        - the first characters after the sample names are "_S[0-9]{1,3}"
        - Lane data is present ("_L001"), if more than one lane, the fastq.gz should be merged at begining.
        - R1 and R2 files are both present
'''

if len(sys.argv) < 2:
    print(usage)
    sys.exit()

# input paths
path_fastq = sys.argv[1]

# output paths
path_manifest_pe = sys.argv[2] # 'manifest_pe.csv'

# list of full fastq.gz paths for R1 and R2 files
list_fastq_full = sorted(glob.glob(os.path.join(path_fastq, '*.fastq.gz')))
list_fastq_full_forward = [x for x in list_fastq_full if ('_R1_' in x)]
list_fastq_full_reverse = [x for x in list_fastq_full if ('_R2_' in x)]

if len(list_fastq_full_forward) != len(list_fastq_full_reverse)
    print ("Please check paired of file")
    sys.exit()


def compareList(l1,l2):
   l1.sort()
   l2.sort()
   if(l1!=l2):
        print ("Please check paired of file")
        sys.exit()
    else:
        return l1

fnames_f = []
fl = len(list_fastq_full_forward)
for f in range(fl):
    fname = os.path.basename(list_fastq_full_forward[f]).split("_")[0]
    fanmes.appned(fname)

fnames_r = []
fl = len(list_fastq_full_reverse)
for f in range(fl):
    fname = os.path.basename(list_fastq_full_reverse[f]).split("_")[0]
    fanmes_r.appned(fname)

fanems = compareList(fnames_f,fnames_r)


Targetdict={"sample-id":[], "":[], "forward-absolute-filepath":[], "reverse-absolute-filepath":[]}

for f in range(len(fnames)):
    try:
        Targetdict["sample-id"] = fnames[f]
        Targetdict["forward-absolute-filepath"] = list_fastq_full_forward[f] 
        Targetdict["reverse-absolute-filepath"] = list_fastq_full_reverse[f]
    except Exception:
        print ("There are something wrong in paired of files")

# write manifest_pe

target_df=pd.DataFrame.from_dict(Targetdict)

target_df.to_csv("manifest.csv")
