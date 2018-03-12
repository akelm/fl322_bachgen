#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 17:31:35 2018

@author: ania
"""

import os
import re
import numpy as np
import copy
import xml.etree.ElementTree as ET
from pathlib import PureWindowsPath


    
    
def fl322_batch(basefilename, fldirpath,expdir,
                     EX1,EX2,EXSTEP,EM1,EM2,EMSTEP,ODSTEP,EXSLIT,EMSLIT,TIME,TYPE='em'):   
    '''
    Creates experimental files, collected into bachtes, for Horiba Jobin-Yvon Flurolog Fl-322
    spectrofluorimeter equipped with a PMT (S1) and reference photodiode (R1) together with correction files.
    Works with our instrument, might not work on yours.
    '''
    def batch_gen(file_list,expdir,limit=50):
        filename=os.path.join(dir_path,'batch1.jyb')
        tree = ET.parse(filename)
        root = tree.getroot()    
        # batch_operations=list(root.iterfind('BatchOperations'))[0]
        
        filename_batch=os.path.join(dir_path,'batch_exp.xml')
        tree_batch=ET.parse(filename_batch)
        root_batch=tree_batch.getroot()
        ex_file=root_batch.find('ExperimentFile')
        batch_ind=root_batch.find('BatchIndex')
        
        batch_nums=range( len(file_list)//limit + bool(len(file_list)%limit) )
        
        for batchn in batch_nums:
            # zeruje batchoperations
            op_list=[]
            # iteruje po plikach
            for num in range( min(limit,len(file_list)-batchn*limit) ):
                ex_file.set('FileName',file_list[num])
                batch_ind.set('IndexNumber',str(num+1))
                op_list.append(copy.deepcopy(root_batch))
            root[3].extend(op_list)
    #        ET.dump(tree)
            tree.write(os.path.join(expdir,'batch_'+str(batchn+1)+'.jyb'),encoding='UTF-8')     
            
    def exp_file(basefilename,expdir,EX1,EXSLIT,EM1,EMSLIT,endval,STEP,TIME,fldirpath='c:',TYPE='em'):
        
        ex_dict={'Increment':STEP,'Begin':EM1, 'End':endval}
        if TYPE=='ex':
            filename=os.path.join(dir_path,'exc1.xml')
            ex_dict['Begin']=EX1
        else:
            filename=os.path.join(dir_path,'em1.xml')
                    
        tree = ET.parse(filename)
        root = tree.getroot()
        children=root.getchildren()
        tags=[child.tag for child in children]
        root_dict=dict(zip(tags,range(len(children))))
        
        # general
        ind=root_dict['general']
        root[ind].set('resultsFile',basefilename)
        root[ind].set('resultsDir',str(PureWindowsPath( fldirpath)))
        
        # StartOps
        so_dict={0: EX1, 1: EXSLIT, 2: EXSLIT, 3: EXSLIT, 5: EM1, 6: EMSLIT, 7: EMSLIT, 8: EMSLIT, 12:TIME, 19:TIME}
        ind=root_dict['StartOps']
        for key in so_dict.keys():
            root[ind][key][0][0].set('Value',str(so_dict[key]))
        #print(ET.tostring(root))
        # ExpAxis
        ind=root_dict['ExpAxis']
        for key in ex_dict.keys():
            root[ind][0].set(key,str(ex_dict[key]))
        root[ind][0][0][0][0][0].set('Value',str(ex_dict['Begin']))
        
    #    print(expdir+basefilename)
        tree.write(os.path.join(expdir,basefilename+'.xml'),encoding='UTF-8')
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    postfix=basefilename+'_'+TYPE+'_'+str(EX1)+'x'+str(EX2)+'_'+str(EM1)+'x'+str(EM2)+'_'+str(EXSLIT)+'x'+str(EMSLIT)
    fldirpath=os.path.join(fldirpath,postfix)
    expdir=os.path.join(expdir,postfix)
    
    if not os.path.exists(expdir):
        os.makedirs(expdir)
    
    if TYPE=='ex':
        v=np.array(range(EM1,EM2+EMSTEP,EMSTEP))
        iter_mac=np.zeros([v.size,3])
        iter_mac[:,0]=EX1
        iter_mac[:,1]=v
        iter_mac[:,2]=np.array(list(map(min,np.ones(len(v))*EX2,v-ODSTEP)))
        iter_mac=np.array([v for v in list(iter_mac) if v[0]<v[2]])
        STEP=EXSTEP
        namenum=0
    else:
        v=np.array(range(EX1,EX2+EXSTEP,EXSTEP))
        iter_mac=np.zeros([v.size,3])
        iter_mac[:,0]=v
        iter_mac[:,1]=np.array(list(map(max,np.ones(len(v))*EM1,v+ODSTEP)))
        iter_mac[:,2]=EM2
        iter_mac=np.array([v for v in list(iter_mac) if v[1]<v[2]])
        STEP=EMSTEP
        namenum=1

    file_list=[]    
    for wavel in list(iter_mac):
        # zapisuje plik eksperymentu
        exp_file(basefilename+str(wavel[1 if TYPE=='ex' else 0]),expdir,wavel[0],EXSLIT,wavel[1],EMSLIT,wavel[2],STEP,
                                             TIME,fldirpath,TYPE)
        # dodaje nazwe pliku do listy
        file_list.append(str(PureWindowsPath(os.path.join(fldirpath,basefilename+str(wavel[namenum])+'.xml'))))
        
    # tworze batche
    batch_gen(file_list,expdir,limit=50)

#==============================================================================
#==============================================================================
#==============================================================================
# # #     sortowanie wynikow do macierzy
#==============================================================================
#==============================================================================
#==============================================================================
    


def fl322_to_matrix(dirr,ex1,ex2,exstep,em1,em2,emstep,TYPE='em'):
    def sort_key(string):
        ss=os.path.splitext(string)[0]
        numb=int(re.findall(r'\d+', ss)[-1])
        return numb

    files=os.listdir(dirr)
    files.sort(key=lambda s: sort_key(s))
#    print(files)
    inds=[]
    for file in files:
#        print(file)
#        print(os.path.splitext(file)[1])
        if os.path.splitext(file)[1]!='.DAT':
            inds.append(files.index(file))
#            files.pop(ind)
#            print(file in files)
    for ind in inds[::-1]:
        files.pop(ind)


    example=np.genfromtxt(os.path.join(dirr,files[3]),dtype=float)
    coln=int(example.shape[1]/2)

    lm=len(range(em1,em2+emstep,emstep))
    lx=len(range(ex1,ex2+exstep,exstep))
    
    mac=np.zeros([lm,lx,coln])
#    print(mac.shape)
    order=1-2*(TYPE=='ex')
    if TYPE=='ex':
        l=lm
    else: 
        l=lx
    l=0    
    for file in files:
        widmo=np.genfromtxt(os.path.join(dirr,file),dtype=float).swapaxes(0,1)
        l1=widmo.shape[1]
        vector=[slice(l1),l]
#        print(l)
        for k in range(coln):
#            print([*vector[::order],k])
#            print(mac.shape)
            mac[[*vector[::order],k]]=widmo[k*2+1,:]
        l+=1

    for n in range(coln):
        name=os.path.join(dirr,'macierz_'+str(n))
        np.savetxt(name,np.squeeze(mac[:,:,n]),delimiter=' ')
        
        
        
        
        
        
        
        