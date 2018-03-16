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
#import sys

    
    
def fl322_batch(basefilename, fldirpath,expdir,
                     EX1,EX2,EXSTEP,EM1,EM2,EMSTEP,ODSTEP,ODSTEPSO,EXSLIT,EMSLIT,TIME,TYPE='em'):   
    '''
    Creates experimental files, collected into bachtes, for Horiba Jobin-Yvon Flurolog Fl-322
    spectrofluorimeter equipped with a PMT (S1) and reference photodiode (R1) together with correction files.
    Works with our instrument, might not work on yours.
    '''
    def batch_gen(limit=50):
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
            
    def exp_file(basefilename,EX1,EM1,endval,STEP):
        
        ex_dict={'Increment':STEP,'Begin':EM1, 'End':endval}
        if TYPE=='ex':
            filename=os.path.join(dir_path,'exc1.xml')
            ex_dict['Begin']=EX1
        else: # 'em'
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
    
    def exp_vector():
        em_vect=np.array(range(EM1,EM2+EMSTEP,EMSTEP))
        ex_vect=np.array(range(EX1,EX2+EXSTEP,EXSTEP))
        if TYPE=='ex':
    #        reducting em points to the ones where excitation finishes after em/2+odstepso
            if ODSTEPSO:
                ex_start=np.min(np.where(ex_vect[None,:]>(em_vect[:,None]/2+ODSTEPSO), ex_vect[None,:],np.inf),axis=1)
                em_vect=em_vect[ex_start!=np.inf]
                ex_start=ex_start[ex_start!=np.inf]
            else:
                ex_start=EX1*np.ones(em_vect.shape)
                
            if ODSTEP:
                ex_stop=np.max(np.where(ex_vect[None,:]<em_vect[:,None]-ODSTEP, ex_vect[None,:],0),axis=1)
                ex_start=ex_start[ex_stop!=0]
                em_vect=em_vect[ex_stop!=0]
                ex_stop=ex_stop[ex_stop!=0]
            else:
                ex_stop=EX2*np.ones(em_vect.shape)
            
            iter_mac=np.zeros([em_vect.size,3])
            iter_mac[:,0]=ex_start
            iter_mac[:,1]=em_vect
            iter_mac[:,2]=ex_stop
            iter_mac=np.array([v for v in list(iter_mac) if v[0]<v[2]])
    #        namenum=0
        else: # TYPE='em'
            if ODSTEPSO:
                em_stop=np.max(np.where(em_vect[None,:]<(ex_vect[:,None]*2-ODSTEPSO), em_vect[None,:],0),axis=1)
                em_stop=em_stop[em_stop!=0]
                ex_vect=ex_vect[em_stop!=0]
                #ex_stop=ex_stop[ex_stop~=0]
            else:
                em_stop=EM2*np.ones(ex_vect.shape)
            
            if ODSTEP:
                em_start=np.min(np.where(em_vect[None,:]>(ex_vect[:,None]+ODSTEP), em_vect[None,:],np.inf),axis=1)
                ex_vect=ex_vect[em_start!=np.inf]
                em_start=em_start[ex_start!=np.inf]
                em_stop=em_stop[ex_start!=np.inf]
            else:
                em_start=EM1*np.ones(ex_vect.shape)
            
            
    #        v=np.array(range(EX1,EX2+EXSTEP,EXSTEP))
            iter_mac=np.zeros([ex_vect.size,3])
            iter_mac[:,0]=ex_vect
            iter_mac[:,1]=em_start
            iter_mac[:,2]=em_stop
            iter_mac=np.array([v for v in list(iter_mac) if v[1]<v[2]])
            #        namenum=1
        
        return iter_mac
        
#    print(os.getcwd())
    basefilename=basefilename[:4]
    dir_path = os.getcwd()
    postfix=basefilename+'_'+TYPE+'_'+str(EX1)+'x'+str(EX2)+'_'+str(EM1)+'x'+str(EM2)+'_'+str(EXSLIT)+'x'+str(EMSLIT)
    fldirpath=os.path.join(fldirpath,postfix)
    expdir=os.path.join(expdir,postfix)
    
    if not os.path.exists(expdir):
        os.makedirs(expdir)

    STEP=[EXSTEP,EMSTEP]
    iter_mac=exp_vector()
    namenum=1 if TYPE=='em' else 0


    file_list=[]    
    for wavel in list(iter_mac):
        # zapisuje plik eksperymentu
        exp_file(basefilename+str(wavel[int(not namenum)]),wavel[0],wavel[1],wavel[2],STEP[namenum])
        # dodaje nazwe pliku do listy
        file_list.append(str(PureWindowsPath(os.path.join(fldirpath,basefilename+str(wavel[int(not namenum)])+'.xml'))))
        
    # tworze batche
    batch_gen(limit=50)

#==============================================================================
#==============================================================================
#==============================================================================
# # #     sortowanie wynikow do macierzy
#==============================================================================
#==============================================================================
#==============================================================================
    
#from difflib import SequenceMatcher

def fl322_to_matrix(dirr):
    
    def remove_nondat(lista):
        inds=[]
        for ind,item in enumerate(lista):
            if '.DAT' not in item[-4:]:
                inds.append(ind)
        for ind in inds[::-1]:
            lista.pop(ind)
        
#    def sort_key(string):
#        ss=os.path.splitext(string)[0]
#        numb=int(re.findall(r'\d+', ss)[-1])
#        return numb

    files=os.listdir(dirr)
    remove_nondat(files)
    files.sort(key=lambda s: int(s[-8:-4]))

    example=np.genfromtxt(os.path.join(dirr,files[0]),dtype=float,delimiter='\t')
    coln=int(example.shape[1]/2)
    l2=len(files)
    l1=example.shape[0]
    wavel=example[:,0]

    
    mac=np.zeros([l1,l2,coln])
    for k in range(coln):
        mac[:,0,k]=example[:,k*2+1]
    
    wavel2=[int(files[0][-8:-4])]
    
    for col,file in enumerate(files[1:]):
        widmo=np.genfromtxt(os.path.join(dirr,file),dtype=float,delimiter='\t')
        wavel2.append(int(file[-8:-4]))
        wave=widmo[:,0]
        # dopasowywuje zakres wavel
        order=int(not wave[0]<wavel[0]) # 0 jesli wave jest pierwsze
        matr=[wave,wavel]    
        if not set(wave).intersection(wavel): # no common values
            valsep=wave[1]-wave[0]
            
            gapvals=np.arange(matr[order][-1]+valsep,matr[1-order][0],valsep)
            wavel=np.concatenate((matr[order],gapvals,matr[1-order]),axis=0)
            mac1=np.zeros([wave.shape[0]+gapvals.shape[0],l2,coln])
        else: # there is some intersection
#            print('len(set(wave)) ',len(set(wave)))
#            print('len(set(wavel)) ',len(set(wavel)))
#            print('len(set(wave).union(set(wavel))) ',np.array(list(set(wave).union(set(wavel))).sort()))
            wavel=np.array(list(set(wave).union(set(wavel))))
            wavel.sort(axis=0)
#            print(wavel.shape[0])
            mac1=np.zeros([wavel.shape[0]-mac.shape[0],l2,coln])
        mac_list=[mac1,mac]
        mac=np.concatenate((mac_list[order],mac_list[1-order]),axis=0)
        
        ind1start=list(wavel).index(wave[0])
        ind1end=list(wavel).index(wave[-1])+1
        sl=slice(ind1start,ind1end,1)
#        print(wavel[0],wavel[-1])
        for k in range(coln):
#            print([*vector[::order],k])
#            print(mac.shape)
            mac[sl,col+1,k]=widmo[:,k*2+1]
            
            
            
#    lm=len(range(em1,em2+emstep,emstep))
#    lx=len(range(ex1,ex2+exstep,exstep))
#    
#    mac=np.zeros([lm,lx,coln])
##    print(mac.shape)
#    order=1-2*(TYPE=='ex')
#    if TYPE=='ex':
#        l=lm
#    else: 
#        l=lx
#        
#    for l,file in enumerate(files):
#        widmo=np.genfromtxt(os.path.join(dirr,file),dtype=float).swapaxes(0,1)
#        l1=widmo.shape[1]
#        vector=[slice(l1),l]
##        print(l)
#        for k in range(coln):
##            print([*vector[::order],k])
##            print(mac.shape)
#            mac[[*vector[::order],k]]=widmo[k*2+1,:]
        

    for n in range(coln):
        name=os.path.join(dirr,'macierz_'+str(n))
        np.savetxt(name,np.squeeze(mac[:,:,n]),delimiter=' ')
   
#    print(wavel2)
    [X,M]=np.meshgrid(wavel,np.array(wavel2),indexing='ij')
    np.savetxt(os.path.join(dirr,'X'),X,delimiter=' ')
    np.savetxt(os.path.join(dirr,'M'),M,delimiter=' ')
    
         
        
        
        
        
        
        
        