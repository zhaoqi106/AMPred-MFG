import datetime
import argparse
import numpy as np
import dgl
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch import scatter
from torch.utils.tensorboard import SummaryWriter


from models.model import AMPred_MFG
from models.utils import GraphDataset_Classification,GraphDataLoader_Classification,\
                  AUC,RMSE,\
                  GraphDataset_Regression,GraphDataLoader_Regression,confusion_matrix1
from torch.optim import Adam
from data.split_data import get_classification_dataset
# from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

writer = SummaryWriter('logs')

def main(args):
    max_score_list=[]
    max_aupr_list=[]
    task_type=None
    if args.dataset in ['Tox21', 'ClinTox',
                      'SIDER', 'BBBP', 'BACE']:
        task_type='classification'
    else:
        task_type='regression'
    strpath = 'results/AMES/test.txt'
    for seed in range(args.seed,args.seed+args.folds):
        print('folds:',seed)
        f = open(strpath,'a',encoding='utf-8')
        f.write(f"##########folds:{seed}\n")
        f.close()
        if task_type=='classification':
            metric=AUC

            train_gs,train_ls,train_tw,val_gs,val_ls,test_gs,test_ls,\
                morgan_fp_list_train,morgan_fp_list_val,morgan_fp_list_test,\
                maccs_fp_train,maccs_fp_val,maccs_fp_test,\
                rdit_fp_train,rdit_fp_val,rdit_fp_test,\
                X_train,X_val,X_test,\
                A_train,A_val,A_test=get_classification_dataset(args.dataset,args.n_jobs,seed,args.split_ratio)

            print(len(train_ls),len(val_ls),len(test_ls),train_tw)

            # morgan_fp_list_train = torch.FloatTensor(morgan_fp_list_train)
            train_ds = GraphDataset_Classification(train_gs, train_ls,morgan_fp_list_train,maccs_fp_train,rdit_fp_train,X_train,A_train)
            train_dl = GraphDataLoader_Classification(train_ds, num_workers=0, batch_size=args.batch_size,
                                       shuffle=args.shuffle)
            task_pos_weights=train_tw
            criterion_atom = torch.nn.BCEWithLogitsLoss(pos_weight=task_pos_weights.to(args.device))
            criterion_fg = torch.nn.BCEWithLogitsLoss(pos_weight=task_pos_weights.to(args.device))
            criterion_figerprint = torch.nn.BCEWithLogitsLoss(pos_weight=task_pos_weights.to(args.device))
        else:
            print("")

        val_gs = dgl.batch(val_gs).to(args.device)
        val_labels=val_ls.to(args.device)
        morgan_fp_list_val=torch.tensor(morgan_fp_list_val).to(args.device)
        maccs_fp_val = torch.tensor(maccs_fp_val).to(args.device)
        rdit_fp_val = torch.tensor(rdit_fp_val).to(args.device)

        test_gs=dgl.batch(test_gs).to(args.device)
        test_labels=test_ls.to(args.device)
        morgan_fp_list_test=torch.tensor(morgan_fp_list_test).to(args.device)
        maccs_fp_test = torch.tensor(maccs_fp_test).to(args.device)
        rdit_fp_test = torch.tensor(rdit_fp_test).to(args.device)

        X_val = torch.tensor(X_val).to(args.device)
        A_val = torch.tensor(A_val).to(args.device)
        X_test = torch.tensor(X_test).to(args.device)
        A_test= torch.tensor(A_test).to(args.device)
        model = AMPred_MFG(val_labels.shape[1],
                      args,
                      criterion_atom,
                      criterion_fg,
                      criterion_figerprint
                      ).to(args.device)
        print(model)
        opt = Adam(model.parameters(),lr=args.learning_rate)
        print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))
        #scheduler = torch.optim.lr_scheduler.CosineAnnea lingWarmRestarts(opt,T_0=50,eta_min=1e-4,verbose=True)
    
        
        best_val_score=0 if task_type=='classification' else 999
        best_val_aupr=0 if task_type=='classification' else 999
        best_epoch=0
        best_test_score=0
        best_test_aupr=0
        
        for epoch in range(args.epoch):
            f = open(strpath,'a',encoding='utf-8')
            model.train()
            traYAll = []
            traPredictAll = []
            loss_all=0.0
            out_all =[]

            for i, (gs, labels,macc,fp,rdit,x,a) in enumerate(train_dl):
                traYAll += labels.detach().cpu().numpy().tolist()

                gs = gs.to(args.device)
                labels = labels.to(args.device).float()
                fp = fp.to(args.device).float()
                macc = macc.to(args.device).float()
                rdit = rdit.to(args.device).float()
                x = x.to(args.device)
                a = a.to(args.device)
                # print(i)
                af=gs.nodes['atom'].data['feat']
                bf = gs.edges[('atom', 'interacts', 'atom')].data['feat']
                fnf = gs.nodes['func_group'].data['feat']
                fef=gs.edges[('func_group', 'interacts', 'func_group')].data['feat']
                molf=gs.nodes['molecule'].data['feat']
                fp_pred,fg_pred,logits,dist_fp_fg_loss,out= model(gs, af, bf,fnf,fef,molf,labels,macc,fp,rdit,x,a)
                out = out.cpu().detach().numpy().tolist()
                # motif__1.append(motif__)
                # macc__1.append(macc__)
                # logitss = torch.cat((motif__,macc__),-1)
                # ########logitss_list+=logitss.detach().cpu().numpy().tolist()
                ######################################
                # if task_type == 'classification':
                #     logits = (torch.sigmoid(fp_pred)+torch.sigmoid(fg_pred)) / 2
                #     dist_fp_fg_loss = dist_loss(torch.sigmoid(fp_pred), torch.sigmoid(fg_pred)).mean()
                # else:
                #     logits = (fp_pred + fg_pred) / 2
                #     dist_fp_fg_loss = dist_loss(fp_pred, fg_pred).mean()
                # loss_atom = criterion_atom(fp_pred, labels).mean()
                loss_fp= criterion_atom(fp_pred, labels).mean()
                loss_motif = criterion_fg(fg_pred, labels).mean()
                loss = loss_motif + loss_fp + args.dist * dist_fp_fg_loss
                ##################################################
                opt.zero_grad()
                loss.backward()
                loss_all+=loss
                opt.step()
                traPredictAll += logits.detach().cpu().numpy().tolist()
                out_all +=out
            # from models.utils import tsen
            # tsen(epoch,out_all,traYAll)
            train_score,train_AUPRC=metric(traYAll,traPredictAll)



            model.eval()
            PED = []
            with torch.no_grad():
                    val_fp = morgan_fp_list_val
                    val_macc = maccs_fp_val
                    val_rdit = rdit_fp_val
                    val_X = X_val
                    val_A = A_val
                    val_af = val_gs.nodes['atom'].data['feat']
                    val_bf = val_gs.edges[('atom', 'interacts', 'atom')].data['feat']
                    val_fnf = val_gs.nodes['func_group'].data['feat']
                    val_fef=val_gs.edges[('func_group', 'interacts', 'func_group')].data['feat']
                    val_molf=val_gs.nodes['molecule'].data['feat']
                    val_logits_fp,val_logits_motif,val_logits,val_dist_fp_fg_loss,val_out= model(val_gs, val_af, val_bf, val_fnf,val_fef,val_molf,val_labels,val_macc,val_fp,val_rdit,val_X,val_A)

                    test_rdit = rdit_fp_test
                    test_fp = morgan_fp_list_test
                    test_macc = maccs_fp_test
                    test_X = X_test
                    test_A = A_test

                    test_af = test_gs.nodes['atom'].data['feat']
                    test_bf = test_gs.edges[('atom', 'interacts', 'atom')].data['feat']
                    test_fnf = test_gs.nodes['func_group'].data['feat']
                    test_fef=test_gs.edges[('func_group', 'interacts', 'func_group')].data['feat']
                    test_molf=test_gs.nodes['molecule'].data['feat']
                    test_logits_fp,test_logits_motif,test_logits,test_dist_fp_fg_loss,test_out= model(test_gs, test_af, test_bf, test_fnf,test_fef,test_molf,val_labels,test_macc,test_fp,test_rdit,test_X,test_A)
                    ###################################################
                    # if task_type=='classification':
                    #     val_logits=(torch.sigmoid(val_logits_fp)+torch.sigmoid(val_logits_motif))/2
                    #     test_logits=(torch.sigmoid(test_logits_fp)+torch.sigmoid(test_logits_motif))/2
                    # else:
                    #     val_logits=(val_logits_fp+val_logits_motif)/2
                    #     test_logits=(test_logits_fp+test_logits_motif)/2

                    # pred_y.extend(test_logits.detach().cpu().numpy())




                    val_score,val_AUPRC= metric(val_labels.detach().cpu().numpy().tolist(), val_logits.detach().cpu().numpy().tolist())

                    test_score,test_AUPRC=metric(test_labels.detach().cpu().numpy().tolist(), test_logits.detach().cpu().numpy().tolist())

                    PED.extend(test_logits.detach().cpu().numpy().round())
                    # test_lables=test_labels.detach().cpu()
                    # ture_y.extend(test_lables)


                    # acc = accuracy_score(ture_y, PED)
                    # auc = roc_auc_score(ture_y, pred_y)
                    ###################################################
                    if task_type=='classification':
                        if best_val_score<val_score:
                            best_val_score=val_score
                            best_test_score=test_score
                            best_epoch=epoch
                        if best_val_aupr<val_AUPRC:
                            best_val_aupr=val_AUPRC
                            best_test_aupr=test_AUPRC
                            best_epoch=epoch
                        print('#####################')

                        print("-------------------Epoch {}-------------------".format(epoch))

                        f.write(f"Epoch:{epoch}\t")
                        print(f":loss=={loss_all / i}")
                        # f.write(f"loss=={loss_all / i}\n")
                        print("Train AUROC: {}".format(train_score)," Train AUPRC: {}".format(train_AUPRC))
                        print("Val AUROC: {}".format(val_score)," Val AUPRC: {}".format(val_AUPRC))
                        print("Test AUROC: {}".format(test_score)," Test AUPRC: {}".format(test_AUPRC))
                        # writer.add_scalar('Train/auc1', train_score, global_step=epoch)
                        # writer.add_scalar('Val/auc1', val_score, global_step=epoch)
                        # writer.add_scalar('Test/auc1', test_score, global_step=epoch)
                        f.write(
                            f"Train AUROC: {train_score}\t" + f"Val AUROC: {val_score}\t" + f'Test AUROC: {test_score}\t\n')


                        acc = accuracy_score(test_labels.detach().cpu().numpy().tolist(), PED)
                        precision = precision_score(test_labels.detach().cpu().numpy().tolist(), PED)
                        recall = recall_score(test_labels.detach().cpu().numpy().tolist(), PED)
                        f1 = f1_score(test_labels.detach().cpu().numpy().tolist(), PED)
                        print('测试集ACC,precision,recall,f1', acc,precision,recall,f1)
                        TN, FP, FN, TP = confusion_matrix(test_labels.detach().cpu().numpy().tolist(), PED).ravel()

                        SPE = TN / (TN + FP)
                        SEN = TP / (TP + FN)
                        NPV = TN / (TN + FN)
                        PPV = TP / (TP + FP)
                        MCC = (TP * TN - FP * FN) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5
                        print('TN, FP, FN, TP:', TN, FP, FN, TP)

                        print('SPE, SEN, NPV, PPV, MCC:', SPE, SEN, NPV, PPV, MCC)
                        if epoch == 199:
                            f.write(f"ACC:{acc},SPE:{SPE}, SEN:{SEN}, NPV:{NPV}, PPV:{PPV}, MCC:{MCC}\n")
                    elif task_type=='regression':
                        if best_val_score>val_score:
                            best_val_score=val_score
                            best_test_score=test_score
                            best_epoch=epoch
                        print('#####################')
                        print("-------------------Epoch {}-------------------".format(epoch))
                        f.write(f"Epoch:{epoch}\n")
                        print("Train RMSE: {}".format(train_score))
                        print("Val RMSE: {}".format(val_score))
                        print('Test RMSE: {}'.format(test_score))
                        # writer.add_scalar('Train/rmse', train_score, global_step=epoch)
                        # writer.add_scalar('Val/auc', val_score, global_step=epoch)
                        # writer.add_scalar('Test/auc',test_score , global_step=epoch)
                        f.write(f"Train RMSE: {train_score}\t"+f"Val RMSE: {val_score}\t"+f'Test RMSE: {test_score}\t\n')


            ###################################################

        max_score_list.append(best_test_score)
        max_aupr_list.append(best_test_aupr)
        print('best model in epoch ',best_epoch)
        print('best val score is ',best_val_score)
        print('test score in this epoch is',best_test_score)
        if task_type=='classification':
            print('best val aupr is ',best_val_aupr)
            print('corresponding best test aupr is ',best_test_aupr)

        f.close()
    print("AUROC:\n")
    print(max_score_list)
    print(np.mean(max_score_list),'+-',np.std(max_score_list))
    print("AUPRC:\n")
    print(np.mean(max_aupr_list),'+-',np.std(max_aupr_list))

    try:
        f=open(strpath+datetime.datetime.now().strftime('%m%d_%H%M')+'.txt','a',encoding='utf-8');
        f.write('\n'.join([key+': '+str(value) for key, value in vars(args).items()])+'\n')
        if task_type=="classification":
            f.write("AUROC:")
        f.write(str(np.mean(max_score_list))+'+-'+str(np.std(max_score_list))+'\n')
        for i in max_score_list:
            f.write(str(i)+" ")
        if task_type=="classification":
            f.write("\nAUPRC:")
            f.write(str(np.mean(max_aupr_list))+'+-'+str(np.std(max_aupr_list))+'\n')
        f.close()
    except:
        f=open(args.dataset+'result_'+datetime.datetime.now().strftime('%m%d_%H%M')+'.txt','a',encoding='utf-8');  
        f.write('\n'.join([key+': '+str(value) for key, value in vars(args).items()])+'\n')
        if task_type=="classification":
            f.write("AUROC:")
        f.write(str(np.mean(max_score_list))+'+-'+str(np.std(max_score_list))+'\n')
        for i in max_score_list:
            f.write(str(i)+" ")
        if task_type=="classification":
            f.write("AUPRC:")
            f.write(str(np.mean(max_aupr_list))+'+-'+str(np.std(max_aupr_list))+'\n')
        f.close()
    
def write_record(path, message):
    file_obj = open(path, 'a')
    file_obj.write('{}\n'.format(message))
    file_obj.close()



if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str,help='dataset name')
    p.add_argument('--seed', type=int, default=0, help='seed used to shuffle dataset')
    p.add_argument('--atom_in_dim', type=int, default=37, help='atom feature init dim')
    p.add_argument('--bond_in_dim', type=int, default=13, help='bond feature init dim')
    p.add_argument('--ss_node_in_dim', type=int, default=50, help='func group node feature init dim')
    p.add_argument('--ss_edge_in_dim', type=int, default=37, help='func group edge feature init dim')
    p.add_argument('--mol_in_dim', type=int, default=167, help='molecule fingerprint init dim')
    p.add_argument('--learning_rate', type=float, default=1e-5, help='Adam learning rate')
    p.add_argument('--epoch', type=int, default=200, help='train epochs')
    p.add_argument('--batch_size', type=int, default=512, help='batch size for train dataset')
    p.add_argument('--num_neurons', type=list, default=[512],help='num_neurons in MLP')
    p.add_argument('--input_norm', type=str, default='layer', help='input norm')
    p.add_argument('--drop_rate',  type=float, default=0.2, help='dropout rate in MLP')
    p.add_argument('--hid_dim', type=int, default=96, help='node, edge, fg hidden dims in Net')
    p.add_argument('--device', type=str, default='cuda:0', help='fitting device')
    p.add_argument('--dist',type=float,default=0.005,help='dist loss func hyperparameter lambda')
    p.add_argument('--split_ratio',type=list,default=[0.8,0.1,0.1],help='ratio to split dataset')
    p.add_argument('--folds',type=int,default=1,help='k folds validation')
    p.add_argument('--n_jobs',type=int,default=10,help='num of threads for the handle of the dataset')
    p.add_argument('--resdual',type=bool,default=False,help='resdual choice in message passing')
    p.add_argument('--shuffle',type=bool,default=False,help='whether to shuffle the train dataset')
    p.add_argument('--attention',type=bool,default=True,help='whether to use global attention pooling')
    p.add_argument('--step',type=int,default=4,help='message passing steps')
    p.add_argument('--agg_op',type=str,choices=['max','mean','sum'],default='mean',help='aggregations in local augmentation')
    p.add_argument('--mol_FP',type=str,choices=['atom','ss','both','none'],default='ss',help='cat mol FingerPrint to Motif or Atom representation'
                   )
    p.add_argument('--gating_func',type=str,choices=['Softmax','Sigmoid','Identity'],default='Sigmoid',help='Gating Activation Function'
                   )
    p.add_argument('--ScaleBlock',type=str,choices=['Share','Contextual'],default='Contextual',help='Self-Rescaling Block'
                   )
    p.add_argument('--heads',type=int,default=4,help='Multi-head num')
    args = p.parse_args()
    main(args)
