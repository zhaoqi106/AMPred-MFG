import torch
import torch.nn as nn
import dgl
import math

from torch.nn import Linear

from layer import GraphAttentionLayer
from .MLP import MLP
from .embedding import LinearBn
from .layers import DGL_MPNNLayer
from .readout import Readout

    


class Transformer(nn.Module):
    def __init__(self,args):
        super(Transformer,self).__init__()
        self.linear_layers = nn.ModuleList([nn.Linear(args.hid_dim, args.hid_dim,bias=False) for _ in range(3)])
        self.W_o=nn.Linear(args.hid_dim, args.hid_dim)
        self.heads=args.heads
        self.hid_dim = args.hid_dim
        self.d_k=args.hid_dim//args.heads
        self.q_linear = nn.Linear(args.hid_dim, args.hid_dim)
        self.k_linear = nn.Linear(args.hid_dim, args.hid_dim)
        self.v_linear = nn.Linear(args.hid_dim, args.hid_dim)
    def forward(self,fine_messages,coarse_messages,motif_features):
        batch_size=fine_messages.shape[0]
        hid_dim=fine_messages.shape[-1]
        # Q=motif_features
        # K=[]
        # K.append(fine_messages.unsqueeze(1))
        # K.append(coarse_messages.unsqueeze(1))
        # K=torch.cat(K,dim=1)
        # Q=Q.view(batch_size, -1, 1,hid_dim).transpose(1, 2)
        # K=K.view(batch_size, -1, 1,hid_dim).transpose(1, 2)
        # V=K
        Q = self.q_linear(motif_features)
        K = self.k_linear(motif_features)
        V = self.v_linear(motif_features)
        Q = Q.view(batch_size, -1, 1, hid_dim).transpose(1, 2)
        K = K.view(batch_size, -1, 1, hid_dim).transpose(1, 2)
        V = V.view(batch_size, -1, 1, hid_dim).transpose(1, 2)

        Q, K, V = [l(x).view(batch_size, -1,self.heads,self.d_k).transpose(1, 2)
                                      for l, x in zip(self.linear_layers, (Q,K,V))]   
        #print(Q[0],K.transpose(-2, -1)[0])
        message_interaction=torch.matmul( Q,K.transpose(-2, -1))/self.d_k
        #print(message_interaction[0])
        att_score=torch.nn.functional.softmax(message_interaction,dim=-1)
        motif_messages=torch.matmul(att_score, V).transpose(1, 2).contiguous().view(batch_size, -1, hid_dim)
        motif_messages=self.W_o(motif_messages)
        return motif_messages.squeeze(1)

class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()

        self.weight = nn.Parameter(torch.randn(65, 65), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(65), requires_grad=True)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        X, A = inputs
        xw = torch.matmul(X, self.weight)
        out = torch.matmul(A, xw)

        out += self.bias
        out = self.relu(out)

        return out, A
class MGraphModel(nn.Module):
    def __init__(self):
        super(MGraphModel, self).__init__()
        self.num_head = 4

        self.layers = nn.Sequential(
            GCN(),
            GCN(),
        )

        self.proj = nn.Sequential(
            nn.Linear(26000, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.att = GraphAttentionLayer()

    def forward(self, X, A):
        # GCN
        # out = self.layers((X, A))[0]

        # GAT
        features = []
        for i in range(X.shape[0]):
            feature_temp = []
            x, a = X[i], A[i]
            # 2层gat
            for _ in range(self.num_head):
                ax = self.att(x, a)
                feature_temp.append(ax)
            feature_temp = torch.cat(feature_temp, dim=1)
            features.append(feature_temp)
        out = torch.stack(features, dim=0)
        out = out.view(out.size(0), -1)
        out = self.proj(out)

        return out

class AMPred_MFG(nn.Module):
    def __init__(self,
                 out_dim: int,
                 args,
                 criterion_atom,
                 criterion_motif,
                 criterion_figerprint,
                 ):
        super(AMPred_MFG, self).__init__()
        self.args=args
        self.atom_encoder = nn.Sequential(
            LinearBn(args.atom_in_dim,args.hid_dim),
            nn.ReLU(inplace = True),
            nn.Dropout(p =args.drop_rate),
            LinearBn(args.hid_dim,args.hid_dim),
            nn.ReLU(inplace = True)
        )
        self.motif_encoder = nn.Sequential(
            LinearBn(args.ss_node_in_dim,args.hid_dim),
            nn.ReLU(inplace = True),
            nn.Dropout(p =args.drop_rate),
            LinearBn(args.hid_dim,args.hid_dim),
            nn.ReLU(inplace = True)
        )
        self.step=args.step 
        self.agg_op=args.agg_op
        self.mol_FP=args.mol_FP
        self.motif_mp_layer=DGL_MPNNLayer(args.hid_dim,nn.Linear(args.ss_edge_in_dim,args.hid_dim*args.hid_dim),args.resdual)
        self.atom_mp_layer=DGL_MPNNLayer(args.hid_dim,nn.Linear(args.bond_in_dim,args.hid_dim*args.hid_dim),args.resdual)
        self.motif_update=nn.GRUCell(args.hid_dim,args.hid_dim)
        self.atom_update=nn.GRUCell(args.hid_dim,args.hid_dim)
        self.fp_readout = Readout(args,ntype='atom',use_attention=args.attention)
        self.motif_readout=Readout(args,ntype='func_group',use_attention=args.attention)
        self.tr=Transformer(args)
        #define the predictor

        atom_MLP_inDim=args.hid_dim*2
        Motif_MLP_inDim=args.hid_dim*2
        if self.mol_FP=='atom':
            atom_MLP_inDim=atom_MLP_inDim+args.mol_in_dim
        elif self.mol_FP=='ss':
            Motif_MLP_inDim=Motif_MLP_inDim+args.mol_in_dim
            #2215
            atom_MLP_inDim=167
        elif self.mol_FP=='both':
            atom_MLP_inDim=atom_MLP_inDim+args.mol_in_dim
            Motif_MLP_inDim=Motif_MLP_inDim+args.mol_in_dim

        
        self.output_af = MLP(atom_MLP_inDim,
                                 out_dim,
                                 dropout_prob=args.drop_rate, 
                                 num_neurons=args.num_neurons,input_norm=args.input_norm)
        self.output_ff = MLP(Motif_MLP_inDim,
                             out_dim,
                             dropout_prob=args.drop_rate,
                             num_neurons=args.num_neurons,input_norm=args.input_norm)
        self.criterion_atom =criterion_atom
        self.criterion_motif =criterion_motif
        self.criterion_figerprint =criterion_figerprint
        self.dist_loss=torch.nn.MSELoss(reduction='none')
        self.fu2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=args.drop_rate),
        )
        self.lin_skip = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=args.drop_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=args.drop_rate),
        )
        self.lin_beta = nn.Sequential(
            nn.Linear(384, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=args.drop_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=args.drop_rate),
        )
        self.fpecfp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=args.drop_rate),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=args.drop_rate),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=args.drop_rate),
        )
        self.fprdit = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=args.drop_rate),
        )
        self.fpmacc = nn.Sequential(
            nn.Linear(167, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=args.drop_rate),
        )
        self.fp_all = nn.Sequential(
            nn.Linear(3239, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(p=args.drop_rate),
        )
        self.ffn = nn.Sequential(
            nn.Linear(359, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=args.drop_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=args.drop_rate),
        )
        self.ffn1 = nn.Linear(128, 1)

        self.graph = MGraphModel()
    def forward(self, g, af, bf, fnf, fef,mf,labels,macc,fp,rdit,X,A):
        #gat
        X = self.graph(X, A)
        with g.local_scope():
            ufnf=self.motif_encoder(fnf)
            uaf=self.atom_encoder(af)

            for i in range(self.step):
                ufnm=self.motif_mp_layer(g[('func_group', 'interacts', 'func_group')],ufnf,fef)
                uam=self.atom_mp_layer(g[('atom', 'interacts', 'atom')],uaf,bf)
                g.nodes['atom'].data['_uam']=uam
                if self.agg_op=='sum':
                    g.update_all(dgl.function.copy_u('_uam','uam'),dgl.function.sum('uam','agg_uam'),\
                             etype=('atom', 'a2f', 'func_group'))
                elif self.agg_op=='max':
                    g.update_all(dgl.function.copy_u('_uam','uam'),dgl.function.max('uam','agg_uam'),\
                             etype=('atom', 'a2f', 'func_group'))
                elif self.agg_op=='mean':
                    g.update_all(dgl.function.copy_u('_uam','uam'),dgl.function.mean('uam','agg_uam'),\
                             etype=('atom', 'a2f', 'func_group'))         
                augment_ufnm=g.nodes['func_group'].data['agg_uam']

                ufnm=self.tr(augment_ufnm,ufnm,ufnf)

                
                ufnf=self.motif_update(ufnm,ufnf)
                uaf=self.atom_update(uam,uaf)
            #readout
            rdit = self.fpecfp(self.fprdit(rdit))
            ecfp = self.fpecfp(fp)
            macc = self.fpmacc(macc)



            motif_readout=self.motif_readout(g,ufnf)
            motif_representation=motif_readout
            motif_pred=self.ffn(motif_representation)
            fp_all = torch.cat((macc,ecfp,rdit),dim=-1)

            fu1 = torch.cat((fp_all,X),-1)
            fu2 =motif_pred
            # fu2 = self.fu2(fu2)
            # fu2 =motif_pred
            x_r = self.lin_skip(fu1)
            # x_r= macc
            beta = self.lin_beta(torch.cat([x_r, fu2,   x_r-fu2], dim=1)).sigmoid()

            out = beta * x_r + (1 - beta) * fu2

            task_type = 'classification'
            dist_loss = torch.nn.MSELoss(reduction='none')

            if task_type == 'classification':
                logits = torch.sigmoid(self.ffn1(out))
                dist_fp_fg_loss = dist_loss(torch.sigmoid(x_r), torch.sigmoid(motif_pred)).mean()

            x_r = self.ffn1(out)
            motif_pred =self.ffn1(motif_pred)
            return x_r,motif_pred,logits,dist_fp_fg_loss,out
