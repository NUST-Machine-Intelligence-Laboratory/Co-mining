"""
@author:    Guan'an Wang
@contact:   guan.wang0706@gmail.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['TripletLoss']


class RankingLoss:

    def __init__(self):
        pass

    def _label2similarity(self, label1, label2):
        '''
        compute similarity matrix of label1 and label2
        :param label1: torch.Tensor, [m]
        :param label2: torch.Tensor, [n]
        :return: torch.Tensor, [m, n], {0, 1}
        '''
        m, n = len(label1), len(label2)
        l1 = label1.view(m, 1).expand([m, n])
        l2 = label2.view(n, 1).expand([n, m]).t()
        similarity = l1 == l2
        return similarity

    def _batch_hard(self, mat_distance, mat_similarity, more_similar):

        if more_similar is 'smaller':
            sorted_mat_distance, _ = torch.sort(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1, descending=True)
            hard_p = sorted_mat_distance[:, 0]
            sorted_mat_distance, _ = torch.sort(mat_distance + (9999999.) * (mat_similarity), dim=1, descending=False)
            hard_n = sorted_mat_distance[:, 0]
            return hard_p, hard_n

        elif more_similar is 'larger':
            sorted_mat_distance, _ = torch.sort(mat_distance + (9999999.) * (1 - mat_similarity), dim=1, descending=False)
            hard_p = sorted_mat_distance[:, 0]
            sorted_mat_distance, _ = torch.sort(mat_distance + (-9999999.) * (mat_similarity), dim=1, descending=True)
            hard_n = sorted_mat_distance[:, 0]
            return hard_p, hard_n


class TripletLoss(RankingLoss):
    '''
    Compute Triplet loss augmented with Batch Hard
    Details can be seen in 'In defense of the Triplet Loss for Person Re-Identification'
    Args:
        margin(float or 'soft'): if float, use nn.MarginRankingLoss, if 'soft', use nn.SoftMarginLoss
    '''

    def __init__(self, margin, metric, reduce=True):
        '''
        :param margin: float or 'soft', for MarginRankingLoss with margin and soft margin
        :param bh: batch hard
        :param metric: l2 distance or cosine distance      ['cosine' or  'euclidean']
        '''

        assert isinstance(margin, float) or margin=='soft', \
            'margin must be type float or value \'soft\', but got {}'.format(margin)
        if isinstance(margin, float):
            self.margin_loss = nn.MarginRankingLoss(margin=margin, reduce=reduce)
        elif margin == 'soft':
            self.margin_loss = nn.SoftMarginLoss(reduce=reduce)
        self.metric = metric


    def __call__(self,*args ):
        '''
        :param emb: torch.Tensor, [m, dim]
        :param label: torch.Tensor, [b]
        '''
        if len(args)==2:
            emb, label = args
            if self.metric == 'cosine':
                mat_dist = cosine_dist(emb, emb)
                mat_sim = self._label2similarity(label, label)
                hard_p, _ = self._batch_hard(mat_dist, mat_sim.float(), more_similar='larger')

                mat_dist = cosine_dist(emb, emb)
                mat_sim = self._label2similarity(label, label)
                _, hard_n = self._batch_hard(mat_dist, mat_sim.float(), more_similar='larger')

                margin_label = -torch.ones_like(hard_p)

            elif self.metric == 'euclidean':
                mat_dist = euclidean_dist(emb, emb)
                mat_sim = self._label2similarity(label, label)
                hard_p, _ = self._batch_hard(mat_dist, mat_sim.float(), more_similar='smaller')

                mat_dist = euclidean_dist(emb, emb)
                mat_sim = self._label2similarity(label, label)
                _, hard_n = self._batch_hard(mat_dist, mat_sim.float(), more_similar='smaller')

                margin_label = torch.ones_like(hard_p)
        elif len(args)==4:
            emb, label, emb_xbm, label_xbm = args
            if self.metric == 'cosine':
                mat_dist = cosine_dist(emb, emb_xbm)
                mat_sim = self._label2similarity(label, label_xbm)
                hard_p, _ = self._batch_hard(mat_dist, mat_sim.float(), more_similar='larger')

                mat_dist = cosine_dist(emb, emb_xbm)
                mat_sim = self._label2similarity(label, label_xbm)
                _, hard_n = self._batch_hard(mat_dist, mat_sim.float(), more_similar='larger')

                margin_label = -torch.ones_like(hard_p)

            elif self.metric == 'euclidean':
                mat_dist = euclidean_dist(emb, emb_xbm)
                mat_sim = self._label2similarity(label, label_xbm)
                hard_p, _ = self._batch_hard(mat_dist, mat_sim.float(), more_similar='smaller')

                mat_dist = euclidean_dist(emb, emb_xbm)
                mat_sim = self._label2similarity(label, label_xbm)
                _, hard_n = self._batch_hard(mat_dist, mat_sim.float(), more_similar='smaller')

                margin_label = torch.ones_like(hard_p)
        
        # print('n:',hard_n,'p:', hard_p)
        if self.margin_loss.__class__.__name__ == 'MarginRankingLoss':            #  (x1,x2,y)      max(0, -y(x1-x2)+ margin)  =  max(0, -(n-p-m))      
            return self.margin_loss(hard_n, hard_p, margin_label) 
        elif self.margin_loss.__class__.__name__ == 'SoftMarginLoss':               #  (x,y)  log(1+e^(-yi*xi)) =  -log(e^yx / e^yx+1)  x larger, loss smaller
            return self.margin_loss(hard_n-hard_p, margin_label)


class TopkRankingLoss:
    
    def __init__(self):
        pass

    def _label2similarity(sekf, label1, label2):
        '''
        compute similarity matrix of label1 and label2
        :param label1: torch.Tensor, [m]
        :param label2: torch.Tensor, [n]
        :return: torch.Tensor, [m, n], {0, 1}
        '''
        m, n = len(label1), len(label2)
        l1 = label1.view(m, 1).expand([m, n])
        l2 = label2.view(n, 1).expand([n, m]).t()
        similarity = l1 == l2
        return similarity
    
    def _batch_hard(self, mat_distance, mat_similarity, more_similar,trip_lowk, trip_upk):
    
        if more_similar is 'smaller':
            sorted_mat_distance, _ = torch.sort(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1, descending=True)
            hard_p = sorted_mat_distance[:, trip_lowk:trip_upk].mean(-1)
            sorted_mat_distance, _ = torch.sort(mat_distance + (9999999.) * (mat_similarity), dim=1, descending=False)
            hard_n = sorted_mat_distance[:, trip_lowk:trip_upk].mean(-1)
            return hard_p, hard_n 

        elif more_similar is 'larger':
            sorted_mat_distance, _ = torch.sort(mat_distance + (9999999.) * (1 - mat_similarity), dim=1, descending=False)
            hard_p = sorted_mat_distance[:, trip_lowk:trip_upk].mean(-1)
            sorted_mat_distance, _ = torch.sort(mat_distance + (-9999999.) * (mat_similarity), dim=1, descending=True)
            hard_n = sorted_mat_distance[:, trip_lowk:trip_upk].mean(-1)
            return hard_p, hard_n


    def _batch_hard_2(self, mat_distance, mat_similarity, more_similar,plowk, pupk, nlowk, nupk):
        
        if more_similar is 'smaller':
            
            sorted_mat_distance_p, index_p = torch.sort(mat_distance * mat_similarity, dim=1, descending=True)
            topp = sorted_mat_distance_p[:, plowk:pupk]
            hard_p = (topp*topp.gt(1e-6)).mean(-1)
            
            # isclean_p = isclean_matrix.gather(dim=1, index = index_p[:, plowk:pupk])*topp.gt(1e-6)

            sorted_mat_distance_n, index_n = torch.sort(mat_distance * (1-mat_similarity), dim=1, descending=False)
            topn = sorted_mat_distance_n[:, nlowk:nupk]
            hard_n =(topn*topn.gt(1e-6)).mean(-1)
            # isclean_n = isclean_matrix.gather(dim=1, index = index_n[:, nlowk:nupk])*topn.gt(1e-6)
            
            return hard_p, hard_n #, isclean_p , isclean_n

        elif more_similar is 'larger':
            sorted_mat_distance_p, index_p = torch.sort(mat_distance * mat_similarity, dim=1, descending=False)
            topp = sorted_mat_distance_p[:, plowk:pupk]
            hard_p = (topp*topp.gt(1e-6)).mean(-1)
            
            # isclean_p = isclean_matrix.gather(dim=1, index = index_p[:, plowk:pupk])*topp.gt(1e-6)
            
            sorted_mat_distance_n, index_n = torch.sort(mat_distance * (1-mat_similarity), dim=1, descending=True)
            topn = sorted_mat_distance_n[:, nlowk:nupk]
            hard_n =(topn*topn.gt(1e-6)).mean(-1)
            # isclean_n = isclean_matrix.gather(dim=1, index = index_n[:, nlowk:nupk])*topn.gt(1e-6)
            
            return hard_p, hard_n #, isclean_p , isclean_n
                
    def _batch_hard_isclean(self, mat_distance, mat_similarity, more_similar,plowk, pupk, nlowk, nupk, isclean_matrix):
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
        if more_similar is 'smaller':
            
            sorted_mat_distance_p, index_p = torch.sort(mat_distance * mat_similarity, dim=1, descending=True)
            topp = sorted_mat_distance_p[:, plowk:pupk]
            hard_p = (topp*topp.gt(1e-6)).mean(-1)
            
            isclean_p = isclean_matrix.gather(dim=1, index = index_p[:, plowk:pupk])*topp.gt(1e-6)

            sorted_mat_distance_n, index_n = torch.sort(mat_distance * (1-mat_similarity), dim=1, descending=False)
            topn = sorted_mat_distance_n[:, nlowk:nupk]
            hard_n =(topn*topn.gt(1e-6)).mean(-1)
            isclean_n = isclean_matrix.gather(dim=1, index = index_n[:, nlowk:nupk])*topn.gt(1e-6)
            
            return hard_p, hard_n , isclean_p , isclean_n

        elif more_similar is 'larger':
            sorted_mat_distance_p, index_p = torch.sort(mat_distance * mat_similarity, dim=1, descending=False)
            topp = sorted_mat_distance_p[:, plowk:pupk]
            hard_p = (topp*topp.gt(1e-6)).mean(-1)
            
            isclean_p = isclean_matrix.gather(dim=1, index = index_p[:, plowk:pupk])*topp.gt(1e-6)
            
            sorted_mat_distance_n, index_n = torch.sort(mat_distance * (1-mat_similarity), dim=1, descending=True)
            topn = sorted_mat_distance_n[:, nlowk:nupk]
            hard_n =(topn*topn.gt(1e-6)).mean(-1)
            isclean_n = isclean_matrix.gather(dim=1, index = index_n[:, nlowk:nupk])*topn.gt(1e-6)
            
            return hard_p, hard_n , isclean_p , isclean_n
        
    def _batch_hard_truth(self, mat_distance, mat_similarity, truth_similarity, more_similar,plowk, pupk, nlowk, nupk):
        
        N = mat_distance.size(0)
        if more_similar is 'smaller':
            
            hard_p = torch.zeros(N).to(mat_distance.device)
            truth_p = torch.zeros(N).to(mat_distance.device)
            for i in range(N):
                p_mat = mat_distance[i][mat_similarity[i]]
                ind = mat_similarity[i]
                sorted_mat_distance_p, index_p  = torch.sort(p_mat, descending=True)
                len_current = ind.sum().item()
                # if int( len_current*pupk) - int( len_current*plowk) >0: 
                    # hard_p[i]= sorted_mat_distance_p[ int( len_current*plowk ): int( len_current*pupk )].mean()
                    # truth_p[i] = truth_similarity[i][ind][ index_p[ int( len_current*plowk ): int( len_current*pupk )] ].float().mean()
                if len_current > int( plowk ):
                    hard_p[i]= sorted_mat_distance_p[ int( plowk ): min(int( pupk ), len_current )].mean()
                    truth_p[i] = truth_similarity[i][ind][ index_p[ int( plowk ): min(int( pupk ), len_current )] ].float().mean()
   
            hard_n = torch.zeros(N).to(mat_distance.device)
            truth_n = torch.zeros(N).to(mat_distance.device)
            for i in range(N):
                n_mat = mat_distance[i][~mat_similarity[i]]
                ind = (~mat_similarity)[i]
                sorted_mat_distance_n, index_n  = torch.sort(n_mat, descending=False)
                len_current = ind.sum().item()
                # if int( len_current*nupk )- int( len_current*nlowk ) >0: 
                    # hard_n[i]= sorted_mat_distance_n[ int( len_current*nlowk ): int( len_current*nupk )].mean()
                    # truth_n[i] = (~truth_similarity)[i][ind][ index_n[ int( len_current*nlowk ): int( len_current*nupk )] ].float().mean()
                if len_current > int( nlowk ):    
                    hard_n[i]= sorted_mat_distance_n[ int( nlowk ): min(int( nupk ),len_current )].mean()
                    truth_n[i] = (~truth_similarity)[i][ind][ index_n[ int( nlowk ): min(int( nupk ),len_current )] ].float().mean()
            
            return hard_p, hard_n , truth_p , truth_n

        elif more_similar is 'larger':
            
            hard_p = torch.zeros(N).to(mat_distance.device)
            truth_p = torch.zeros(N).to(mat_distance.device)
            for i in range(N):
                p_mat = mat_distance[i][mat_similarity[i]]
                ind = mat_similarity[i]

                sorted_mat_distance_p, index_p  = torch.sort(p_mat, descending=False)
                len_current = ind.sum().item()
                # if int( len_current*pupk) - int( len_current*plowk) >0: 
                    # hard_p[i]= sorted_mat_distance_p[ int( len_current*plowk ): int( len_current*pupk )].mean()
                    # truth_p[i] = truth_similarity[i][ind][ index_p[ int( len_current*plowk ): int( len_current*pupk )] ].float().mean()
                if len_current > int( plowk ):    
                    hard_p[i]= sorted_mat_distance_p[ int( plowk ):min(int( pupk ), len_current )].mean()
                    truth_p[i] = truth_similarity[i][ind][ index_p[ int( plowk ): min(int( pupk ), len_current )] ].float().mean()
                    
            hard_n = torch.zeros(N).to(mat_distance.device)
            truth_n = torch.zeros(N).to(mat_distance.device)
            for i in range(N):
                n_mat = mat_distance[i][~mat_similarity[i]]
                ind = (~mat_similarity)[i]
                sorted_mat_distance_n, index_n  = torch.sort(n_mat, descending=True)
                len_current = ind.sum().item()
                # if int( len_current*nupk )- int( len_current*nlowk ) >0:                     
                    # hard_n[i]= sorted_mat_distance_n[ int( len_current*nlowk ): int( len_current*nupk )].mean()
                    # truth_n[i] = (~truth_similarity)[i][ind][ index_n[ int( len_current*nlowk ): int( len_current*nupk )] ].float().mean()
                if len_current > int( nlowk ):   
                    hard_n[i]= sorted_mat_distance_n[ int( nlowk ): min(int( nupk ), len_current)].mean()
                    truth_n[i] = (~truth_similarity)[i][ind][ index_n[ int( nlowk ): min(int( nupk ), len_current)] ].float().mean()
            return hard_p, hard_n , truth_p , truth_n
        
class Topk3Loss(TopkRankingLoss):
    '''
    Compute Triplet loss augmented with Batch Hard
    Details can be seen in 'In defense of the Triplet Loss for Person Re-Identification'
    Args:
        margin(float or 'soft'): if float, use nn.MarginRankingLoss, if 'soft', use nn.SoftMarginLoss
    '''

    def __init__(self, margin, metric, plowk=5, pupk=10, nlowk=5, nupk=10,  reduce=True):
        '''
        :param margin: float or 'soft', for MarginRankingLoss with margin and soft margin
        :param bh: batch hard
        :param metric: l2 distance or cosine distance      ['cosine' or  'euclidean']
        '''

        assert isinstance(margin, float) or margin=='soft', \
            'margin must be type float or value \'soft\', but got {}'.format(margin)
        if isinstance(margin, float):
            self.margin_loss = nn.MarginRankingLoss(margin=margin, reduce=reduce)
            self.margin = margin
        elif margin == 'soft':
            self.margin_loss = nn.SoftMarginLoss(reduce=reduce)
        self.metric = metric
        self.trip_plowk = plowk
        self.trip_pupk = pupk
        self.trip_nlowk = nlowk
        self.trip_nupk = nupk
        
    def __call__(self,*args ):
        '''
        :param emb: torch.Tensor, [m, dim]
        :param label: torch.Tensor, [b]
        '''
        if len(args)==2:
            emb, label = args
            if self.metric == 'cosine':
                mat_dist = cosine_dist(emb, emb)
                mat_sim = self._label2similarity(label, label)
                hard_p, _ = self._batch_hard(mat_dist, mat_sim.float(), more_similar='larger',trip_lowk=self.trip_lowk, trip_upk=self.trip_upk )

                mat_dist = cosine_dist(emb, emb)
                mat_sim = self._label2similarity(label, label)
                _, hard_n = self._batch_hard(mat_dist, mat_sim.float(), more_similar='larger',trip_lowk=self.trip_lowk, trip_upk=self.trip_upk )

                margin_label = -torch.ones_like(hard_p)

            elif self.metric == 'euclidean':
                mat_dist = euclidean_dist(emb, emb)
                mat_sim = self._label2similarity(label, label)
                hard_p, _ = self._batch_hard(mat_dist, mat_sim.float(), more_similar='smaller',trip_lowk=self.trip_lowk, trip_upk=self.trip_upk )

                mat_dist = euclidean_dist(emb, emb)
                mat_sim = self._label2similarity(label, label)
                _, hard_n = self._batch_hard(mat_dist, mat_sim.float(), more_similar='smaller',trip_lowk=self.trip_lowk, trip_upk=self.trip_upk )

                margin_label = torch.ones_like(hard_p)
        elif len(args)==4:
            emb, label, emb_xbm, label_xbm = args
            if self.metric == 'cosine':
                mat_dist = cosine_dist(emb, emb_xbm)
                mat_sim = self._label2similarity(label, label_xbm)
                hard_p, _ = self._batch_hard(mat_dist, mat_sim.float(), more_similar='larger',trip_lowk=self.trip_lowk, trip_upk=self.trip_upk )

                mat_dist = cosine_dist(emb, emb_xbm)
                mat_sim = self._label2similarity(label, label_xbm)
                _, hard_n = self._batch_hard(mat_dist, mat_sim.float(), more_similar='larger',trip_lowk=self.trip_lowk, trip_upk=self.trip_upk )

                margin_label = -torch.ones_like(hard_p)

            elif self.metric == 'euclidean':
                mat_dist = euclidean_dist(emb, emb_xbm)
                mat_sim = self._label2similarity(label, label_xbm)
                hard_p, _ = self._batch_hard(mat_dist, mat_sim.float(), more_similar='smaller',trip_lowk=self.trip_lowk, trip_upk=self.trip_upk )

                mat_dist = euclidean_dist(emb, emb_xbm)
                mat_sim = self._label2similarity(label, label_xbm)
                _, hard_n = self._batch_hard(mat_dist, mat_sim.float(), more_similar='smaller',trip_lowk=self.trip_lowk, trip_upk=self.trip_upk )

                margin_label = torch.ones_like(hard_p)
                
        elif len(args)==6:
            emb, label,isclean, emb_xbm, label_xbm, isclean_xbm = args
            isclean_matrix = isclean.unsqueeze(1).repeat(1,isclean_xbm.size(0)) * isclean_xbm.unsqueeze(0).repeat(isclean.size(0),1)
            if self.metric == 'cosine':
                mat_dist = cosine_dist(emb, emb_xbm)
                mat_sim = self._label2similarity(label, label_xbm)
                hard_p, hard_n, isclean_p , isclean_n  = self._batch_hard_isclean(mat_dist, mat_sim.float(), more_similar='larger',trip_lowk=self.trip_lowk, trip_upk=self.trip_upk , isclean_matrix=isclean_matrix)

                margin_label = -torch.ones_like(hard_p)

            elif self.metric == 'euclidean':
                mat_dist = euclidean_dist(emb, emb_xbm)
                mat_sim = self._label2similarity(label, label_xbm)
                hard_p, hard_n, isclean_p , isclean_n = self._batch_hard_isclean(mat_dist, mat_sim.float(), more_similar='smaller',trip_lowk=self.trip_lowk, trip_upk=self.trip_upk , isclean_matrix=isclean_matrix)

                margin_label = torch.ones_like(hard_p)
                
        elif len(args)==8:
            emb, label,isclean, groundtruth, emb_xbm, label_xbm, isclean_xbm, groundtruth_xbm = args
            isclean_matrix = isclean.unsqueeze(1).repeat(1,isclean_xbm.size(0)) * isclean_xbm.unsqueeze(0).repeat(isclean.size(0),1)
            if self.metric == 'cosine':
                mat_dist = cosine_dist(emb, emb_xbm)
                mat_sim = self._label2similarity(label, label_xbm)
                hard_p, hard_n, isclean_p , isclean_n  = self._batch_hard_isclean(mat_dist, mat_sim.float(), more_similar='larger', plowk=self.trip_plowk, pupk=self.trip_pupk, nlowk=self.trip_nlowk, nupk=self.trip_nupk, isclean_matrix=isclean_matrix)

                margin_label = -torch.ones_like(hard_p)

            elif self.metric == 'euclidean':
                mat_dist = euclidean_dist(emb, emb_xbm)
                mat_sim = self._label2similarity(label, label_xbm)
                hard_p, hard_n, isclean_p , isclean_n = self._batch_hard_isclean(mat_dist, mat_sim.float(), more_similar='smaller', plowk=self.trip_plowk, pupk=self.trip_pupk, nlowk=self.trip_nlowk, nupk=self.trip_nupk, isclean_matrix=isclean_matrix)

                margin_label = torch.ones_like(hard_p)                
        
        # print('n:',hard_n,'p:', hard_p)
        if self.margin_loss.__class__.__name__ == 'MarginRankingLoss':            #  (x1,x2,y)      max(0, -y(x1-x2)+ margin)  =  max(0, -(n-p-m))      
            return self.margin_loss(hard_n, hard_p, margin_label), isclean_p.sum(1)/ (self.trip_pupk-self.trip_plowk), isclean_n.sum(1)/ (self.trip_nupk-self.trip_nlowk), hard_p.mean() , hard_n.mean() , (hard_n-hard_p).mean()
        elif self.margin_loss.__class__.__name__ == 'SoftMarginLoss':               #  (x,y)  log(1+e^(-yi*xi)) =  -log(e^yx / e^yx+1)  x larger, loss smaller
            return self.margin_loss(hard_n-hard_p, margin_label), isclean_p.sum(1)/ (self.trip_pupk-self.trip_plowk), isclean_n.sum(1)/ (self.trip_nupk-self.trip_nlowk), hard_p.mean() , hard_n.mean() , (hard_n-hard_p).mean()



class TopksimpleLoss(TopkRankingLoss):
    '''
    Compute Triplet loss augmented with Batch Hard
    Details can be seen in 'In defense of the Triplet Loss for Person Re-Identification'
    Args:
        margin(float or 'soft'): if float, use nn.MarginRankingLoss, if 'soft', use nn.SoftMarginLoss
    '''

    def __init__(self, margin, metric, plowk=5, pupk=10, nlowk=5, nupk=10,  reduce=True):
        '''
        :param margin: float or 'soft', for MarginRankingLoss with margin and soft margin
        :param bh: batch hard
        :param metric: l2 distance or cosine distance      ['cosine' or  'euclidean']
        '''

        # assert isinstance(margin, float) or margin=='soft', \
        #     'margin must be type float or value \'soft\', but got {}'.format(margin)
        if margin == 'soft':
            self.margin_loss = nn.SoftMarginLoss(reduce=reduce)
        else:
            margin = float(margin)
            self.margin_loss = nn.MarginRankingLoss(margin= margin, reduce=reduce)
            self.margin = margin
        
        self.metric = metric
        self.trip_plowk = plowk
        self.trip_pupk = pupk
        self.trip_nlowk = nlowk
        self.trip_nupk = nupk
        
    def __call__(self,*args ):
        '''
        :param emb: torch.Tensor, [m, dim]
        :param label: torch.Tensor, [b]
        '''
        if self.metric == 'cosine':
            if len(args)==2:
                emb, label = args
                mat_dist = cosine_dist(emb, emb) +1
                mat_sim = self._label2similarity(label, label)
            elif len(args)==4:
                emb, label, emb_xbm, label_xbm = args
                mat_dist = cosine_dist(emb, emb_xbm) +1
                mat_sim = self._label2similarity(label, label_xbm)
            hard_p, hard_n = self._batch_hard_2(mat_dist, mat_sim.float(), more_similar='larger', plowk=self.trip_plowk, pupk=self.trip_pupk, nlowk=self.trip_nlowk, nupk=self.trip_nupk)
            margin_label = -torch.ones_like(hard_p)
        elif self.metric == 'euclidean':
            if len(args)==2:
                emb, label = args
                mat_dist = euclidean_dist(emb, emb)
                mat_sim = self._label2similarity(label, label)
            elif len(args)==4:
                emb, label, emb_xbm, label_xbm = args
                mat_dist = euclidean_dist(emb, emb_xbm)
                mat_sim = self._label2similarity(label, label_xbm)
            hard_p, hard_n = self._batch_hard_2(mat_dist, mat_sim.float(), more_similar='smaller', plowk=self.trip_plowk, pupk=self.trip_pupk, nlowk=self.trip_nlowk, nupk=self.trip_nupk)
            margin_label = -torch.ones_like(hard_p)    
        if self.margin_loss.__class__.__name__ == 'MarginRankingLoss':            #  (x1,x2,y)      max(0, -y(x1-x2)+ margin)  =  max(0, -(n-p-m))      
            return self.margin_loss(hard_n, hard_p, self.margin*margin_label) , hard_p,  hard_n   
        elif self.margin_loss.__class__.__name__ == 'SoftMarginLoss':               #  (x,y)  log(1+e^(-yi*xi)) =  -log(e^yx / e^yx+1)  x larger, loss smaller
            return self.margin_loss(hard_n-hard_p, margin_label), hard_p,  hard_n        
        
       
class TopktruthLoss(TopkRankingLoss):
    '''
    Compute Triplet loss augmented with Batch Hard
    Details can be seen in 'In defense of the Triplet Loss for Person Re-Identification'
    Args:
        margin(float or 'soft'): if float, use nn.MarginRankingLoss, if 'soft', use nn.SoftMarginLoss
    '''

    def __init__(self, margin, metric, plowk=5, pupk=10, nlowk=5, nupk=10,  reduce=True):
        '''
        :param margin: float or 'soft', for MarginRankingLoss with margin and soft margin
        :param bh: batch hard
        :param metric: l2 distance or cosine distance      ['cosine' or  'euclidean']
        '''

        # assert isinstance(margin, float) or margin=='soft', \
        #     'margin must be type float or value \'soft\', but got {}'.format(margin)
        if margin == 'soft':
            self.margin_loss = nn.SoftMarginLoss(reduce=reduce)
        else:
            margin = float(margin)
            self.margin_loss = nn.MarginRankingLoss(margin= margin, reduce=reduce)
            self.margin = margin
        
        self.metric = metric
        self.trip_plowk = plowk
        self.trip_pupk = pupk
        self.trip_nlowk = nlowk
        self.trip_nupk = nupk
        
    def __call__(self,*args ):
        '''
        :param emb: torch.Tensor, [m, dim]
        :param label: torch.Tensor, [b]
        '''
        if self.metric == 'cosine':
            if len(args)==2:
                emb, label = args
                mat_dist = cosine_dist(emb, emb) 
                mat_sim = self._label2similarity(label, label)
            elif len(args)==4:
                emb, label, emb_xbm, label_xbm = args
                mat_dist = cosine_dist(emb, emb_xbm) 
                mat_sim = self._label2similarity(label, label_xbm)
            elif len(args)==6:
                emb, label,truth, emb_xbm, label_xbm,truth_xbm = args
                mat_dist = cosine_dist(emb, emb_xbm) 
                mat_sim = self._label2similarity(label, label_xbm)
                truth_sim = self._label2similarity(truth, truth_xbm)
            hard_p, hard_n, truth_p , truth_n = self._batch_hard_truth(mat_dist, mat_sim, truth_sim, more_similar='larger', plowk=self.trip_plowk, pupk=self.trip_pupk, nlowk=self.trip_nlowk, nupk=self.trip_nupk)
            margin_label = -torch.ones_like(hard_p)   # first should be smaller
        elif self.metric == 'euclidean':
            if len(args)==2:
                emb, label = args
                mat_dist = euclidean_dist(emb, emb)
                mat_sim = self._label2similarity(label, label)
            elif len(args)==4:
                emb, label, emb_xbm, label_xbm = args
                mat_dist = euclidean_dist(emb, emb_xbm)
                mat_sim = self._label2similarity(label, label_xbm)
            elif len(args)==6:
                emb, label,truth, emb_xbm, label_xbm,truth_xbm = args
                mat_dist = euclidean_dist(emb, emb_xbm) 
                mat_sim = self._label2similarity(label, label_xbm)
                truth_sim = self._label2similarity(truth, truth_xbm)
            hard_p, hard_n, truth_p , truth_n = self._batch_hard_truth(mat_dist, mat_sim, truth_sim, more_similar='smaller', plowk=self.trip_plowk, pupk=self.trip_pupk, nlowk=self.trip_nlowk, nupk=self.trip_nupk)
            margin_label = torch.ones_like(hard_p)      #   first input should be ranked higher
        if self.margin_loss.__class__.__name__ == 'MarginRankingLoss':            #  (x1,x2,y)      max(0, -y(x1-x2)+ margin)  =  max(0, (n-p+m))      
            return self.margin_loss(hard_n, hard_p, margin_label), truth_p , truth_n , hard_p.mean(), hard_n.mean() 
        elif self.margin_loss.__class__.__name__ == 'SoftMarginLoss':               #  (x,y)  log(1+e^(-yi*xi)) =  -log(e^yx / e^yx+1)  x larger, loss smaller
            return self.margin_loss(hard_n-hard_p, margin_label), truth_p , truth_n , hard_p.mean(), hard_n.mean()              
        
def cosine_dist(x, y):
    '''
    compute cosine distance between two matrix x and y
    with size (n1, d) and (n2, d) and type torch.tensor
    return a matrix (n1, n2)
    '''

    x = F.normalize(x, p=2, dim=1)
    y = F.normalize(y, p=2, dim=1)
    return torch.matmul(x, y.transpose(0, 1))


def euclidean_dist(x, y):
    """
    compute euclidean distance between two matrix x and y
    with size (n1, d) and (n2, d) and type torch.tensor
    return a matrix (n1, n2)
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(mat1=x, mat2=y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def label2similarity( label1, label2):
    '''
    compute similarity matrix of label1 and label2
    :param label1: torch.Tensor, [m]
    :param label2: torch.Tensor, [n]
    :return: torch.Tensor, [m, n], {0, 1}
    '''
    m, n = len(label1), len(label2)
    l1 = label1.view(m, 1).expand([m, n])
    l2 = label2.view(n, 1).expand([n, m]).t()
    similarity = l1 == l2
    return similarity

def calculate_rank_cleanrate(emb, label, truth, emb_xbm, label_xbm, truth_xbm, prank2=100, nrank2=4096):
    mat_dist = cosine_dist(emb, emb_xbm) 
    mat_sim = label2similarity(label, label_xbm)
    truth_sim = label2similarity(truth, truth_xbm)

    _, index_p = torch.sort(  mat_dist + ~mat_sim*99999, dim=-1, descending=False)
    
    zero_truth_sim = truth_sim.clone()
    zero_truth_sim[ ~mat_sim] = False
    truth_rank_p = zero_truth_sim.gather(dim=1,index = index_p[:,  0: prank2])
    divide_p = torch.sort(mat_sim.float(), dim=-1, descending=True)[0] .sum(0)[ 0: prank2]
    
    
    one_truth_sim = ~truth_sim.clone()
    one_truth_sim[ mat_sim] = False
    _, index_n = torch.sort(  mat_dist - mat_sim*99999,dim=-1, descending=True)
    truth_rank_n =one_truth_sim.gather(dim=1,index = index_n[:,  0: nrank2])
    divide_n = torch.sort( (~mat_sim).float(), dim=-1, descending=True)[0] .sum(0)[ 0: nrank2]


    return truth_rank_p.float().sum(0).int(), divide_p.int(), \
        truth_rank_n.float().sum(0).int(), divide_n.int(),  #mat_dist, mat_sim, truth_sim


    
