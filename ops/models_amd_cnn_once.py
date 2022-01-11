from torch import nn
from ops.basic_ops import ConsensusModule
from ops.transforms import *
from torch.nn.init import normal_, constant_
import torch.nn.functional as F
#from efficientnet_pytorch import EfficientNet
from ops.net_flops_table import feat_dim_dict
from ops.amd_net_flops_table import feat_dim_of_res50_block


from torch.distributions import Categorical
import math
from .transformer import TransformerModel, PositionalEncoding, ScaledDotProductAttention
import pdb

def init_hidden(batch_size, cell_size):
    init_cell = torch.Tensor(batch_size, cell_size).zero_()
    if torch.cuda.is_available():
        init_cell = init_cell.cuda()
    return init_cell


class SqueezeTwice(torch.nn.Module):
    def forward(self, x):
        return x.squeeze(-1).squeeze(-1)
    
class TSN_Amd(nn.Module):
    def __init__(self, num_class, num_segments,
                 base_model='resnet101', consensus_type='avg', before_softmax=True, dropout=0.8,
                 crop_num=1, partial_bn=True, pretrain='imagenet', fc_lr5=False, args=None):
        super(TSN_Amd, self).__init__()
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.pretrain = pretrain
        self.reverese_try_cnt = 0
        self.fc_lr5 = fc_lr5
        self.is_shift = False

        # TODO(yue)
        self.args = args
        self.rescale_to = args.rescale_to
        if self.args.ada_reso_skip:
            base_model = self.args.backbone_list[0] if len(self.args.backbone_list) >= 1 else None
        self.base_model_name = base_model
        self.num_class = num_class
        self.multi_models = False
        self.time_steps = self.num_segments
        self.use_transformer = args.use_transformer       

        self._prepare_base_model(base_model) #return self.base_model 
        self.consensus = ConsensusModule(consensus_type, args=self.args)

        if self.args.ada_reso_skip:
            self.reso_dim = self._get_resolution_dimension()
            self.skip_dim = len(self.args.skip_list)
            self.action_dim = self._get_action_dimension()
            self._prepare_policy_net()
            self._extends_to_multi_models()
            self._prepare_fc(num_class)          #return self.new_fc

            
        if self.args.ada_depth_skip:
            self.block_cnn_dict   = nn.ModuleDict()
            self.block_rnn_dict   = nn.ModuleDict()
            self.block_fc_dict    = nn.ModuleDict()
            self.action_fc_dict   = nn.ModuleDict()
            self.pos_encoding_dict = nn.ModuleDict()
            self.block_rnn2fc_dict = nn.ModuleDict()
            if self.args.use_distil_loss_to_rnn or self.args.use_conf_btw_blocks:
                self.block_pred_rnn_fc_dict = nn.ModuleDict()
            elif self.args.use_distil_loss_to_cnn:
                self.block_pred_cnn_fc_dict = nn.ModuleDict()
            
            if self.args.use_early_exit:
                self.early_exit_dict = nn.ModuleDict()
                
            if self.args.voting_policy:
                self.vote_fc_dict = nn.ModuleDict()
                self.vote_dim = 2 #0(not support), 1(support)
                
            if self.args.use_local_policy_module:
                self.local_policy_dict = nn.ModuleDict()


            
            self.block_rnn_list = self.args.block_rnn_list
            
            if self.args.skip_twice:
                self.amd_action_dim = 3 # 0 , 1, 2 
            else:
                self.amd_action_dim = 2 #0 , 1 (skip(0) or pass(1))


            self._split_base_cnn_to_block(self.base_model)
            self._prepare_policy_block(self.base_model)
            self._prepare_pos_encoding()
            
        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)
            
        if self.use_transformer:
            self.transformer = TransformerModel()

    def _split_base_cnn_to_block(self, _model):
        self.block_cnn_dict['base']   = torch.nn.Sequential(*(list(_model.children())[:4]))
        self.block_cnn_dict['conv_2'] = torch.nn.Sequential(*(list(_model.children())[4]))
        self.block_cnn_dict['conv_3'] = torch.nn.Sequential(*(list(_model.children())[5]))
        self.block_cnn_dict['conv_4'] = torch.nn.Sequential(*(list(_model.children())[6]))
        self.block_cnn_dict['conv_5'] = torch.nn.Sequential(*(list(_model.children())[7]))
        
    def _prepare_policy_block(self, _model): #avg-pooling / fc / P.E. / lstm 
        def make_a_linear(input_dim, output_dim):
            linear_model = nn.Linear(input_dim, output_dim)
            normal_(linear_model.weight, 0, 0.001)
            constant_(linear_model.bias, 0)
            return linear_model
        
        for name in self.args.block_rnn_list:
#             if name is not 'base':
            feat_dim = feat_dim_of_res50_block[name]
            self.block_fc_dict[name] = torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                SqueezeTwice(),
                make_a_linear(feat_dim, feat_dim)
            )
            if self.args.diff_to_rnn:
                feat_dim = feat_dim*2
            
#             self.block_rnn2fc_dict[name] = make_a_linear(feat_dim, self.args.hidden_dim)
            self.block_rnn_dict[name] = torch.nn.LSTMCell(input_size=feat_dim, hidden_size=self.args.hidden_dim)
            self.action_fc_dict[name] = make_a_linear(self.args.hidden_dim, self.amd_action_dim)
            if self.args.use_early_stop:
                self.early_stop_decision_block = make_a_linear(self.num_class, self.amd_action_dim) #kill or not

            if self.args.use_distil_loss_to_rnn or self.args.use_conf_btw_blocks:
                self.block_pred_rnn_fc_dict[name] =  make_a_linear(self.args.hidden_dim, self.num_class)

            elif self.args.use_distil_loss_to_cnn:
                self.block_pred_cnn_fc_dict[name] = torch.nn.Sequential(
                    torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                    SqueezeTwice(),
                    make_a_linear(feat_dim, feat_dim)
                )
            if self.args.voting_policy:
                self.vote_fc_dict[name] =  make_a_linear(self.args.hidden_dim, 1)
                
            if self.args.use_local_policy_module:
                self.local_policy_dict[name] = torch.nn.Sequential(
                    torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                    SqueezeTwice(),
                    make_a_linear(3*feat_dim, feat_dim),
                    torch.nn.ReLU(),
                    make_a_linear(feat_dim, self.amd_action_dim)    
                )
            if self.args.use_early_exit:
                self.early_exit_dict[name] = make_a_linear(self.args.hidden_dim, 1)
                
                

        
        lstm_feat_dim = 2048   #getattr(self.base_model, 'fc').in_features
        if self.args.amd_consensus_type == "lstm":
            self.last_rnn = torch.nn.LSTMCell(input_size=feat_dim, hidden_size=lstm_feat_dim)
            self.new_fc = make_a_linear(feat_dim, self.num_class)

        
        elif self.args.amd_consensus_type == "attention":
            self.att_fc = make_a_linear(feat_dim, 64)
            self.new_fc = make_a_linear(64, self.num_class)
        
        else:
            self.new_fc = make_a_linear(feat_dim_of_res50_block['conv_5'], self.num_class)
            
            
            

    
    def _prepare_pos_encoding(self):
        for name in self.block_rnn_dict.keys() :
            feat_dim = feat_dim_of_res50_block[name]
            self.pos_encoding_dict[name] = PositionalEncoding(feat_dim, dropout=0.1, max_len=16)
            

    def _prep_a_net(self, model_name, shall_pretrain):
        if "efficientnet" in model_name:
            if shall_pretrain:
                model = EfficientNet.from_pretrained(model_name)
            else:
                model = EfficientNet.from_named(model_name)
            model.last_layer_name = "_fc"
        else:
            model = getattr(torchvision.models, model_name)(shall_pretrain)
            if self.is_shift:
                print('Adding temporal shift...')
                from ops.temporal_shift import make_temporal_shift
                make_temporal_shift(model, self.num_segments,
                                n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)

            if "resnet" in model_name:
                model.last_layer_name = 'fc'
            elif "mobilenet_v2" in model_name:
                model.last_layer_name = 'classifier'
        return model


    def _prepare_base_model(self, base_model):
        self.input_size = 224
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]
        if self.args.ada_reso_skip:
            shall_pretrain = len(self.args.model_paths) == 0 or self.args.model_paths[0].lower() != 'none'
            for bbi, backbone_name in enumerate(self.args.backbone_list):
                model = self._prep_a_net(backbone_name, shall_pretrain)
                self.base_model_list.append(model)
        elif self.args.ada_depth_skip:
            shall_pretrain = len(self.args.model_paths) == 0 or self.args.model_paths[0].lower() != 'none'
            self.base_model = self._prep_a_net(base_model, shall_pretrain)
        else:
            self.base_model = self._prep_a_net(base_model, self.pretrain == 'imagenet')

 
    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN_Amd, self).train(mode)
        if self._enable_pbn and mode:
            print("Freezing BatchNorm2D except the first one.")
            if self.args.ada_reso_skip:
                models = [self.lite_backbone]
                if self.multi_models:
                    models = models + self.base_model_list
            else:
                models = [self.base_model]

            for the_model in models:
                count = 0
                bn_scale = 1
                for m in the_model.modules():
                    if isinstance(m, nn.BatchNorm2d):  # TODO(yue)
                        count += 1
                        if count >= (2 * bn_scale if self._enable_pbn else bn_scale):
                            m.eval()
                            # shutdown update in frozen mode
                            m.weight.requires_grad = False
                            m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []
        custom_ops = []

        conv_cnt = 0
        bn_cnt = 0
        for n, m in self.named_modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                if self.fc_lr5:
                    lr5_weight.append(ps[0])
                else:
                    normal_weight.append(ps[0])
                if len(ps) == 2:
                    if self.fc_lr5:
                        lr10_bias.append(ps[1])
                    else:
                        normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.LayerNorm):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))

            elif isinstance(m, torch.nn.LSTMCell):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                normal_weight.append(ps[1])
                normal_bias.append(ps[2])
                normal_bias.append(ps[3])

            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {} {}. Need to give it a learning policy".format(n, type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
            {'params': custom_ops, 'lr_mult': 1, 'decay_mult': 1,
             'name': "custom_ops"},
            # for fc
            {'params': lr5_weight, 'lr_mult': 5, 'decay_mult': 1,
             'name': "lr5_weight"},
            {'params': lr10_bias, 'lr_mult': 10, 'decay_mult': 0,
             'name': "lr10_bias"},
        ]

  
    
    def block_cnn_backbone(self, name, input_data, the_base_model, signal=-1, indices_list=[], boost=False, b_t_c=False, **kwargs):
        if name is 'base':
            _b, _tc, _h, _w = input_data.shape  # TODO(yue) input (B, T*C, H, W)
            _t, _c = _tc // 3, 3

            if b_t_c:
                input_b_t_c = input_data.view(_b, _t, _c, _h, _w)
            else:
                input_2d = input_data.view(_b * _t, _c, _h, _w)  

            if b_t_c:
                feat = the_base_model(input_b_t_c, signal=signal, **kwargs)
            else:
                feat = the_base_model(input_2d)
        else:
            feat = the_base_model(input_data)

        return feat
    
    def block_fc_backbone(self, name, input_data, new_fc, signal=-1, indices_list=[], boost=False, b_t_c=False,
                 **kwargs):
        
        feat = input_data
        _bt = input_data.shape[0]
        _b, _t = _bt // self.time_steps, self.time_steps
        
        if b_t_c:
            _base_out = new_fc(feat.view(_b * _t, -1)).view(_b, _t, -1)
        else:
            _base_out = new_fc(feat).view(_b, _t, -1)
       
        return _base_out


        
    """
    input: B, T*C, H, W 
    output: B, T*C, H', W'
    ex: depend on conv_name
    """
    def pass_cnn_block(self, name, input_data):
        if self.args.amd_freeze_backbone:
            with torch.no_grad():
                return self.block_cnn_backbone(name, input_data, self.block_cnn_dict[name]) 

        return self.block_cnn_backbone(name, input_data, self.block_cnn_dict[name]) 
                    
    def gate_fc_rnn_block_full(self, input_data_dict, tau):
        
        all_r_list = []
        hx_list = []
        raw_r_list = []

        
        sup_return = None
        sup2_return = None
        # input_data = input_data.detach()

        base_out_dict = {}
        for name in self.block_rnn_dict:
            input_data = input_data_dict[name]
            base_out_dict[name] = self.block_fc_backbone(name, input_data, self.block_fc_dict[name])
        batch_size = base_out_dict[name].shape[0]
        
        hx_l_t = init_hidden(batch_size, self.args.hidden_dim).unsqueeze(1).repeat(1, len(self.block_rnn_dict.keys()), 1) 
        cx_l_t = init_hidden(batch_size, self.args.hidden_dim).unsqueeze(1).repeat(1, len(self.block_rnn_dict.keys()), 1)
        num_of_policy = len(self.block_rnn_dict.keys())
        per_time_r_list = []
        
        raw_hx_list = []
        exit_r_list = []
        for t in range(self.time_steps):
            old_r_t = torch.cat([torch.zeros(batch_size, 1), torch.ones(batch_size, 1)], 1).cuda()
            local_hx_list = []
            local_cx_list = []
            per_time_r_list.append(old_r_t)
            for i, name in enumerate(self.block_rnn_dict.keys()):
                _bt, _c, _h, _w = input_data_dict[name].shape
                _b, _t = _bt // self.time_steps, self.time_steps

                rnn_input = base_out_dict[name][:, t]
                hx = hx_l_t[:,i]
                cx = cx_l_t[:,i]
#                 hx = self.block_rnn2fc_dict[name](rnn_input)
#                 hx = torch.max(torch.stack([hx, self.block_rnn2fc_dict[name](rnn_input)], dim=2), dim=2)
                hx, cx = self.block_rnn_dict[name](rnn_input, (hx, cx))


                feat_t = hx
                p_t = torch.log(F.softmax(self.action_fc_dict[name](feat_t), dim=1).clamp(min=1e-8))
                r_t = torch.cat(
                    [F.gumbel_softmax(p_t[b_i:b_i + 1], tau, True) for b_i in range(p_t.shape[0])])

                if self.args.use_conf_btw_blocks or self.args.use_local_policy_module:
                    raw_r_list.append(r_t)
                    raw_hx_list.append(hx)

                if self.args.use_early_exit:
                    input_feat_t = feat_t.detach().clone()
                    exit_r = F.sigmoid(self.early_exit_dict[name](input_feat_t)).clamp(min=1e-8)
                    exit_r_list.append(exit_r)

                take_bool =  old_r_t[:,-1].unsqueeze(-1) > 0.5
                take_old_ = old_r_t[:,-2].unsqueeze(-1)
                take_curr_ = old_r_t[:,-1].unsqueeze(-1)
                take_old = torch.tensor(~take_bool, dtype=torch.float).cuda()
                take_curr = torch.tensor(take_bool, dtype=torch.float).cuda()
                r_t = old_r_t * take_old + r_t * take_curr
                old_r_t = r_t
#                 _r_t = old_r_t * take_old + r_t * take_curr
#                 old_r_t = _r_t
    
                per_time_r_list.append(r_t)  
                local_hx_list.append(hx)
                local_cx_list.append(cx)
                
            hx_l_t = take_old.unsqueeze(-1) * hx_l_t + take_curr.unsqueeze(-1) * torch.stack(local_hx_list, dim=1)
            cx_l_t = take_old.unsqueeze(-1) * cx_l_t + take_curr.unsqueeze(-1) * torch.stack(local_cx_list, dim=1)
            
        r_list = torch.stack(per_time_r_list, dim=1).view(_b, _t, num_of_policy+1, -1) #TK * (B, 2) -> B,T*K,2 -> B, T, K, 2                                                               
                                                                                    
        if self.args.use_conf_btw_blocks or self.args.use_local_policy_module:
            sup_return = torch.stack(raw_hx_list, dim=1).view(_b, _t, num_of_policy, -1) #TK*(B,feat) -> B,T*K,feat -> B, T, K, feat
            sup2_return = torch.stack(raw_r_list, dim=1).view(_b, _t, num_of_policy, -1) #TK*(B,2) -> B,T*K,2 -> B, T, K, 2
        
        if self.args.use_early_exit:
            exit_r_t = torch.stack(exit_r_list, dim=0).view(_b, _t, num_of_policy)
            return r_list, sup_return, sup2_return, exit_r_t
               
        return r_list, sup_return, sup2_return, None

    
    
    def pass_last_fc_block(self, name, input_data):
        if self.args.amd_freeze_backbone:
            with torch.no_grad():
                avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
                input_data = avgpool(input_data)
                input_data = torch.nn.Dropout(p=self.dropout)(input_data).squeeze(-1).squeeze(-1)
                return self.block_fc_backbone(name, input_data, self.new_fc) 
        
        avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        input_data = avgpool(input_data)
        input_data = torch.nn.Dropout(p=self.dropout)(input_data).squeeze(-1).squeeze(-1)
        return self.block_fc_backbone(name, input_data, self.new_fc)
                    
    
    def pass_last_rnn_block(self, name, input_data, candidate):
        avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))

        input_data = avgpool(input_data)
        input_data = torch.nn.Dropout(p=self.dropout)(input_data).squeeze(-1).squeeze(-1)
        
        
        _bt = input_data.shape[0]
        _b, _t = _bt // self.time_steps, self.time_steps
        _input_data = input_data.view(_b, _t, -1)
        feat_dim = _input_data.shape[-1]

        hx = init_hidden(_b, feat_dim)
        cx = init_hidden(_b, feat_dim)
        
        for t in range(self.time_steps):
            _rnn_input = _input_data[:, t] * candidate[:,t,-1].unsqueeze(-1)
            hx, cx = self.last_rnn(_rnn_input, (hx, cx))

        return self.new_fc(hx)
                   
    def pass_pred_block(self, name, hx_l):
        feat = hx_l # B,T,512
        _b, _t = feat.shape[0], feat.shape[1]
       
        return self.block_pred_rnn_fc_dict[name](feat.view(_b*_t, -1)).view(_b, _t, -1)

        
    
    def using_online_policy(self):
        if any([self.args.offline_lstm_all, self.args.offline_lstm_last]):
            return False
        elif any([self.args.random_policy, self.args.all_policy]):
            return False
        elif self.args.real_scsampler:
            return False
        else:
            return True

    def input_fusion(self, input_data, r):
        # TODO data: B * TC * H * W
        # TODO r   : B * T * T
        _b, _tc, _h, _w = input_data.shape
        _c = _tc // self.args.num_segments
        fuse_data_list = []

        for bi in range(_b):
            if self.args.identity_prior:
                prior = torch.eye(self.args.num_segments).to(input_data.device)
            else:
                prior = 0
            if self.args.lower_mask:
                mask = torch.tril(torch.ones(self.args.num_segments, self.args.num_segments)).to(input_data.device)
            else:
                mask = 1
            real_r = (r[bi] + prior) * mask
            if self.args.direct_lower_mask:
                real_r = torch.tril(real_r)
            if self.args.row_normalization:
                real_r = real_r / (real_r.sum(dim=1, keepdim=True).clamp_min(1e-6))
            fused_data = torch.matmul(real_r, input_data[bi].view(self.args.num_segments, _c * _h * _w))
            fuse_data_list.append(fused_data)
        return torch.stack(fuse_data_list, dim=0).view(_b, _tc, _h, _w)

   
    def late_fusion(self, base_out_list, in_matrix, out_matrix):
        return base_out_list

    def forward(self, *argv, **kwargs): 
        input_list = kwargs["input"]
        batch_size = input_list[0].shape[0]  # TODO(yue) input[0] B*(TC)*H*W
        _input = input_list[0]
        if "epoch" in kwargs:
            self.epoch = kwargs["epoch"]
        else:
            self.epoch=0

        if self.args.skip_twice:
            candidate_list = torch.cat([torch.zeros(batch_size, self.time_steps, 1), torch.zeros(batch_size, self.time_steps, 1), torch.ones(batch_size, self.time_steps, 1)], 2) #B, T, A
        else:
            candidate_list = torch.cat([torch.zeros(batch_size, self.time_steps, 1), torch.ones(batch_size, self.time_steps, 1)], 2).cuda() #B, T, A

        candidate_log_list = []
        all_policy_result_list = []
        skip_result_list = []
        all_dual_policy_result_list = []
        all_similarity_list = []
        block_out_list = []
        take_bool = candidate_list[:,:,-1] > 0.5
        candidate_log_list.append(torch.tensor(take_bool, dtype=torch.float).cuda())
        
        if self.args.skip_twice:
            take_bool = candidate_list > 0.5
            all_policy_result_list.append(torch.tensor(take_bool, dtype=torch.float).cuda())
            
        if "tau" not in kwargs:
            kwargs["tau"] = None
        tau = kwargs["tau"]
        feat_dict = {}
        for name in self.block_cnn_dict.keys():
            # input image tensor with 224 size
            _input = self.pass_cnn_block(name, _input)
            feat_dict[name] = _input
        
        #B,T,K,2/  B,T,K,feat/ B,T,K,2
        r_l_t, hx_l_t, all_policy_result_l_t, exit_r_t = self.gate_fc_rnn_block_full(feat_dict, tau) 
                
        if self.args.use_conf_btw_blocks:
            for i, name in enumerate(self.block_rnn_list):
                block_out_list.append(self.pass_pred_block(name, hx_l_t[:,:,i,:]))

            
        return_supp = None
        if self.args.amd_consensus_type == "avg":
            last_feat = feat_dict[list(self.block_cnn_dict.keys())[-1]]
            if self.args.use_conf_btw_blocks:
                block_out = self.pass_last_fc_block('new_fc', last_feat)
                block_out_list.append(block_out)

            else:
                block_out = self.pass_last_fc_block('new_fc', last_feat)
            
#             if self.args.use_early_stop_inf and not self.training :
#                 modify_candidate_list = []
#                 selected_bool = r_l_t[:,:,-1,-1].unsqueeze(-1) > 0.5 #B, T
#                 selected_block_outs = block_out * torch.tensor(selected_bool, dtype=torch.float).cuda()
#                 num_class = block_out.shape[-1]

#                 early_stop_thr = 0.999
#                 for b_i in range(batch_size):
#                     max_i = self.time_steps
#                     early_stop_limit = 1
#                     avg_block_out = 0
#                     selected_frame_cnt = torch.tensor(0.0, dtype=torch.float)
#                     output_class_dict = {}
#                     output_val = 0.0
#                     boundary_start = False
#                     for t_i in range(self.time_steps):
#                         avg_block_out += selected_block_outs[b_i, t_i, :]
#                         is_selected = r_l_t[b_i,t_i,-1,-1] > 0.5
#                         if is_selected: 
#                             selected_frame_cnt +=1
#                             output_base_out = F.softmax(avg_block_out/selected_frame_cnt, dim=-1)

#                             candidate_output_class = output_base_out.max(dim=0)[1].cpu().item()
#                             candidate_output_val = output_base_out.max(dim=0)[0].cpu()
# #                             if candidate_output_val > early_stop_thr:
# #                                 boundary_start = True


#                             if candidate_output_val > early_stop_thr:
#                                 if candidate_output_class in output_class_dict:
#                                     output_class_dict[candidate_output_class] -= 1
#                                     if output_class_dict[candidate_output_class] is 0:
#                                         max_i = (t_i+1)
#                                         break                                    
#                                 else:
#                                     output_class_dict[candidate_output_class] = early_stop_limit-1
#                                     max_i = (t_i+1)
#                                     break  
# #                         else:
# #                             if boundary_start:
# #                                 max_i = (t_i+1)
# #                                 break
#                     stage_cnt= r_l_t.shape[2]
#                     modify_candidate_list.append(torch.cat((torch.ones(max_i, stage_cnt), torch.zeros(self.time_steps-max_i, stage_cnt)), dim=0).cuda()) #B * K,1

#                 modify_candidate_l_t = torch.stack(modify_candidate_list, dim=0) # B, T, K
#                 r_l_t = modify_candidate_l_t.unsqueeze(-1) * r_l_t
                
            if self.args.use_early_stop_inf and not self.training :
                modify_candidate_list = []
                selected_bool = r_l_t[:,:,-1,-1].unsqueeze(-1) > 0.5 #B, T
                selected_block_outs = block_out * torch.tensor(selected_bool, dtype=torch.float).cuda()
                num_class = block_out.shape[-1]

                early_stop_thr = 0.999
                for b_i in range(batch_size):
                    max_i = self.time_steps
                    early_stop_limit = 8
                    avg_block_out = 0
                    selected_frame_cnt = torch.tensor(0.0, dtype=torch.float)
                    output_class_dict = {}
                    output_val = 0.0
                    boundary_start = False
                    for t_i in range(self.time_steps):
                        avg_block_out += selected_block_outs[b_i, t_i, :]
                        is_selected = r_l_t[b_i,t_i,-1,-1] > 0.5
                        
                        
                        if is_selected:
                            selected_frame_cnt +=1
                            output_base_out = F.softmax(avg_block_out/selected_frame_cnt, dim=-1)
                            candidate_output_val = output_base_out.max(dim=0)[0].cpu()

                            if (t_i > early_stop_limit-1) and (candidate_output_val > early_stop_thr):
                                max_i = (t_i+1)
                                break                                    

                    stage_cnt= r_l_t.shape[2]
                    modify_candidate_list.append(torch.cat((torch.ones(max_i, stage_cnt), torch.zeros(self.time_steps-max_i, stage_cnt)), dim=0).cuda()) #B * K,1

                modify_candidate_l_t = torch.stack(modify_candidate_list, dim=0) # B, T, K
                r_l_t = modify_candidate_l_t.unsqueeze(-1) * r_l_t
                
#             self.args.use_early_exit_inf = True
            if self.args.use_early_exit_inf and not self.training:
                exit_flag_bool = exit_r_t > 0.6
                exit_flag_t = torch.tensor(exit_flag_bool, dtype=torch.float).cuda()
#                 pdb.set_trace()
#                 pdb.set_trace()
#                 skipped_layer_bool_prev = r_l_t[:,:,:-1,-1] < 0.5
#                 skipped_layer_bool_curr = r_l_t[:,:,1:,-1] < 0.5
#                 skip_l_b_prev = torch.tensor(skipped_layer_bool_prev, dtype=torch.float).cuda()
#                 skip_l_b_curr = torch.tensor(skipped_layer_bool_curr, dtype=torch.float).cuda()
#                 skip_flag_t = skip_l_b_curr - skip_l_b_prev
#                 terminate_flag_t = exit_flag_t * skip_flag_t
                
                passed_layer_bool = r_l_t[:,:,1:,-1] > 0.5
                passed_l_b = torch.tensor(passed_layer_bool, dtype=torch.float).cuda()
                terminate_flag_t = passed_l_b * exit_flag_t

#                 skipped_layer_bool = r_l_t[:,:,-1,-1] < 0.5
#                 skip_l_b = torch.tensor(skipped_layer_bool, dtype=torch.float).cuda()
#                 terminate_flag_t = exit_flag_t[:,:,-1] * skip_l_b
                stage_cnt= r_l_t.shape[2]

                modify_candidate_list = []
                for b_i in range(batch_size):
                    max_i = self.time_steps
                    for t_i in range(self.time_steps):
                        if terminate_flag_t[b_i, t_i,-1] == 1:
                            max_i = (t_i+1)
                            break  
#                     pdb.set_trace()
       
                    modify_candidate_list.append(torch.cat((torch.ones(max_i, stage_cnt), torch.zeros(self.time_steps-max_i, stage_cnt)), dim=0).cuda()) #B * K,1
#                 pdb.set_trace()
        
#                 modify_candidate_list = []
#                 for b_i in range(batch_size):
#                     selected_exist = False
#                     max_i = self.time_steps
#                     for t_i in range(self.time_steps):
#                         if r_l_t[b_i,t_i,-1,-1] == 1:
#                             selected_exist = True
#                             continue
                        
#                         if selected_exist and terminate_flag_t[b_i, t_i,-1] == 1:
#                             max_i = (t_i+1)
#                             break  
                            
#                     modify_candidate_list.append(torch.cat((torch.ones(max_i, stage_cnt), torch.zeros(self.time_steps-max_i, stage_cnt)), dim=0).cuda()) #B * K,1
#                 pdb.set_trace()

                modify_candidate_l_t = torch.stack(modify_candidate_list, dim=0) # B, T, K
                r_l_t = modify_candidate_l_t.unsqueeze(-1) * r_l_t
              
            output = self.amd_combine_logits(r_l_t[:,:,:,-1], block_out)
#             output = self.fixed_amd_combine_logits(r_l_t, block_out)
            
#             if not self.training:
#                 accum_r_l = []
#                 _old_r_t = r_l_t[:, :, 0, :]
#                 accum_r_l.append(_old_r_t)
#                 _b, _t, _k, _a = r_l_t.shape #B, T, K, 2
#                 for k_i in range(1, _k):
#                     _curr_r_t = r_l_t[:,:,k_i,:]
#                     take_bool = _old_r_t[:,:,-1].unsqueeze(-1) >0.5
#                     take_curr = torch.tensor(take_bool, dtype=torch.float).cuda()
#                     take_old = torch.tensor(~take_bool, dtype=torch.float).cuda()

#                     _old_r_t = take_old * _old_r_t + take_curr * _curr_r_t
#                     accum_r_l.append(_old_r_t)

#                 r_l_t = torch.stack(accum_r_l, dim=2) #B, T, K, 2


            return_supp = block_out
            
            
        elif self.args.amd_consensus_type == "random_avg":
            block_out = self.pass_last_fc_block('new_fc', feat_dict[list(self.block_cnn_dict.keys())[-1]])
            
            r_all = torch.zeros(batch_size, self.time_steps, self.amd_action_dim).cuda()
            for i_bs in range(batch_size):
                for i_t in range(self.time_steps):
                    rand = 1 if torch.randint(100, [1]) < self.args.random_ratio else 0
                    r_all[i_bs, i_t, rand] = 1.0
            
            output = self.amd_combine_logits(r_all[:,:,-1], block_out)
            candidate_log_list = r_all[:,:,-1].unsqueeze(-1).expand(-1,-1,6)
            return output.squeeze(1), candidate_log_list, None, None, block_out
        
        elif self.args.amd_consensus_type == "lstm":
            block_out = None
            output = self.pass_last_rnn_block('new_fc', feat_dict[list(self.block_cnn_dict.keys())[-1]], r_l_t[:,:,-1,-1])
        
        if self.args.use_conf_btw_blocks or self.args.use_early_stop :
            return output.squeeze(1), r_l_t[:,:,:,-1], all_policy_result_l_t, torch.stack(block_out_list, dim=2), return_supp, exit_r_t
        else:
            return output.squeeze(1), r_l_t[:,:,:,-1], None, None, block_out, exit_r_t
            
    def amd_distil_combine_logits(self, r_l, base_out_l):
        # r_l        B, T, A, 
        # base_out   B, T, K, #class
        
        batch_size = base_out_l.shape[0]
        pred_tensor = base_out_l
        r_tensor = r_l.unsqueeze(-1)
        t_tensor = torch.sum(r_l, dim=[1, 2]).unsqueeze(-1).clamp(1)  # TODO sum T, K to count frame
        return (pred_tensor * r_tensor).sum(dim=[1, 2]) / t_tensor

    def amd_combine_logits(self, r_, base_out):
        # TODO r         N, T 
        # TODO base_out  N, T, C
        
        # voter_list N, T, 1
        batch_size = base_out.shape[0]
        pred_tensor = base_out
        
        '''
        b_, t_, c_ = r_.shape
        selected = (r_[:,:,-1] > 0).float()
        selected = selected.unsqueeze(-1).expand(b_, t_, c_)
        
        r_ = selected * r_
        r_tensor = torch.sum(r_, dim=[2]).unsqueeze(-1)/float(c_)
        '''
        
        _r = r_[:,:,-1]
        r_tensor = _r.unsqueeze(-1)
        t_tensor = torch.sum(_r, dim=[1]).unsqueeze(-1).clamp(1)  # TODO sum T, K to count frame
        
        return (pred_tensor * r_tensor).sum(dim=[1]) / t_tensor
    
    
    def fixed_amd_combine_logits(self, r, base_out):
        # TODO r         B, T, K, 2
        # TODO base_out  B, T, #class
        
        batch_size = base_out.shape[0]
        pred_tensor = base_out

        _r = r
        if self.training:
            accum_r = _r.prod(dim=2)[:,:,-1]
            
            if self.args.use_stoch_select:
                xor_dummy = (torch.rand(accum_r.shape).cuda() < (1/16)).float()
                accum_r = accum_r *(1-xor_dummy) + (xor_dummy) * (1-accum_r)

           
            r_tensor = accum_r.unsqueeze(-1) #B, T, 1
            t_tensor = torch.sum(r_tensor, dim=[1, 2]).unsqueeze(-1).clamp(1)  # TODO sum T, K to count frame
            
        else:
            selected = torch.sum(_r[:,:,:,-1], dim=[2]) == ( len(self.args.block_rnn_list)+1 ) #B, T
            r_tensor = torch.tensor(selected, dtype=torch.float).unsqueeze(-1).cuda() #B, T, 1
            t_tensor = torch.sum(r_tensor, dim=[1, 2]).unsqueeze(-1).clamp(1)  # TODO sum T, K to count frame
            
        
        return (pred_tensor * r_tensor).sum(dim=[1]) / t_tensor

        
        
    def combine_logits(self, r, base_out_list, ind_list):
        # TODO r                N, T, K
        # TODO base_out_list  < K * (N, T, C)
        pred_tensor = torch.stack(base_out_list, dim=2)
        r_tensor = r[:, :, :self.reso_dim].unsqueeze(-1)
        t_tensor = torch.sum(r[:, :, :self.reso_dim], dim=[1, 2]).unsqueeze(-1).clamp(1)  # TODO sum T, K to count frame
        return (pred_tensor * r_tensor).sum(dim=[1, 2]) / t_tensor

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self, flip=True):
        if flip:
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
        else:
            print('#' * 20, 'NO FLIP!!!')
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66])])
        
    
