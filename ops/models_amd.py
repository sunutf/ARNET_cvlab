from torch import nn
from ops.basic_ops import ConsensusModule
from ops.transforms import *
from torch.nn.init import normal_, constant_
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
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
            if self.args.use_distil_loss_to_rnn or self.args.use_conf_btw_blocks:
                self.block_pred_rnn_fc_dict = nn.ModuleDict()
            elif self.args.use_distil_loss_to_cnn:
                self.block_pred_cnn_fc_dict = nn.ModuleDict()
                
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
        
        for name in self.block_cnn_dict.keys():
#             if name is not 'base':
            feat_dim = feat_dim_of_res50_block[name]
            self.block_fc_dict[name] = torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                SqueezeTwice(),
                make_a_linear(feat_dim, feat_dim)
            )
            if self.args.diff_to_rnn:
                feat_dim = feat_dim*2

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
                self.vote_fc_dict[name] =  make_a_linear(self.args.hidden_dim, self.vote_dim)
                
            if self.args.use_local_policy_module:
                self.local_policy_dict[name] = torch.nn.Sequential(
                    torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                    SqueezeTwice(),
                    make_a_linear(3*feat_dim, feat_dim),
                    torch.nn.ReLU(),
                    make_a_linear(feat_dim, self.amd_action_dim)    
                )
                
                

        
        lstm_feat_dim = 2048   #getattr(self.base_model, 'fc').in_features
        if self.args.amd_consensus_type == "lstm":
            self.last_rnn = torch.nn.LSTMCell(input_size=feat_dim, hidden_size=lstm_feat_dim)
            self.new_fc = make_a_linear(feat_dim, self.num_class)

        
        elif self.args.amd_consensus_type == "attention":
            self.att_fc = make_a_linear(feat_dim, 64)
            self.new_fc = make_a_linear(64, self.num_class)
        
        else:
            self.new_fc = make_a_linear(feat_dim, self.num_class)

    
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
                    
    def gate_fc_rnn_block(self, name, input_data, candidate_list, tau, voter_stack = None):
        
        r_list = []
        voter_list = []
        hx_list = []
        raw_r_list = []
        redundant_r_list = []
        
        sup_return = None
        sup2_return = None
        if name in self.block_rnn_dict.keys(): # gate activate = policy on 
            base_out = self.block_fc_backbone(name, input_data, self.block_fc_dict[name])
            if self.args.pe_at_rnn:
                base_out = self.pos_encoding_dict[name](base_out)
            
            old_hx = None
            batch_size = base_out.shape[0]
            hx = init_hidden(batch_size, self.args.hidden_dim)
            cx = init_hidden(batch_size, self.args.hidden_dim)
            
            store_recent_pass_out = torch.zeros(batch_size, base_out.shape[-1], dtype=torch.long).cuda() 
            store_recent_pass = torch.zeros(batch_size, 1, dtype=torch.long).cuda()
            
            prev_base_out = torch.zeros(batch_size, base_out.shape[-1], dtype=torch.long).cuda()

            for t in range(self.time_steps):
                old_r_t = candidate_list[:, t, :].cuda() #B, K
                
                if t !=0 and self.args.skip_twice:
                    take_bool =  r_t[:,1].unsqueeze(-1) > 0.5 #skip twice
                    take_old_r = torch.tensor(~take_bool, dtype=torch.float).cuda()
                    old_r_t = old_r_t * take_old_r 
                    

                if self.args.frame_independent:
                    feat_t = base_out[:, t]
                else:
                    rnn_input = base_out[:, t]
                    if self.args.diff_to_rnn:
                        
                        recent_base_out = torch.cat([base_out[b_i, store_recent_pass[b_i][-1]].unsqueeze(0) for b_i in range(batch_size)], dim=0)
                        rnn_input = torch.cat((rnn_input, (rnn_input - recent_base_out)), dim = 1)
                    
                    elif self.args.diff_to_policy:
                        cossim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
                        relu = torch.nn.ReLU()
                        
                        cossim_outs = cossim(rnn_input, store_recent_pass_out).unsqueeze(-1)
                        skip_action_outs = relu(cossim_outs)
                        pass_action_outs = 1 - skip_action_outs
                        diff_action_outs = skip_action_outs * torch.cat((skip_action_outs, pass_action_outs), dim=1)
                        

                        hx, cx = self.block_rnn_dict[name](rnn_input, (hx, cx))
                        feat_t = hx
                        p_t = torch.log(F.softmax(self.action_fc_dict[name](feat_t)+diff_action_outs, dim=1).clamp(min=1e-8))
                        r_t = torch.cat(
                            [F.gumbel_softmax(p_t[b_i:b_i + 1], tau, True) for b_i in range(p_t.shape[0])])
                    
                    elif self.args.strong_skip_sim:
                        cossim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
                        relu = torch.nn.ReLU()
                    
                        cossim_outs = cossim(rnn_input, store_recent_pass_out).unsqueeze(-1)
                        skip_action_outs = relu(cossim_outs)
                        
                        action_out_bool = skip_action_outs > 0.98 
                        hard_skip_action = torch.tensor(action_out_bool, dtype=torch.float).cuda()
                        depend_policy_action = torch.tensor(~action_out_bool, dtype=torch.float).cuda()
                        
                        hx, cx = self.block_rnn_dict[name](rnn_input, (hx, cx))
                        feat_t = hx
                        p_t = torch.log(F.softmax(self.action_fc_dict[name](feat_t), dim=1).clamp(min=1e-8))
                        r_t = torch.cat(
                            [F.gumbel_softmax(p_t[b_i:b_i + 1], tau, True) for b_i in range(p_t.shape[0])])
                    
                        skip_r_t = torch.cat([torch.ones(p_t.shape[0]).unsqueeze(-1), torch.zeros(p_t.shape[0]).unsqueeze(-1)], dim=1).cuda() 
                        
                        r_t = hard_skip_action * skip_r_t + depend_policy_action * r_t
                                                
                    elif self.args.use_local_policy_module:
                        local_compare_input = torch.cat((prev_base_out, rnn_input, (rnn_input - prev_base_out)), dim = 1)
                        redundant_r = F.softmax(local_policy_dict[name], dim=1).clamp(min=1e-8)
                        
                        hx, cx = self.block_rnn_dict[name](rnn_input, (hx, cx))
                        feat_t = hx
                        noisy_r = F.softmax(self.action_fc_dict[name](feat_t), dim=1).clamp(min=1e-8)
                        
                        p_t = torch.log(redundant_r * noisy_r)
                        r_t = torch.cat(
                            [F.gumbel_softmax(p_t[b_i:b_i + 1], tau, True) for b_i in range(p_t.shape[0])])
                        
                        redundant_r_list.append(redundant_r)
                        noisy_r_list.append(noisy_r)
                        
                        prev_base_out = rnn_input


                    else:
                        hx, cx = self.block_rnn_dict[name](rnn_input, (hx, cx))
                        feat_t = hx
                        p_t = torch.log(F.softmax(self.action_fc_dict[name](feat_t), dim=1).clamp(min=1e-8))
                        r_t = torch.cat(
                            [F.gumbel_softmax(p_t[b_i:b_i + 1], tau, True) for b_i in range(p_t.shape[0])])
                        
                       
                    
#                     if self.args.voting_policy:
#                         passed_in_prev_stage = old_r_t > 0.5
#                         passed_in_this_stage = r_t[:, 1].unsqueeze(-1).cuda() > 0.5
#                         passed_in_prev_stage = torch.tensor(~passed_in_prev_stage, dtype=torch.float).cuda() #trick pass ->0
#                         failed_in_this_stage = torch.tensor(passed_in_this_stage, dtype=torch.float).cuda() #fail -> 0
                        
#                         voter = (passed_in_prev_stage + failed_in_this_stage) >0.5 # prev pass :0 + curr skip:0 -> 0 ~ -> 1(voter)
#                         voter = torch.tensor(~voter, dtype=torch.float).cuda()
                        
#                         v_t = voter * feat_t
#                         p_v_t = torch.log(F.softmax(self.vote_fc_dict[name](v_t), dim=1).clamp(min=1e-8))
#                         r_v_t = torch.cat([F.gumbel_softmax(p_v_t[b_i:b_i + 1], tau, True) for b_i in range(p_t.shape[0])])
                        
#                         voting_result = r_v_t[:,1].unsqueeze(-1)
#                         voter_list.append(voter * voting_result)
                        
                    if self.args.use_conf_btw_blocks:
                        raw_r_list.append(r_t)
                    
                    
#                     take_old = old_r_t[:,-2].unsqueeze(-1)
#                     take_curr = old_r_t[:,-1].unsqueeze(-1)
                    take_bool =  old_r_t[:,-1].unsqueeze(-1) > 0.5
                    take_old_ = torch.tensor(~take_bool, dtype=torch.float).cuda()
                    take_curr_ = torch.tensor(take_bool, dtype=torch.float).cuda()
                    r_t = old_r_t * take_old_ + r_t * take_curr_
                                        
                    
                    check_to_store_bool = r_t[:,-1].unsqueeze(-1)>0.5
                    check_to_store = torch.tensor(check_to_store_bool, dtype=torch.float).cuda()
                    store_recent_pass_out = rnn_input * check_to_store
                    r_list.append(r_t)  
                                                      
                    if old_hx is not None:
                        hx = old_hx * take_old_ + hx * take_curr_

                    hx_list.append(hx)
                    old_hx = hx
            #check all skip case
#             r_list = torch.stack(r_list, dim=1)
#             _check_empty_candidate = r_list.sum(dim=1)
#             take_bool_r = _check_empty_candidate[:,1] > 0.5
#             take_bool_r = take_bool_r.unsqueeze(-1).repeat(1,2)
#             take_old_r  = torch.tensor(~take_bool_r, dtype=torch.float).cuda()
#             take_curr_r = torch.tensor(take_bool_r, dtype=torch.float).cuda()
            
#             take_old_r = take_old_r.unsqueeze(1).expand(-1,16,-1)
#             take_curr_r = take_curr_r.unsqueeze(1).expand(-1,16,-1)
#             r_list = take_old_r * candidate_list.cuda() + take_curr_r * r_list

            r_list = torch.stack(r_list, dim=1)
            
            if self.args.use_distil_loss_to_rnn:
                sup_return = torch.stack(hx_list, dim=1)
            elif self.args.use_distil_loss_to_cnn:
                sup_return = base_out
            elif self.args.use_conf_btw_blocks:
                sup_return = torch.stack(hx_list, dim=1)
                sup2_return = torch.stack(raw_r_list, dim=1)
               
            if self.args.voting_policy:
                sup_return = voter_stack + torch.stack(voter_list, dim=1)
        
            if self.args.use_local_policy_module:
                redundant_t = torch.stack(redundant_r_list, dim=1) #B, T, 2
                noisy_t     = torch.stack(noisy_r_list, dim=1) #B, T, 2
                duar_r_list = torch.cat(redundant_t.unsqueeze(-2), noisy_t.unsqueeze(-2), dim = -2) #B, T, 2', 2
                
                return r_list, sup_return, sup2_return, dual_r_list, similarity_gt_list
        return r_list, sup_return, sup2_return

    
    
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
#         reverse_bool = torch.randint(0, 2, [1]) > 0# 1(reverse)
        
#         if reverse_bool:
#             _b, _tc, _h, _w = _input.shape  
#             _t, _c = _tc // 3, 3

#             _input = _input.view(_b, _t, _c, _h, _w)
#             _input = torch.flip(_input, [1])
#             _input = _input.view(_b, _tc, _h, _w)

        candidate_list = torch.zeros(batch_size, self.time_steps, 1)
        if self.args.skip_twice:
            candidate_list = torch.cat([torch.zeros(batch_size, self.time_steps, 1), torch.zeros(batch_size, self.time_steps, 1), torch.ones(batch_size, self.time_steps, 1)], 2) #B, T, A
        else:
            candidate_list = torch.cat([torch.zeros(batch_size, self.time_steps, 1), torch.ones(batch_size, self.time_steps, 1)], 2).cuda() #B, T, A

        candidate_log_list = []
        all_policy_result_list = []
        skip_result_list = []
        all_dual_policy_result_list = []
        block_out_list = []
        take_bool = candidate_list[:,:,-1] > 0.5
        candidate_log_list.append(torch.tensor(take_bool, dtype=torch.float).cuda())
        
        if self.args.skip_twice:
            take_bool = candidate_list > 0.5
            all_policy_result_list.append(torch.tensor(take_bool, dtype=torch.float).cuda())
            
        if "tau" not in kwargs:
            kwargs["tau"] = None
        tau = kwargs["tau"]
        voter_list = None
        for name in self.block_cnn_dict.keys():
            # input image tensor with 224 size
            _input = self.pass_cnn_block(name, _input) 

#             if name is not 'base' and name in self.block_rnn_list:
            if name in self.block_rnn_list:
                if self.args.voting_policy:
                    voter_list = torch.zeros(batch_size, self.time_steps, 1, dtype=torch.float).cuda() #B, T, 1
                    _candidate_list, voter_list, _ = self.gate_fc_rnn_block(name, _input, candidate_list, tau, voter_list)

                # update candidate_list based on policy rnn
                elif self.args.use_local_policy_module:
                    _candidate_list, hx_list, raw_r_list, raw_dual_r_list, similarity_gt_list= self.gate_fc_rnn_block(name, _input, candidate_list, tau, None)
                    all_dual_policy_result_list.append(raw_dual_r_list)
                else :
                    _candidate_list, hx_list, raw_r_list = self.gate_fc_rnn_block(name, _input, candidate_list, tau, None)

                if self.args.use_distil_loss_to_rnn :
                    skip_new_list = (candidate_list[:,:,-1] - _candidate_list[:,:,-1]).cuda()
                    skip_result_list.append(skip_new_list)
                    block_out_list.append(self.pass_pred_block(name, hx_list))

                elif self.args.use_distil_loss_to_cnn:
                    base_out = hx_list
                    block_out_list.append(self.block_fc_backbone(name, base_out, self.block_pred_cnn_fc_dict[name]))

                elif self.args.use_conf_btw_blocks:
                    block_out_list.append(self.pass_pred_block(name, hx_list))

                candidate_list = _candidate_list

#                 take_bool = candidate_list[:,:,1] > 0.5
#                 candidate_log_list.append(torch.tensor(take_bool, dtype=torch.float).cuda())

                candidate_log_list.append(candidate_list[:,:,-1])
                if self.args.skip_twice:
                    all_policy_result_list.append(candidate_list)
                if self.args.use_conf_btw_blocks:
                    all_policy_result_list.append(raw_r_list)
        
            
        return_supp = None
        if self.args.amd_consensus_type == "avg":
            if self.args.use_distil_loss_to_rnn or self.args.use_distil_loss_to_cnn:
                skip_result_list.append(candidate_list[:,:,-1].cuda()) 
                skip_r_l = torch.stack(skip_result_list, dim=2) # (B,T) -> B,T,K 
                
                block_out = self.pass_last_fc_block('new_fc', _input)
                block_out_list.append(block_out)
                block_r_l = torch.stack(block_out_list, dim=2) # (B,T,#class) -> (B,T,K,#class)
                
                output = self.amd_distil_combine_logits(skip_r_l, block_r_l)
            elif self.args.use_conf_btw_blocks:
                block_out = self.pass_last_fc_block('new_fc', _input)
                block_out_list.append(block_out)
                output = self.amd_combine_logits(candidate_list[:,:,-1], block_out, voter_list)

            else:
                block_out = self.pass_last_fc_block('new_fc', _input)
                output = self.amd_combine_logits(candidate_list[:,:,-1], block_out, voter_list)
                
            if self.args.use_early_stop:
                e_p_t = torch.log(F.softmax(self.early_stop_decision_block(block_out), dim=2).clamp(min=1e-8))
                e_r_t = torch.cat(
                    [F.gumbel_softmax(e_p_t[b_i:b_i + 1, t_i:t_i+1], tau, True) for b_i in range(e_p_t.shape[0]) for t_i in range(self.time_steps)]) #B*T, 2(kill/pass)
                
                e_r_t = e_r_t.view(e_p_t.shape[0], e_p_t.shape[1], -1) # B,T,2
                selected_bool = candidate_list[:,:,-1].unsqueeze(-1) > 0.5
                choose_selected_es = e_r_t * torch.tensor(selected_bool, dtype=torch.float).cuda()
    
                if not self.training:
                    modify_candidate_list = []
                    for b_i in range(batch_size):
                        max_i = 16
                        for t_i in range(self.time_steps):
                            if choose_selected_es[b_i,t_i,0] == 1:
                                max_i = (t_i+1)
                                break

                        stage_cnt= len(candidate_log_list)
                        modify_candidate_list.append(torch.cat((torch.ones(max_i, stage_cnt), torch.zeros(self.time_steps-max_i, stage_cnt)), dim=0).cuda()) #T,K
                    
                    modify_candidate_l_t = torch.stack(modify_candidate_list, dim=0) # B, T, K
                    candidate_log_list = modify_candidate_l_t * torch.stack(candidate_log_list, dim=2)
                    output = self.amd_combine_logits(candidate_log_list[:,:,-1], block_out, voter_list)
                else:
                    candidate_log_list = torch.stack(candidate_log_list, dim=2)
                return_supp = choose_selected_es

            else:
                if self.args.use_local_policy_module:
                    return_supp = all_dual_policy_result_list
                else:
                    return_supp = block_out
                candidate_log_list = torch.stack(candidate_log_list, dim=2)
            
            
        elif self.args.amd_consensus_type == "random_avg":
            block_out = self.pass_last_fc_block('new_fc', _input)
            
            r_all = torch.zeros(batch_size, self.time_steps, self.amd_action_dim).cuda()
            for i_bs in range(batch_size):
                for i_t in range(self.time_steps):
                    rand = 1 if torch.randint(100, [1]) < self.args.random_ratio else 0
                    r_all[i_bs, i_t, rand] = 1.0
            
            output = self.amd_combine_logits(r_all[:,:,-1], block_out, voter_list)
            candidate_log_list = r_all[:,:,-1].unsqueeze(-1).expand(-1,-1,5)
        
        elif self.args.amd_consensus_type == "lstm":
            block_out = None
            output = self.pass_last_rnn_block('new_fc', _input, candidate_list)
             
#         elif self.args.amd_consensus_type is 'attention':
#             _att_input = _input[:, t] * candidate_list[:,t,-1].unsqueeze(-1)
#             attention = ScaledDotProductAttention(temperature=64 ** 0.5)
            
        if self.args.skip_twice:
            return output.squeeze(1), candidate_log_list, torch.stack(all_policy_result_list, dim=2), torch.stack(block_out_list, dim=2), block_out
        if self.args.use_conf_btw_blocks or self.args.use_early_stop :
            return output.squeeze(1), candidate_log_list, torch.stack(all_policy_result_list, dim=2), torch.stack(block_out_list, dim=2), return_supp
        if self.args.use_local_policy_module:
            return output.squeeze(1), candidate_log_list, torch.stack(all_policy_result_list, dim=2), torch.stack(block_out_list, dim=2), return_supp, similarity_result
       
        else:
            return output.squeeze(1), candidate_log_list, None, None, block_out
            
    def amd_distil_combine_logits(self, r_l, base_out_l):
        # r_l        B, T, K, 
        # base_out   B, T, K, #class
        
        batch_size = base_out_l.shape[0]
        pred_tensor = base_out_l
        r_tensor = r_l.unsqueeze(-1)
        t_tensor = torch.sum(r_l, dim=[1, 2]).unsqueeze(-1).clamp(1)  # TODO sum T, K to count frame
        return (pred_tensor * r_tensor).sum(dim=[1, 2]) / t_tensor

    def amd_combine_logits(self, r, base_out, voter):
        # TODO r         N, T 
        # TODO base_out  N, T, C
        
        # voter_list N, T, 1
        batch_size = base_out.shape[0]
        pred_tensor = base_out
        r_tensor = r.unsqueeze(-1)
        t_tensor = torch.sum(r, dim=[1]).unsqueeze(-1).clamp(1)  # TODO sum T, K to count frame
        
        if self.args.voting_policy:
            hold_idx = 0
            for b_i in range(batch_size):
                for t_i in range(self.time_steps):
                    if r_tensor[b_i, t_i, -1] is 1:
                        hold_idx = t_i
                    if voter[b_i, t_i, :] is 1:
                        r_tensor[b_i, hold_idx, -1] += 1.0
            
            
            t_tensor = t_tensor + torch.sum(voter, dim=[1]).unsqueeze(-1).clamp(1)
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
        
    
