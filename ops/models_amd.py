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
from .transformer import TransformerModel, PositionalEncoding
import pdb

def init_hidden(batch_size, cell_size):
    init_cell = torch.Tensor(batch_size, cell_size).zero_()
    if torch.cuda.is_available():
        init_cell = init_cell.cuda()
    return init_cell

# class FramePositionalEncoding(nn.Module):

#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x + self.pe[:x.size(0)].squeeze
#         return self.dropout(x)
    
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
            
            self.block_rnn_list = self.args.block_rnn_list
            
            self.amd_action_dim = 2 #0 , 1 (skip(0) or pass(1))

#             self.frame_pos_encoder = FramePositionalEncoding(d_model=self.args.hidden_dim, max_len=self.time_steps) 

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
            if name is not 'base':
                feat_dim = feat_dim_of_res50_block[name]
                self.block_fc_dict[name] = torch.nn.Sequential(
                    torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                    SqueezeTwice(),
                    make_a_linear(feat_dim, feat_dim)
                )
                self.block_rnn_dict[name] = torch.nn.LSTMCell(input_size=feat_dim, hidden_size=self.args.hidden_dim)
                self.action_fc_dict[name] = make_a_linear(self.args.hidden_dim, self.amd_action_dim)

        
        feat_dim = 2048   #getattr(self.base_model, 'fc').in_features
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
        return self.block_cnn_backbone(name, input_data, self.block_cnn_dict[name]) 
                    
    def gate_fc_rnn_block(self, name, input_data, candidate_list, tau):
        
        r_list = []
        if name in self.block_rnn_dict.keys(): # gate activate = policy on 
            base_out = self.block_fc_backbone(name, input_data, self.block_fc_dict[name])
            if self.args.pe_at_rnn:
                base_out = self.pos_encoding_dict[name](base_out)
            
            old_hx = None
            batch_size = base_out.shape[0]
            hx = init_hidden(batch_size, self.args.hidden_dim)
            cx = init_hidden(batch_size, self.args.hidden_dim)
            

            for t in range(self.time_steps):
                old_r_t = candidate_list[:, t, 1].unsqueeze(-1).cuda()

                if self.args.frame_independent:
                    feat_t = base_out[:, t]
                else:
                    hx, cx = self.block_rnn_dict[name](base_out[:, t], (hx, cx))
                    feat_t = hx
                    p_t = torch.log(F.softmax(self.action_fc_dict[name](feat_t), dim=1).clamp(min=1e-8))
                    r_t = torch.cat(
                        [F.gumbel_softmax(p_t[b_i:b_i + 1], tau, True) for b_i in range(p_t.shape[0])])
                    
                    take_bool =  old_r_t > 0.5
                    take_old = torch.tensor(~take_bool, dtype=torch.float).cuda()
                    take_curr = torch.tensor(take_bool, dtype=torch.float).cuda()
                    r_t = old_r_t * take_old + r_t * take_curr
                    r_list.append(r_t)  # TODO as decision

                                      
                    if old_hx is not None:
                        hx = old_hx * take_old + hx * take_curr
                    
                    old_hx = hx
            
            r_list = torch.stack(r_list, dim=1)

        return r_list
    
    def pass_last_fc_block(self, name, input_data):
        avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        input_data = avgpool(input_data)
        input_data = torch.nn.Dropout(p=self.dropout)(input_data).squeeze(-1).squeeze(-1)
        return self.block_fc_backbone(name, input_data, self.new_fc) 
                   
    
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
        if not self.args.ada_reso_skip and not self.args.ada_depth_skip:  # TODO simple TSN
            _, base_out = self.backbone(kwargs["input"][0], self.base_model, self.new_fc,
                                        signal=self.args.default_signal)
            output = self.consensus(base_out)
            return output.squeeze(1)

        input_list = kwargs["input"]
        batch_size = input_list[0].shape[0]  # TODO(yue) input[0] B*(TC)*H*W
        _input = input_list[0]
        candidate_list = torch.ones(batch_size, self.time_steps, self.amd_action_dim) #B, T, K
        #candidate_list_log = torch.zeros(batch_size, self.time_steps, 1+len(self.args.block_rnn_list)) #B, T, K'
        candidate_log_list = []
        take_bool = candidate_list[:,:,1] > 0.5
        candidate_log_list.append(torch.tensor(take_bool, dtype=torch.float).cuda())
        if "tau" not in kwargs:
            kwargs["tau"] = None
        tau = kwargs["tau"]
        for name in self.block_cnn_dict.keys():
            # input image tensor with 224 size
            _input = self.pass_cnn_block(name, _input) 

            if name is not 'base':
                # update candidate_list based on policy rnn
                candidate_list = self.gate_fc_rnn_block(name, _input, candidate_list, tau)

                take_bool = candidate_list[:,:,1] > 0.5
                candidate_log_list.append(torch.tensor(take_bool, dtype=torch.float).cuda())

        block_out = self.pass_last_fc_block('new_fc', _input)

        output = self.amd_combine_logits(candidate_list[:,:,1], block_out)
        return output.squeeze(1), torch.stack(candidate_log_list, dim=2), None, block_out

    

    def amd_combine_logits(self, r, base_out):
        # TODO r         N, T 
        # TODO base_out  N, T, C
        pred_tensor = base_out
        r_tensor = r.unsqueeze(-1)
        t_tensor = torch.sum(r, dim=[1]).unsqueeze(-1).clamp(1)  # TODO sum T, K to count frame
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
        
    