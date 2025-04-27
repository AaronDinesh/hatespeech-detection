import torch
from torch import nn
import torch.nn.functional as F

from .layer_utils import calc_mlp_dims, create_act, glorot, zeros, MLP


class TabularFeatCombiner(nn.Module):
    r"""
     Combiner module for combining text features with categorical and numerical features
     The methods of combining, specified by :obj:`tabular_config.combine_feat_method` are shown below.
     :math:`\mathbf{m}` denotes the combined multimodal features,
     :math:`\mathbf{x}` denotes the output text features from the transformer,
     :math:`\mathbf{c}` denotes the categorical features, :math:`\mathbf{t}` denotes the numerical features,
     :math:`h_{\mathbf{\Theta}}` denotes a MLP parameterized by :math:`\Theta`, :math:`W` denotes a weight matrix,
     and :math:`b` denotes a scalar bias

     - **text_only**

         .. math::
             \mathbf{m} = \mathbf{x}

     - **concat**

         .. math::
             \mathbf{m} = \mathbf{x} \, \Vert \, \mathbf{c} \, \Vert \, \mathbf{n}

     - **individual_mlps_on_cat_and_numerical_feats_then_concat**

         .. math::
             \mathbf{m} = \mathbf{x} \, \Vert \, h_{\mathbf{\Theta_c}}( \mathbf{c}) \, \Vert \, h_{\mathbf{\Theta_n}}(\mathbf{n})

     - **attention_on_cat_and_numerical_feats** self attention on the text features

         .. math::
             \mathbf{m} = \alpha_{x,x}\mathbf{W}_x\mathbf{x} + \alpha_{x,c}\mathbf{W}_c\mathbf{c} + \alpha_{x,n}\mathbf{W}_n\mathbf{n}

       where :math:`\mathbf{W}_x` is of shape :obj:`(out_dim, text_feat_dim)`,
       :math:`\mathbf{W}_c` is of shape :obj:`(out_dim, cat_feat_dim)`,
       :math:`\mathbf{W}_n` is of shape :obj:`(out_dim, num_feat_dim)`, and the attention coefficients :math:`\alpha_{i,j}` are computed as

         .. math::
             \alpha_{i,j} =
             \frac{
             \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
             [\mathbf{W}_i\mathbf{x}_i \, \Vert \, \mathbf{W}_j\mathbf{x}_j]
             \right)\right)}
             {\sum_{k \in \{ x, c, n \}}
             \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
             [\mathbf{W}_i\mathbf{x}_i \, \Vert \, \mathbf{W}_k\mathbf{x}_k]
             \right)\right)}.

    Parameters:
        tabular_config (:class:`~multimodal_config.TabularConfig`):
            Tabular model configuration class with all the parameters of the model.

    """

    def __init__(self, tabular_config):
        super().__init__()
        self.combine_feat_method = tabular_config.combine_feat_method
        self.cat_feat_dim = tabular_config.cat_feat_dim
        self.numerical_feat_dim = tabular_config.numerical_feat_dim
        self.num_labels = tabular_config.num_labels
        self.numerical_bn = tabular_config.numerical_bn
        self.categorical_bn = tabular_config.categorical_bn
        self.mlp_act = tabular_config.mlp_act
        self.mlp_dropout = tabular_config.mlp_dropout
        self.mlp_division = tabular_config.mlp_division
        self.text_out_dim = tabular_config.text_feat_dim
        self.tabular_config = tabular_config

        #define the dimension of the combined features when using text only
        if self.combine_feat_method == "text_only":
            self.final_out_dim = self.text_out_dim

        # 7.1: concat approach
        #define the dimension of the combined features when using text, categorial and numerical features
        elif self.combine_feat_method == "concat":
            # self.final_out_dim = ...
            
            ### BEGIN SOLUTION
            # raise NotImplementedError()
            self.final_out_dim = self.text_out_dim + self.cat_feat_dim + self.numerical_feat_dim
            ### END SOLUTION

        # 7.2: individual_mlps_on_cat_and_numerical_feats_then_concat approach
        elif (
            self.combine_feat_method
            == "individual_mlps_on_cat_and_numerical_feats_then_concat"
        ):
            # This combine method requires two things: 
            #   - the definition of combined features, self.final_out_dim,  when using text, categorial and numerical features with an MLP
            #   - the definition of the MLPs to process the categorical and numerical features
            
            # we need to define the  dimension of the combined features, self.final_out_dim,  when using text, categorial and numerical features
            # before defining self.final_out_dim, we need to consider the following:
            # differently from the concat version, the categorical and numerical features will be passed through a MLP and their dimensionality will change
            # hence, we need to compute the dimensionality of categorical features, output_dim_cat, after passing the categorial features
            # through the MLP
            
           
            output_dim_cat = 0
            if self.cat_feat_dim > 0:
                #reminder: mlp_division indicates the ratio of the number of hidden dims in a current layer to the next MLP layer
                #first, we need to define the output_dim_cat. We use the following approach:
                output_dim_cat = max(
                    self.cat_feat_dim // (self.mlp_division // 2),
                    self.numerical_feat_dim,
                )
                #print("output_dim_cat: ", output_dim_cat)
                # 
                # after we defined the MLP output dimension, we have to define the architecture of the MLP (the configuration of the layers in the MLP)
                #
                # to do this, we use the following simple approach that defines the layers of the MLP based on the input and output dimension of the categorical features
                # (or you can define your own approach):
                #   - devide the input dimension by self.mlp_division until the division result is smaller than the output of the final layer in the MLP (output_dim_cat)
                #   - the result of i-th division represents the output dimesion of the i-th layer and the input dimension of the i+1-th layer
                # i.e. suppose we have cat_feat_dim = 1238, output_dim_cat = 200, and self.mlp_division = 2,  we obtain the dims =  [619, 309, 154]
                # we do not consider dimension 154 since it is smaller than output_dim_cat = 200
                # hence, the MLP will have the following structure:
                #   (cat_mlp): MLP(
                #       (dropout): Dropout(p=0.1, inplace=False)
                #       (activation): ReLU()
                #       (layers): ModuleList(
                #           (0): Linear(in_features=1238, out_features=619, bias=True)
                #           (1): Linear(in_features=619, out_features=309, bias=True)
                #           (2): Linear(in_features=309, out_features=200, bias=True)
                #       )
                #        (bn): ModuleList(
                #            (0): BatchNorm1d(619, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                #            (1): BatchNorm1d(309, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                #         )
                
                # the above strategy is computed by calc_mlp_dim() 
                # check the calc_mlp_dim() implementation in the file layer_utils.py for details
                
                dims = calc_mlp_dims(
                    self.cat_feat_dim, self.mlp_division, output_dim_cat
                )
                
                
                # after defining which will be the input and output dimensions of the layers in the MLP,
                # we can create the MLP that will process the categorial features
                # we use the MLP class implemented in the file layer_utils.py
                # check the MLP class implementation and complete the initialization of the MLP accordingly 
                # self.cat_mlp = MLP (..., return_layer_outs=False, bn=self.categorical_bn,)
                
                ### BEGIN SOLUTION 
                   
                # raise NotImplementedError()
                self.cat_mlp = MLP(
                    self.cat_feat_dim,
                    output_dim_cat,
                    act= self.mlp_act,
                    num_hidden_lyr=len(dims),
                    dropout_prob = self.mlp_dropout,
                    hidden_channels = dims,
                    return_layer_outs = False,
                    bn = self.categorical_bn
                )
  
                ### END SOLUTION
                
            # define the MLP to process also the numerical features and define the new dimension of numerical features
            output_dim_num = 0
            if self.numerical_feat_dim > 0:
                # we define the dimension of the MLP for the numerical features as:
                output_dim_num = self.numerical_feat_dim // (self.mlp_division // 2)

                
                # initialize the MLP: in this case, add only layer after the input layer
                # self.num_mlp = MLP (..., return_layer_outs=False, bn=self.categorical_bn,)
                ### BEGIN SOLUTION 
                   
                # raise NotImplementedError()
                self.num_mlp = MLP(
                    self.numerical_feat_dim,
                    output_dim_num,
                    act= self.mlp_act,
                    num_hidden_lyr = 1,
                    dropout_prob = self.mlp_dropout,
                    return_layer_outs = False,
                    bn = self.categorical_bn
                )
  
                ### END SOLUTION
            # finally, we have the new dimensions of the categorial features and numerical features
            # and we can define the dimension of the combined features when using text, categorial and numerical features
            # self.final_out_dim = ...
            
            ### BEGIN SOLUTION
            # raise NotImplementedError()
            self.final_out_dim = self.text_out_dim + output_dim_cat + output_dim_num
            ### END SOLUTION
 
        #7.3: attention_on_cat_and_numerical_feats approach
        #defining the weights matrices and the MLPs for projecting the tabular features to another dimensionality
        elif self.combine_feat_method == "attention_on_cat_and_numerical_feats":
            assert (
                self.cat_feat_dim + self.numerical_feat_dim != 0
            ), "should have some non-text features for this method"

            output_dim = self.text_out_dim
            print("output_dim: ", output_dim)
            if self.cat_feat_dim > 0:
                # if the dimesion of categorical features is bigger than the dimension of text features
                # project the categorical features to the dimensionality of text features
                if self.cat_feat_dim > self.text_out_dim:
                    output_dim_cat = self.text_out_dim
                    dims = calc_mlp_dims(
                        self.cat_feat_dim,
                        division=self.mlp_division,
                        output_dim=output_dim_cat,
                    )
                    self.cat_mlp = MLP(
                        self.cat_feat_dim,
                        output_dim_cat,
                        num_hidden_lyr=len(dims),
                        dropout_prob=self.mlp_dropout,
                        return_layer_outs=False,
                        hidden_channels=dims,
                        bn=self.categorical_bn,
                    )
                else:
                    output_dim_cat = self.cat_feat_dim
                self.weight_cat = nn.Parameter(torch.rand((output_dim_cat, output_dim)))
                self.bias_cat = nn.Parameter(torch.zeros(output_dim))

            if self.numerical_feat_dim > 0:
                if self.numerical_feat_dim > self.text_out_dim:
                    output_dim_num = self.text_out_dim
                    dims = calc_mlp_dims(
                        self.numerical_feat_dim,
                        division=self.mlp_division,
                        output_dim=output_dim_num,
                    )
                    self.num_mlp = MLP(
                        self.numerical_feat_dim,
                        output_dim_num,
                        num_hidden_lyr=len(dims),
                        dropout_prob=self.mlp_dropout,
                        return_layer_outs=False,
                        hidden_channels=dims,
                        bn=self.numerical_bn,
                    )
                else:
                    output_dim_num = self.numerical_feat_dim
                self.weight_num = nn.Parameter(torch.rand((output_dim_num, output_dim)))
                self.bias_num = nn.Parameter(torch.zeros(output_dim))

            self.weight_transformer = nn.Parameter(
                torch.rand(self.text_out_dim, output_dim)
            )
            self.weight_a = nn.Parameter(torch.rand((1, output_dim + output_dim)))
            self.bias_transformer = nn.Parameter(torch.rand(output_dim))
            self.bias = nn.Parameter(torch.zeros(output_dim))
            self.negative_slope = 0.2
            self.final_out_dim = output_dim
            
            print("self.weight_cat: ", self.weight_cat.shape)
            print("self.weight_num: ", self.weight_num.shape)
            print("self.weight_transformer: ", self.weight_transformer.shape)
            self.__reset_parameters()

        else:
            raise ValueError(
                f"combine_feat_method {self.combine_feat_method} " f"not implemented"
            )

    def forward(self, text_feats, cat_feats=None, numerical_feats=None):
        """
        Args:
            text_feats (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, text_out_dim)`):
                The tensor of text features. This is assumed to be the output from a HuggingFace transformer model
            cat_feats (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, cat_feat_dim)`, `optional`, defaults to :obj:`None`)):
                The tensor of categorical features
            numerical_feats (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, numerical_feat_dim)`, `optional`, defaults to :obj:`None`):
                The tensor of numerical features
        Returns:
            :obj:`torch.FloatTensor` of shape :obj:`(batch_size, final_out_dim)`:
                A tensor representing the combined features

        """
        if cat_feats is None:
            cat_feats = torch.zeros((text_feats.shape[0], 0)).to(text_feats.device)
        if numerical_feats is None:
            numerical_feats = torch.zeros((text_feats.shape[0], 0)).to(
                text_feats.device
            )

        # define the combined features when using only text features
        # reminder: check out the formula in the documentation
        # combined_feats = ...
        if self.combine_feat_method == "text_only":
            combined_feats = text_feats
 
        # 7.1: concat approach
        # define the combined features when using text, categorial and numerical features
        # reminder: check out the formula in the documentation       
        # combined_feats = ...
        
        if self.combine_feat_method == "concat":
            ### BEGIN SOLUTION
            # raise NotImplementedError()
            combined_feats = torch.cat((text_feats, cat_feats, numerical_feats), dim = 1)
            ### END SOLUTION

        # 7.2: individual_mlps_on_cat_and_numerical_feats_then_concat approach
        # define the combined features when using text, and categorial and numerical features with MLP
        # reminder: check out the formula in the documentation       
        # combined_feats = ...
        elif (
            self.combine_feat_method
            == "individual_mlps_on_cat_and_numerical_feats_then_concat"
        ):
            ### BEGIN SOLUTION
            # pass the categorical features through the MLP and get the new categorical features
            # cat_feats = ...
            if cat_feats.shape[1] != 0:
                # raise NotImplementedError()
                cat_feats = self.cat_mlp(cat_feats)
                
            # pass the categorical features through the MLP and get the new numerical features
            # numerical_feats = ...

            if numerical_feats.shape[1] != 0:
                # raise NotImplementedError()
                numerical_feats = self.num_mlp(numerical_feats)
            # combine the text features, new categorical features and new numerical features according to the formula provided
            # combined_feats = ...
            # raise NotImplementedError()
            combined_feats = torch.cat((text_feats, cat_feats, numerical_feats), dim = 1)
            ### END SOLUTION

        #7.3: attention_on_cat_and_numerical_feats approach
        elif self.combine_feat_method == "attention_on_cat_and_numerical_feats":
            # attention keyed by transformer text features
            
            # Note: you may follow the hints reported here, or you can implement the attention formula in your own style
            # hint: you can use torch.mm(input, mat2, *, out=None)
            # torch.mm: Performs a matrix multiplication of the matrices input and mat2.
            #           If input is a (n×m)(n×m) tensor, mat2 is a (m×p)(m×p) tensor, out will be a (n×p)(n×p) tensor.
            
            
            
            # do multiplication of W_x and x: w_text = ...
            # then perform concatenation and multiply by self.weight_a: g_text = ...
            # NOTE: pay attention to the shape of weights and features
            
            ### BEGIN SOLUTION
            # raise NotImplementedError()
            w_text = torch.mm(text_feats, self.weight_transformer)
            g_text = (torch.cat([w_text, w_text], dim=-1) * self.weight_a).sum(dim=1).unsqueeze(0).T

            
            ### END SOLUTION
            
            if cat_feats.shape[1] != 0:
                if self.cat_feat_dim > self.text_out_dim:
                ### BEGIN SOLUTION
                # check out __init__  for self.combine_feat_method == "attention_on_cat_and_numerical_feats"
                # project categorical features to the dimension specified in __init__
                # cat_feats = ...
                
                    # raise NotImplementedError()
                    cat_feats = self.cat_mlp(cat_feats)
                
                # do multiplication of W_c and c: w_cat = ...
                # then perform concatenation and multiply by self.weight_a: g_cat = ...
                # NOTE: pay attention to the shape of weights and features

                # raise NotImplementedError()
                w_cat = torch.mm(cat_feats, self.weight_cat)
                g_cat = (
                    (torch.cat([w_text, w_cat], dim=-1) * self.weight_a)
                    .sum(dim=1)
                    .unsqueeze(0)
                    .T
                )
 
                ### END SOLUTION
            else:
                w_cat = None
                g_cat = torch.zeros(0, device=g_text.device)
            


            if numerical_feats.shape[1] != 0:
                if self.numerical_feat_dim > self.text_out_dim:
                ### BEGIN SOLUTION
                # check out __init__  for self.combine_feat_method == "attention_on_cat_and_numerical_feats"
                # project numerical features to the dimension specified in __init__
                # numerical_feats = ...
                   # raise NotImplementedError()
                    numerical_feats = self.num_mlp(numerical_feats)
                 
                # do multiplication of W_n and n: w_num = ...
                # then perform concatenation and multiply by self.weight_a: g_num = ...
                # NOTE: pay attention to the shape of weights and features
                
                # raise NotImplementedError()
                w_num = torch.mm(numerical_feats, self.weight_num)
                g_num = (
                    (torch.cat([w_text, w_num], dim=-1) * self.weight_a)
                    .sum(dim=1)
                    .unsqueeze(0)
                    .T
                )

                ### END SOLUTION
            else:
                w_num = None
                g_num = torch.zeros(0, device=g_text.device)

            # put together g_text, g_cat, g_num
            # hint: you can use torhc.cat to obtain a tensor of dimension # N by 3
            # alpha = ...
            
            ### BEGIN SOLUTION
            # raise NotImplementedError()
            alpha = torch.cat([g_text, g_cat, g_num], dim=1) 
            ### END SOLUTION

            # apply leaky relu and softmax
            alpha = F.leaky_relu(alpha, 0.02)
            alpha = F.softmax(alpha, -1)
            
            ### BEGIN SOLUTION
            # raise NotImplementedError()
             ### END SOLUTION

            # compute the attention based summation, the result should be saved in combined_feats
            ### BEGIN SOLUTION
            # raise NotImplementedError()
            stack_tensors = [
                tensor for tensor in [w_text, w_cat, w_num] if tensor is not None
            ]
            combined = torch.stack(stack_tensors, dim=1)  # N by 3 by final_out_dim
            outputs_w_attention = alpha[:, :, None] * combined
            combined_feats = outputs_w_attention.sum(dim=1)  # N by final_out_dim
            ### END SOLUTION


        return combined_feats

    def __reset_parameters(self):
        glorot(self.weight_a)
        if hasattr(self, "weight_cat"):
            glorot(self.weight_cat)
            zeros(self.bias_cat)
        if hasattr(self, "weight_num"):
            glorot(self.weight_num)
            zeros(self.bias_num)
        glorot(self.weight_transformer)
        zeros(self.bias_transformer)
