import torch
import torch.nn as nn
import torch.nn.functional as F
from .vfe_template import VFETemplate
from  pcdet.models.backbones_3d.vfe.pillar_vfe import PFNLayer
# 从这里开始
class L2AttentionLayer(nn.Module):
    def __init__(self, channels):
        super(L2AttentionLayer, self).__init__()
        # 定义 Query 和 Key 的卷积层
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        # 共享 Query 和 Key 的权重
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias
        # 定义 Value 的卷积层
        self.v_conv = nn.Conv1d(channels, channels, 1)
        # 定义特征变换的卷积层和 BatchNorm 层
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()  # 激活函数

    def forward(self, x):
        # x是点云特征，xyz是点云坐标
        # 输入 x 的形状: [num_voxels, num_points, channels]
        # 将坐标信息与特征相加

        num_voxels, num_points, channels = x.size()

        # 调整输入形状以适应 Conv1d
        x = x.permute(0, 2, 1)  # [num_voxels, channels, num_points]

        # 计算 Query 和 Key
        x_q = self.q_conv(x).permute(0, 2, 1)  # [num_voxels, num_points, channels // 4]
        x_k = self.k_conv(x)  # [num_voxels, channels // 4, num_points]

        # 计算 L2 范数距离
        x_q_expanded = x_q.unsqueeze(2)  # [num_voxels, num_points, 1, channels // 4]
        x_k_expanded = x_k.unsqueeze(1)  # [num_voxels, 1, num_points, channels // 4]
        diff = x_q_expanded - x_k_expanded  # [num_voxels, num_points, num_points, channels // 4]
        l2_distance = torch.norm(diff, p=2, dim=-1)  # [num_voxels, num_points, num_points]

        # 计算注意力权重
        attention = F.softmax(-l2_distance, dim=-1)  # [num_voxels, num_points, num_points]

        # 计算 Value
        x_v = self.v_conv(x)  # [num_voxels, channels, num_points]

        # 对 Value 进行加权求和
        x_r = torch.einsum('bijk,bkj->bik', attention.unsqueeze(1), x_v)  # [num_voxels, channels, num_points]

        # 通过特征变换和残差连接
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r  # 残差连接

        # 恢复输出形状
        x = x.permute(0, 2, 1)  # [num_voxels, num_points, channels]

        return x

# Example usage
num_voxels = 2
num_points = 5
channels = 64

x = torch.rand(num_voxels, num_points, channels)


attention_layer = L2AttentionLayer(channels)
output = attention_layer(x)
print(output.shape)  # Expected: [num_voxels, num_points, channels]

class VoxelFeature_TA(nn.Module):
    def __init__(self,  channels=64):
        super(VoxelFeature_TA, self).__init__()

        # 定义 1D 卷积层和 BatchNorm 层
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(2*channels, channels, kernel_size=1, bias=False)
        self.pos_xyz = nn.Conv1d(3, channels, 1)  # 用于将坐标信息映射到特征空间
        self.bn1 = nn.BatchNorm1d(channels)
        # 定义四个自注意力层
        self.a1 = L2AttentionLayer(channels)
        self.a2 = L2AttentionLayer(channels)
        self.a3 = L2AttentionLayer(channels)
        self.a4 = L2AttentionLayer(channels)

    def forward(self, x):
        batch_size, _, N = x.size()
        # xyz = xyz.permute(0, 2, 1)  # 调整维度顺序：[batch_size, 3, num_points]
        # xyz = self.pos_xyz(xyz)  # 将坐标信息映射到特征空间
        # 通过第一个卷积层和 BatchNorm，激活函数为 ReLU
        x = F.relu(self.bn1(self.conv1(x)))  # 输出维度：[batch_size, channels, num_points]
        # 通过四个自注意力层
        x1 = self.a1(x)
        x2 = self.a2(x1)
        # x3 = self.a3(x2)
        # x4 = self.a4(x3)
        # 将四个自注意力层的输出拼接
        # x = torch.cat((x1, x2, x3, x4), dim=1)#64*4=256
        x = self.conv2(torch.cat((x1, x2), dim=1))  # 64*2=128 --> 64

        return x


class TAPillarVFE(VFETemplate):                       # PillarFeatureNet 增加了VoxelFeature_TA模块，作用是对points进行attention修正。
    # def __init__(self,
    #              num_input_features=4,
    #              use_norm=True,
    #              num_filters=(64,),
    #              with_distance=False,
    #              voxel_size=(0.16, 0.16, 4),       # []  0.2, 0.2, 4
    #              pc_range=(0, -39.68, -3, 69.12, 39.68, 1 )):  #     0, -40, -3, 70.4, 40, 1
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        """
        Pillar Feature Net with Tripe attention.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param with_distance: <bool>. Whether to include Euclidean distance to points.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """

        """
        带有三重注意力机制的柱状特征网络。
        该网络准备柱状特征并通过 PFNLayers 执行前向传播。该网络的作用类似于 SECOND 中的 second.pytorch.voxelnet.VoxelFeatureExtractor。
        :param num_input_features: <int>。输入特征的数量，可以是 x、y、z 或 x、y、z、r。
        :param use_norm: <bool>。是否包括批归一化。
        :param num_filters: (<int>: N)。N 个 PFNLayers 中每个的特征数量。
        :param with_distance: <bool>。是否包括点到点的欧几里得距离。
        :param voxel_size: (<float>: 3)。体素的尺寸，只使用 x 和 y 尺寸。
        :param pc_range: (<float>: 6)。点云范围，只使用 x 和 y 的最小值。
        """


        if self.with_distance:
            num_point_features += 1
        self.num_filters = self.model_cfg.NUM_FILTERS # 64
        assert len(self.num_filters) > 0

        num_input_features = 64 #cfg['TA']['BOOST_C_DIM']

        # Create PillarFeatureNet layers #num_filters 列表的第一个元素是输入特征的数量，后续元素是每个 PFNLayer 的特征数量
        num_filters = [num_input_features] + list(self.num_filters) #[64,64]
        # 实例化 VoxelFeature_TA 和 PFNLayers
        self.VoxelFeature_TA = VoxelFeature_TA()
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(PFNLayer(in_filters, out_filters, self.use_norm, last_layer=last_layer))
            # the if can also pfn_layers.append(
            #                 PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            #             )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        # 需要柱状（体素）尺寸和 x/y 偏移量以计算柱状偏移量   openpcdet is more!!!!!!!   zzzzzzzzzzzzzzzz!!!!!!!!!!!!!!!
        # self.vx = voxel_size[0]
        # self.vy = voxel_size[1]
        # self.x_offset = self.vx / 2 + pc_range[0]
        # self.y_offset = self.vy / 2 + pc_range[1]
        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0): # voxel_num_points, voxel_count, axis=0
        # 在指定的轴上扩展 actual_num 的维度
        actual_num = torch.unsqueeze(actual_num, axis + 1)

        # 创建一个形状与 actual_num 相同的列表，并在指定轴上设置为 -1
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1

        # 创建一个从 0 到 max_num-1 的张量，并将其形状调整为 max_num_shape
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)

        # 比较 actual_num 和 max_num，生成一个布尔张量，指示哪些位置的 actual_num 大于 max_num
        paddings_indicator = actual_num.int() > max_num

        # 返回布尔张量 paddings_indicator
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):

        # 从输入的 batch_dict 中获取体素特征、体素点数和体素坐标
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        # Find distance of x, y, and z from cluster center
        # 计算 x、y、z 到聚类中心
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        # 计算每个点云点到点云中心的距离
        f_cluster = voxel_features[:, :, :3] - points_mean

        # Find distance of x, y, and z from pillar center
        # 计算 x、y、z 到柱状中心的距离
        f_center = torch.zeros_like(voxel_features[:, :, :3])  # f_center = torch.zeros_like(features[:, :, :2])
        # f_center[:, :, 0] = features[:, :, 0] - (coors[:, 3].float().unsqueeze(1) * self.vx + self.x_offset)
        # f_center[:, :, 1] = features[:, :, 1] - (coors[:, 2].float().unsqueeze(1) * self.vy + self.y_offset)
        f_center[:, :, 0] = voxel_features[:, :, 0] - (
                    coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (
                    coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (
                    coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)
        # features[:, :, :3]：表示取 features 张量的所有元素，并且取每个元素的前三个值。这通常用于获取张量中的前三个特征，比如坐标的 x、y、z 值。
        # features[:, :, 3]：表示取 features 张量的所有元素，并且取每个元素的第四个值。这通常用于获取张量中的某个特定特征，比如某种属性的数值。

        # 在这个上下文中，features 可能是一个形状为 (batch_size, num_points, num_features) 的张量，其中 num_features 表示每个点的特征数量。
        # #因此，features[:, :, :3] 可能代表了每个点的坐标信息，而 features[:, :, 3] 可能代表了某种属性的数值。

        # Combine together feature decorations
        # 组合特征
        # features_ls = [features, f_cluster, f_center]
        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]  # 3,  3,  3
        # 否则，将体素特征中的通道维度之后的部分（通常是法向量或其他特征）、点云相对坐标和点云中心坐标拼接在一起作为最终特征
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
        # (num_voxels, max_points_per_voxel, 3+c) (num_voxels, max_points_per_voxel, 3) (num_voxels, max_points_per_voxel, 3)
        features = torch.cat(features, dim=-1)  # num_voxels, max_points_per_voxel ,9+c=10

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        # print(mask.shape) # [11635, 100]
        # print(mask)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features) #[11635, 100, 1]
        # print(mask.shape)
        features *= mask # torch.Size([11635, 100, 10])
        # print(features.shape)
        # 通过 VoxelFeature_TA 处理特征
        features = self.VoxelFeature_TA(points_mean, features) #torch.Size([11635, 100, 64])
        # print(features.shape) # 1111111111111111111111111111
        # Forward pass through PFNLayers
        # 通过 PFNLayers 进行前向传播
        for pfn in self.pfn_layers:
            features = pfn(features)  # torch.Size([11635, 1, 64])
        # print(features.shape)  # torch.Size([11635, 1, 64])

        features = features.squeeze(1)           #you xiang le yixia  yao fu zhi/ !!!!!    # features is squeezed in pointpillar_scatter.py by me
        batch_dict['pillar_features'] = features  # torch.Size([11635, 1 ,64]) hope to be torch.Size([11635, 64])
        # 将处理后的特征存储在 batch_dict 中，并返回
        # with open('/home/ubuntu/SWW/batch_dict.txt', 'w') as f:
        #     for key, value in batch_dict.items():
        #         if isinstance(value, torch.Tensor):
        #             info = f"{key}: {value.shape}\n"
        #         else:
        #             info = f"{key}: {type(value)}\n"
        #         # print(info.strip())
        #         f.write(info)
        #     f.write("\nFull data_dict:\n")
        #     f.write(str(batch_dict))

        return batch_dict

# class Pct(nn.Module):
#     def __init__(self, args, output_channels=40):
#         super(Pct, self).__init__()
#         self.args = args
#         # 定义两个 1D 卷积层和 BatchNorm 层
#         self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
#         self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm1d(64)
#         self.bn2 = nn.BatchNorm1d(64)
#
#         # 定义 Transformer 模块
#         self.pt_last = Point_Transformer_Last(args)
#         # 定义特征融合模块
#         self.conv_fuse = nn.Sequential(
#             nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
#             nn.BatchNorm1d(1024),
#             nn.LeakyReLU(negative_slope=0.2)
#         )
#         # 定义全连接层和 Dropout 层
#         self.linear1 = nn.Linear(1024, 512, bias=False)
#         self.bn6 = nn.BatchNorm1d(512)
#         self.dp1 = nn.Dropout(p=args.dropout)
#         self.linear2 = nn.Linear(512, 256)
#         self.bn7 = nn.BatchNorm1d(256)
#         self.dp2 = nn.Dropout(p=args.dropout)
#         self.linear3 = nn.Linear(256, 64)  # 输出分类结果
#
#     def forward(self, x):
#         xyz = x.permute(0, 2, 1)  # 调整维度顺序：[batch_size, 3, num_points]
#         batch_size, _, _ = x.size()
#         # 通过第一个卷积层和 BatchNorm，激活函数为 ReLU
#         x = F.relu(self.bn1(self.conv1(x)))  # 输出维度：[batch_size, 64, num_points]
#         # 通过第二个卷积层和 BatchNorm，激活函数为 ReLU
#         x = F.relu(self.bn2(self.conv2(x)))  # 输出维度：[batch_size, 64, num_points]
#         x = x.permute(0, 2, 1)  # 调整维度顺序：[batch_size, num_points, 64]
#
#
#         # 通过 Transformer 模块
#         x = self.pt_last(feature_1, new_xyz)
#         # 将 Transformer 输出与局部特征拼接
#         x = torch.cat([x, feature_1], dim=1)
#         # 通过特征融合模块
#         x = self.conv_fuse(x)
#         # 使用自适应最大池化将特征降维到 1
#         x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
#         # 通过全连接层和 Dropout 层
#         x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
#         x = self.dp1(x)
#         x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
#         x = self.dp2(x)
#         # 输出分类结果
#         x = self.linear3(x)
#         return x