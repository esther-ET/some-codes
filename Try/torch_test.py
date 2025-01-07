import torch
import torch.nn as nn

# # 创建一个随机初始化的张量
# tensor = torch.randn(4)
#
# print("Original Tensor:")
# print(tensor)
#
# # 使用 torch.clamp() 将张量中的元素限制在 [0, 1] 范围内
# clamped_tensor = torch.clamp(tensor, min=0, max=1)
#
# print("Clamped Tensor:")
# print(clamped_tensor)

# 定义一个简单的模块
# class MyModule(nn.Module):
#     def __init__(self):
#         super(MyModule, self).__init__()
#         # 创建一个可训练的参数
#         self.my_param = nn.Parameter(torch.randn(1, requires_grad=True))
#
#     def forward(self, x):
#         # 在前向传播中使用这个参数
#         result = x + self.my_param
#         print("self.my_param:", self.my_param.data) # self.my_param: tensor([-0.3118])
#         print("x:", x) # x: tensor([-1.5320])
#         return result
#
# # 创建模块的实例
# module = MyModule()
# # 打印参数
# print("Initial parameter:", module.my_param.data) # Initial parameter: tensor([-0.3118])
# # 进行一些前向传播
# output = module(torch.randn(1))
# print(torch.randn(1))  #tensor([-0.5163])要注意，这个又是随机生成的新的了，不是前一句的那个了
#
# # 打印前向传播的结果
# print("Output:", output.data) # Output: tensor([-1.8439])
# # 调用优化器更新参数
# optimizer = torch.optim.SGD(module.parameters(), lr=0.1)  # .parameters()返回模块的参数!!!!!!
# optimizer.zero_grad()  # 梯度清零要看看！！！！！！！！！！！！！
# output.backward()
# optimizer.step()
# # 打印更新后的参数
# print("Updated parameter:", module.my_param.data) # Updated parameter: tensor([-0.4118])


import torch

# # 创建一个形状为 (2, 6) 的张量
# tensor = torch.arange(12).reshape(2, 6)
# print("Original Tensor:")
# print(tensor)
# tensor1 = torch.tensor([[1, 2], [3, 4]])
# tensor2 = torch.tensor([[5, 6], [7, 8]])
# tensor = torch.cat([tensor1, tensor2], dim=0)
# print(tensor.shape)

# 将张量沿第 1 维分割成 3 个张量
# chunks = tensor.chunk(3, dim=1)
#
# print("\nChunks:",chunks)
# for chunk in chunks:
#     print(chunk)
# offset = torch.cat([chunks[0], chunks[1]], dim=1)  #tensor([[0, 1],[6, 7]])
#                                                           #tensor([[2, 3],[8, 9]])
# print("\nOffset:",offset)                                 #tensor([[0, 1, 2, 3],[6, 7, 8, 9]])

# import torch
# tensor_3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
#
# # 沿各个维度重复
# repeated_tensor_3d = tensor_3d.repeat(2, 1, 3)  # 沿第0维重复2次，第1维重复1次，第2维重复3次
# print(repeated_tensor_3d)

# import torch
#
# # 假设 tensor 是一个包含边界框编码的张量
# tensor = torch.randn(100, 4)  # 例如，100个边界框，每个边界框有4个坐标
#
# # self.box_coder.code_size 可能是8，表示每个边界框有8个编码值
# code_size = 8
#
# # 使用 view() 方法重塑张量
# reshaped_tensor = tensor.view(-1, code_size)
#
# print(reshaped_tensor.shape)  # 输出: torch.Size([n, 8])，其中 n 是自动计算的批次大小


# features = torch.randn(2,3,5)
# print(features)
# print(features.permute(0, 2, 1).permute(0, 2, 1))
# num = torch.randn(2)
# print(num)
#
# points_mean = features[:, :, :].sum(dim=1, keepdim=False)
# print(points_mean.shape)
# normalizer = torch.clamp_min(num.view(-1, 1), min=1.0).type_as(features)
# points_mean = points_mean / normalizer #求出每个体素内 sum / 点数
# print(normalizer.shape)
# print(points_mean.shape)

# import numpy as np
#
# # 假设有一个浮点数数组
# array = np.array([1.2, 2.6, 3.7, 4.1])
#
# # 四舍五入到最接近的整数，然后转换为整数类型
# rounded_array = np.round(array).astype(np.int32)
#
# print(rounded_array)  # 输出: [1 3 4 4]

# num = torch.randn(5,1)
# shape1 = [1]*len(num.shape) #shape1=[1,1]
# print(shape1)
# shape1[1]=-1 #shape1=[1,-1]
# max1 = torch.arange(4, dtype=torch.float32).view(shape1)
# print(max1)#tensor([[0., 1., 2., 3., 4.]])
# print(max1.shape)#torch.Size([1, 5])
# print(num)#tensor([[ 1.6358],[-0.7542],[ 1.3567],[ 0.8044],[-0.4046]])
# print(num.shape) #torch.Size([5, 1])
# ind= num > max1
# print(ind)
# """
# tensor([[ True,  True, False, False, False],
#         [False, False, False, False, False],
#         [ True,  True, False, False, False],
#         [ True, False, False, False, False],
#         [False, False, False, False, False]])
# """
# print(ind.shape)#torch.Size([5, 5])

# 创建一个3D张量
# tensor_3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
# tensor_3d = torch.randn(2, 3)
# # 转置张量
# transposed_tensor_3d = tensor_3d.t()
#
# print(transposed_tensor_3d.shape)  # 输出转置后的形状
# print(transposed_tensor_3d)

tensor_2d = torch.randn(4,2,3,2)
print(tensor_2d)
sum_dim0 = tensor_2d.sum(dim=0)
print(sum_dim0.shape)
sum_dim1 = tensor_2d.sum(dim=1)
print(sum_dim1)

