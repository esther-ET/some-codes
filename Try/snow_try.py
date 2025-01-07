import torch
class Solution:
    def assign_target(self, gt_boxes_with_classes):
        batch_size = gt_boxes_with_classes.shape[0]
        gt_classes = gt_boxes_with_classes[:, :, -1]
        gt_boxes = gt_boxes_with_classes[:, :, :-1]
        for k in range(batch_size):
            cur_gt = gt_boxes[k]
            cnt = cur_gt.__len__() - 1
            a = cur_gt[cnt].sum()
            while cnt > 0 and a == 0:
                cnt -= 1
            cur_gt = cur_gt[:cnt + 1]
            print(cur_gt)
            cur_gt_classes = gt_classes[k][:cnt + 1]
            print(cur_gt_classes)
if __name__ == '__main__':
    s = Solution()
    print(s.assign_target(torch.randn(2,2,8)))